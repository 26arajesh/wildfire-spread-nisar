"""Prepare Zarr stacks from per-day GeoTIFFs.

This script focuses on the fire-mask TIFFs in `src/data/CampFire_2018/` and
stacks them into a single time-indexed Zarr store (default: `processed/fire_masks.zarr`).

Usage examples:
  # Basic: use files as-is and stack by filename-derived date
  python3 scripts/prep_stack.py --input_dir src/data/CampFire_2018 --out_zarr processed/fire_masks.zarr

  # Provide a template raster to reproject/resample to common grid
  python3 scripts/prep_stack.py --input_dir src/data/CampFire_2018 --out_zarr processed/fire_masks.zarr --template src/data/test/base.tif

The script uses xarray + rioxarray when available. It extracts dates from
filenames using YYYY-MM-DD or YYYYMMDD patterns.
"""

import os
import re
from typing import List, Optional

try:
    import xarray as xr
    import rioxarray
    import numpy as np
except Exception:
    xr = None  # handled at runtime


def list_tiffs(input_dir: str) -> List[str]:
    exts = (".tif", ".tiff")
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if not f.startswith(".") and f.lower().endswith(exts)
    ]
    files.sort()
    return files


def extract_date_from_filename(fname: str) -> str:
    b = os.path.basename(fname)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", b)
    if m:
        return m.group(1)
    m = re.search(r"(\d{8})", b)
    if m:
        s = m.group(1)
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return os.path.splitext(b)[0]


def build_fire_zarr(
    input_dir: str, out_zarr: str, template: Optional[str] = None, chunks: dict = None
):
    if xr is None:
        raise RuntimeError(
            "xarray and rioxarray are required. Install with: pip install xarray rioxarray rasterio zarr"
        )

    files = list_tiffs(input_dir)
    if not files:
        raise FileNotFoundError(f"No TIFFs found in {input_dir}")

    print(f"Found {len(files)} TIFFs. First: {files[0]}")

    datasets = []
    times = []

    template_da = None
    if template:
        print(f"Using template for reprojection/resampling: {template}")
        template_da = rioxarray.open_rasterio(template, masked=True)

    for p in files:
        print("Loading", p)
        da = rioxarray.open_rasterio(p, masked=True)
        # Ensure 2D (bands x y x) -> if bands dimension present, take first band if appropriate
        if "band" in da.dims and da.sizes.get("band", 1) > 1:
            # try to find a FireMask description; else take band 1
            try:
                descs = da.attrs.get("descriptions") or da.attrs.get("long_name")
            except Exception:
                descs = None
            da = da.isel(band=0)

        # Reproject to template grid if requested
        if template_da is not None:
            da = da.rio.reproject_match(template_da)

        # Convert to single-band (y,x)
        if "band" in da.dims:
            da = da.squeeze("band", drop=True)

        # add time coordinate
        t = extract_date_from_filename(p)
        da = da.expand_dims(time=[t])

        datasets.append(da)
        times.append(t)

    print("Concatenating along time dimension...")
    combined = xr.concat(datasets, dim="time")

    # Optionally cast masks to uint8 for compactness
    try:
        combined = combined.fillna(0)
        if np.issubdtype(combined.dtype, np.floating):
            # threshold small floats to 0/1 if appropriate
            if combined.max() <= 1.0:
                combined = (combined > 0).astype("uint8")
            else:
                combined = combined.astype("uint8")
        else:
            combined = combined.astype("uint8")
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_zarr) or ".", exist_ok=True)
    print(f"Writing Zarr to {out_zarr} (this may take a while)")
    # Apply chunking via xarray.chunk before writing; to_zarr doesn't accept a
    # `chunks=` keyword in some xarray versions.
    try:
        if chunks:
            combined = combined.chunk(chunks)
    except Exception:
        print("Warning: could not apply chunking; proceeding without explicit chunks")

    combined.to_dataset(name="fire_mask").to_zarr(out_zarr, consolidated=True)
    print("Done.")


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Stack per-day fire mask TIFFs into a Zarr store"
    )
    p.add_argument(
        "--input_dir",
        default="src/data/CampFire_2018",
        help="Directory with per-day TIFFs",
    )
    p.add_argument(
        "--out_zarr", default="processed/fire_masks.zarr", help="Output Zarr path"
    )
    p.add_argument(
        "--template",
        default=None,
        help="Optional template raster path to reproject/resample to",
    )
    p.add_argument("--chunk_y", type=int, default=256, help="Chunk size in Y (pixels)")
    p.add_argument("--chunk_x", type=int, default=256, help="Chunk size in X (pixels)")
    return p.parse_args()


def main():
    args = parse_args()
    chunks = {"time": 1, "y": args.chunk_y, "x": args.chunk_x}
    build_fire_zarr(
        args.input_dir, args.out_zarr, template=args.template, chunks=chunks
    )


if __name__ == "__main__":
    main()
