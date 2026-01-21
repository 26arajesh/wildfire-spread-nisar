"""Generic Zarr stacking for any data type (fire, weather, satellite, static).

This script stacks per-day (or single) GeoTIFFs into a time-indexed Zarr store,
with optional reprojection/resampling to a template grid and band selection.

Works for:
  - Temporal data (fire, weather, satellite): stacks along time dimension
  - Static data (topography, landcover): stored once (no time dimension)

Usage examples:

  # Stack fire-mask TIFFs (temporal) to fire_masks.zarr
  python3 scripts/prep_stack_general.py \
    --input_dir src/data/CampFire_2018 \
    --out_zarr processed/fire_masks.zarr \
    --data_type temporal \
    --var_name fire_mask

  # Stack weather data with a template grid (reproject/resample)
  python3 scripts/prep_stack_general.py \
    --input_dir data/weather \
    --out_zarr processed/weather.zarr \
    --data_type temporal \
    --var_name weather \
    --template src/data/test/base.tif

  # Stack satellite bands, keep bands 1-3 (RGB)
  python3 scripts/prep_stack_general.py \
    --input_dir data/satellite \
    --out_zarr processed/satellite.zarr \
    --data_type temporal \
    --var_name satellite \
    --band_selection 1,2,3

  # Stack static data (elevation, no time dimension)
  python3 scripts/prep_stack_general.py \
    --input_dir data/elevation \
    --out_zarr processed/static.zarr \
    --data_type static \
    --var_name elevation

Requires: xarray, rioxarray, rasterio, zarr, numpy
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
    """List all .tif/.tiff files in input_dir, sorted."""
    exts = (".tif", ".tiff")
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if not f.startswith(".") and f.lower().endswith(exts)
    ]
    files.sort()
    return files


def extract_date_from_filename(fname: str) -> str:
    """Extract date (YYYY-MM-DD) from filename, or return basename without extension."""
    b = os.path.basename(fname)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", b)
    if m:
        return m.group(1)
    m = re.search(r"(\d{8})", b)
    if m:
        s = m.group(1)
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return os.path.splitext(b)[0]


def parse_band_selection(band_str: Optional[str]) -> Optional[List[int]]:
    """Parse band selection string (e.g., '1,2,3') into list of indices (0-based)."""
    if band_str is None:
        return None
    try:
        bands = [int(b.strip()) - 1 for b in band_str.split(",")]  # Convert to 0-based
        return bands
    except Exception:
        print(f"Warning: could not parse band_selection '{band_str}'; using all bands")
        return None


def build_temporal_zarr(
    input_dir: str,
    out_zarr: str,
    var_name: str,
    template: Optional[str] = None,
    band_selection: Optional[List[int]] = None,
    chunks: dict = None,
):
    """Stack temporal data (one TIFF per day) into time-indexed Zarr.

    Args:
        input_dir: Directory with per-day TIFFs.
        out_zarr: Output Zarr path.
        var_name: Variable name (e.g., 'fire_mask', 'weather', 'satellite').
        template: Optional template raster to reproject/resample to.
        band_selection: Optional list of band indices (0-based) to keep.
        chunks: Chunk sizes dict (e.g., {'time': 1, 'y': 256, 'x': 256}).
    """
    if xr is None:
        raise RuntimeError(
            "xarray and rioxarray are required. Install with: pip install xarray rioxarray rasterio zarr"
        )

    files = list_tiffs(input_dir)
    if not files:
        raise FileNotFoundError(f"No TIFFs found in {input_dir}")

    print(f"Found {len(files)} TIFFs. First: {files[0]}")

    datasets = []

    template_da = None
    if template:
        print(f"Using template for reprojection/resampling: {template}")
        template_da = rioxarray.open_rasterio(template, masked=True)

    for p in files:
        print(f"  Loading {os.path.basename(p)}")
        da = rioxarray.open_rasterio(p, masked=True)

        # Select bands if requested
        if band_selection is not None and "band" in da.dims:
            da = da.isel(band=band_selection)
        elif "band" in da.dims and da.sizes.get("band", 1) > 1:
            # Take first band if multiple and no selection requested
            da = da.isel(band=0)

        # Reproject to template grid if requested
        if template_da is not None:
            da = da.rio.reproject_match(template_da)

        # Convert to (y,x) or (y,x,band) by squeezing
        if "band" in da.dims and da.sizes.get("band", 1) == 1:
            da = da.squeeze("band", drop=True)

        # Add time coordinate
        t = extract_date_from_filename(p)
        da = da.expand_dims(time=[t])

        datasets.append(da)

    print("Concatenating along time dimension...")
    combined = xr.concat(datasets, dim="time")

    # Optional: cast to uint8 if values are 0-255 or 0-1
    try:
        combined = combined.fillna(0)
        if np.issubdtype(combined.dtype, np.floating):
            if combined.max() <= 1.0:
                combined = (combined > 0).astype("uint8")
            else:
                combined = combined.astype("uint8")
        elif not np.issubdtype(combined.dtype, np.integer):
            combined = combined.astype("float32")
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_zarr) or ".", exist_ok=True)
    print(f"Writing Zarr to {out_zarr} with variable '{var_name}'")

    # Apply chunking
    try:
        if chunks:
            combined = combined.chunk(chunks)
    except Exception:
        print("Warning: could not apply chunking; proceeding without explicit chunks")

    combined.to_dataset(name=var_name).to_zarr(out_zarr, consolidated=True)
    print("Done.")


def build_static_zarr(
    input_dir: str,
    out_zarr: str,
    var_name: str,
    template: Optional[str] = None,
    band_selection: Optional[List[int]] = None,
    chunks: dict = None,
):
    """Stack static (non-temporal) data into Zarr (no time dimension).

    Args:
        input_dir: Directory with TIFF files (typically just one or a few).
        out_zarr: Output Zarr path.
        var_name: Variable name (e.g., 'elevation', 'landcover').
        template: Optional template raster to reproject/resample to.
        band_selection: Optional list of band indices (0-based) to keep.
        chunks: Chunk sizes dict (e.g., {'y': 256, 'x': 256} for 2D).
    """
    if xr is None:
        raise RuntimeError(
            "xarray and rioxarray are required. Install with: pip install xarray rioxarray rasterio zarr"
        )

    files = list_tiffs(input_dir)
    if not files:
        raise FileNotFoundError(f"No TIFFs found in {input_dir}")

    print(f"Found {len(files)} TIFF(s). Stacking as static data (no time dimension).")

    # For static, typically use first file or stack all layers
    datasets = []

    template_da = None
    if template:
        print(f"Using template for reprojection/resampling: {template}")
        template_da = rioxarray.open_rasterio(template, masked=True)

    for p in files:
        print(f"  Loading {os.path.basename(p)}")
        da = rioxarray.open_rasterio(p, masked=True)

        # Select bands if requested
        if band_selection is not None and "band" in da.dims:
            da = da.isel(band=band_selection)

        # Reproject to template grid if requested
        if template_da is not None:
            da = da.rio.reproject_match(template_da)

        # Convert to (y,x) or (y,x,band) by squeezing
        if "band" in da.dims and da.sizes.get("band", 1) == 1:
            da = da.squeeze("band", drop=True)

        datasets.append(da)

    # Combine (stack along a new 'layer' dimension if multiple files, else use first)
    if len(datasets) == 1:
        combined = datasets[0]
    else:
        combined = xr.concat(datasets, dim="layer")

    # Optional: cast to appropriate dtype
    try:
        combined = combined.fillna(0)
        if np.issubdtype(combined.dtype, np.floating):
            combined = combined.astype("float32")
        elif not np.issubdtype(combined.dtype, np.integer):
            combined = combined.astype("float32")
    except Exception:
        pass

    os.makedirs(os.path.dirname(out_zarr) or ".", exist_ok=True)
    print(f"Writing Zarr to {out_zarr} with variable '{var_name}'")

    # Apply chunking
    try:
        if chunks:
            combined = combined.chunk(chunks)
    except Exception:
        print("Warning: could not apply chunking; proceeding without explicit chunks")

    combined.to_dataset(name=var_name).to_zarr(out_zarr, consolidated=True)
    print("Done.")


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Generic Zarr stacking for any data type (temporal or static)"
    )
    p.add_argument(
        "--input_dir",
        required=True,
        help="Directory with per-day (or single) TIFF files",
    )
    p.add_argument(
        "--out_zarr",
        required=True,
        help="Output Zarr path (e.g., processed/fire_masks.zarr)",
    )
    p.add_argument(
        "--data_type",
        choices=["temporal", "static"],
        default="temporal",
        help="Data type: 'temporal' (one file per time step) or 'static' (non-temporal)",
    )
    p.add_argument(
        "--var_name",
        required=True,
        help="Variable name in output (e.g., fire_mask, weather, elevation)",
    )
    p.add_argument(
        "--template",
        default=None,
        help="Optional template raster path to reproject/resample to",
    )
    p.add_argument(
        "--band_selection",
        default=None,
        help="Optional comma-separated band indices to keep (1-based, e.g., '1,2,3')",
    )
    p.add_argument("--chunk_y", type=int, default=256, help="Chunk size in Y (pixels)")
    p.add_argument("--chunk_x", type=int, default=256, help="Chunk size in X (pixels)")
    return p.parse_args()


def main():
    args = parse_args()

    # Parse band selection if provided
    band_selection = parse_band_selection(args.band_selection)

    # Determine chunking
    chunks = {"y": args.chunk_y, "x": args.chunk_x}
    if args.data_type == "temporal":
        chunks["time"] = 1

    # Route to appropriate builder
    if args.data_type == "temporal":
        build_temporal_zarr(
            args.input_dir,
            args.out_zarr,
            args.var_name,
            template=args.template,
            band_selection=band_selection,
            chunks=chunks,
        )
    else:
        build_static_zarr(
            args.input_dir,
            args.out_zarr,
            args.var_name,
            template=args.template,
            band_selection=band_selection,
            chunks=chunks,
        )


if __name__ == "__main__":
    main()
