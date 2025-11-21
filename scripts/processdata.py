import os
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from typing import Optional

import os
import re
from typing import Optional, List

import matplotlib.pyplot as plt


def list_vector_files(dirpath: str) -> List[str]:
    """Return list of vector files in dir (shp, geojson, gpkg)."""
    exts = (".shp", ".geojson", ".json", ".gpkg", ".geojsonl")
    files = []
    for entry in os.listdir(dirpath):
        if entry.startswith("."):
            continue
        lower = entry.lower()
        if any(lower.endswith(ext) for ext in exts):
            files.append(os.path.join(dirpath, entry))
    files.sort()
    return files


def list_tiff_files(dirpath: str) -> List[str]:
    """Return list of TIFF files in dir (.tif, .tiff)."""
    exts = (".tif", ".tiff")
    files = []
    for entry in os.listdir(dirpath):
        if entry.startswith("."):
            continue
        lower = entry.lower()
        if any(lower.endswith(ext) for ext in exts):
            files.append(os.path.join(dirpath, entry))
    files.sort()
    return files


def extract_date_from_filename(name: str) -> str:
    """Try to extract a date-like substring from filename for labelling."""
    # common patterns: YYYY-MM-DD, YYYYMMDD
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)
    m = re.search(r"(\d{8})", name)
    if m:
        s = m.group(1)
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return os.path.splitext(os.path.basename(name))[0]


def plot_perimeters(
    perim_files: List[str], base_ax=None, base_crs=None, save_path: Optional[str] = None
):
    try:
        import geopandas as gpd
    except Exception as e:
        raise RuntimeError(
            "geopandas is required to plot perimeters: pip install geopandas"
        ) from e

    n = len(perim_files)
    cmap = plt.get_cmap("viridis")
    fig = None
    ax = base_ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    for i, p in enumerate(perim_files):
        try:
            gdf = gpd.read_file(p)
            if base_crs is not None and gdf.crs != base_crs:
                gdf = gdf.to_crs(base_crs)
            label = extract_date_from_filename(p)
            color = cmap(i / max(1, n - 1))
            gdf.plot(ax=ax, facecolor="none", edgecolor=color, linewidth=1, label=label)
        except Exception as e:
            print(f"Failed to read/plot {p}: {e}")

    ax.legend(title="Perimeter (by file)", loc="upper right", fontsize="small")
    if save_path and fig is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print("Saved perimeter figure to:", save_path)

    return fig, ax


def plot_tiff_overlays(
    tiff_files: List[str],
    base_ax=None,
    alpha: float = 0.4,
    cmap_name: str = "Reds",
    save_path: Optional[str] = None,
):
    try:
        import rasterio
        from rasterio.plot import show
    except Exception as e:
        raise RuntimeError(
            "rasterio is required to plot TIFF overlays: pip install rasterio"
        ) from e

    n = len(tiff_files)
    cmap = plt.get_cmap(cmap_name)
    fig = None
    ax = base_ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    legend_patches = []
    for i, p in enumerate(tiff_files):
        try:
            with rasterio.open(p) as src:
                # choose a band that likely contains mask info
                band_index = 1
                # try to find a band named 'FireMask' if descriptions are present
                try:
                    descs = src.descriptions
                    if descs:
                        for j, d in enumerate(descs, start=1):
                            if d and "firemask" in d.lower():
                                band_index = j
                                break
                except Exception:
                    pass

                arr = src.read(band_index)
                # create binary mask: non-zero interpreted as perimeter
                mask = arr != 0

                color = cmap(i / max(1, n - 1))
                # rasterio.plot.show will place the image correctly when given the transform
                show(
                    mask.astype("uint8"),
                    transform=src.transform,
                    ax=ax,
                    cmap=cmap_name,
                    alpha=alpha,
                )
                import matplotlib.patches as mpatches

                legend_patches.append(
                    mpatches.Patch(color=color, label=extract_date_from_filename(p))
                )
        except Exception as e:
            print(f"Failed to read/plot {p}: {e}")

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            title="Perimeter (by file)",
            loc="upper right",
            fontsize="small",
        )

    if save_path and fig is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print("Saved TIFF overlay figure to:", save_path)

    return fig, ax

    def create_animation_from_tiffs(
        tiff_files: List[str],
        out_path: str,
        fps: int = 2,
        threshold: float = 0.5,
        raster_path: Optional[str] = None,
        cmap_name: str = "Reds",
    ):
        """Create an animation (GIF) from TIFF perimeter files.

        Each TIFF is read; non-zero values (or > threshold when float) are treated as mask.
        If `raster_path` is provided, the raster is used as background for each frame.
        """
        try:
            import imageio
        except Exception:
            raise RuntimeError(
                "imageio is required to create animations: pip install imageio"
            )

        try:
            import rasterio
            import numpy as np
            import matplotlib
            from matplotlib import cm
        except Exception as e:
            raise RuntimeError(
                "rasterio, numpy and matplotlib are required for animation"
            ) from e

        # Load base raster (as RGB) if requested
        base_rgb = None
        base_transform = None
        if raster_path:
            with rasterio.open(raster_path) as src:
                base_transform = src.transform
                if src.count >= 3:
                    r = src.read(1).astype(float)
                    g = src.read(2).astype(float)
                    b = src.read(3).astype(float)

                    # normalize for display
                    def norm(x):
                        x = x - x.min()
                        if x.max() > 0:
                            x = x / x.max()
                        return x

                    r, g, b = norm(r), norm(g), norm(b)
                    base_rgb = np.dstack([r, g, b])
                else:
                    g = src.read(1).astype(float)
                    g = g - g.min()
                    if g.max() > 0:
                        g = g / g.max()
                    base_rgb = np.dstack([g, g, g])

        frames = []
        cmap = cm.get_cmap(cmap_name)
        for p in tiff_files:
            with rasterio.open(p) as src:
                arr = src.read(1).astype(float)
                # normalize or threshold
                if arr.max() > 1.0:
                    mask = arr != 0
                else:
                    mask = arr > threshold

                # create colored mask
                mask_float = mask.astype(float)
                colored = cmap(mask_float)[:, :, :3]

                if base_rgb is None:
                    # simple colored background (white) with mask overlay
                    bg = np.ones_like(colored)
                    frame = np.where(mask[:, :, None], colored, bg)
                else:
                    # combine base_rgb and colored mask with alpha
                    alpha = 0.6
                    frame = (1 - alpha) * base_rgb + alpha * colored

                # convert to uint8
                frame_uint8 = (np.clip(frame, 0, 1) * 255).astype("uint8")
                frames.append(frame_uint8)

        # Write GIF
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"Saved animation to {out_path}")


def main(
    raster_path: Optional[str] = None,
    perim_dir: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Display raster (optional) and overlay perimeters from a directory.

    - If `perim_dir` is a directory, it will plot all vector files inside.
    - If `raster_path` is provided, raster will be shown as base.
    """
    base_crs = None
    fig = None
    ax = None

    if raster_path:
        if not os.path.exists(raster_path):
            raise FileNotFoundError(raster_path)
        try:
            import rasterio
            from rasterio.plot import show

            with rasterio.open(raster_path) as src:
                print("Raster CRS:", src.crs)
                print("Bounds:", src.bounds)
                print("Width x Height:", src.width, "x", src.height)
                fig, ax = plt.subplots(figsize=(10, 10))
                if src.count == 1:
                    show(src.read(1), ax=ax, cmap="gray")
                else:
                    show(src, ax=ax)
                base_crs = src.crs
        except Exception as e:
            print("Could not open raster:", e)

    if perim_dir:
        if not os.path.exists(perim_dir):
            raise FileNotFoundError(perim_dir)
        # Prefer TIFF files (one per day). If present, display them one-by-one.
        tiff_files = list_tiff_files(perim_dir)
        if tiff_files:
            import rasterio
            from rasterio.plot import show

            for p in tiff_files:
                try:
                    with rasterio.open(p) as src:
                        print("Showing:", p)
                        print("  CRS:", src.crs)
                        print("  Bounds:", src.bounds)
                        print("  Width x Height:", src.width, "x", src.height)
                        # choose band index (try to detect FireMask)
                        band_index = 1
                        try:
                            descs = src.descriptions
                            if descs:
                                for j, d in enumerate(descs, start=1):
                                    if d and "firemask" in d.lower():
                                        band_index = j
                                        break
                        except Exception:
                            pass

                        arr = src.read(band_index)
                        # If arr is boolean-like or 0/1, show mask; otherwise show array
                        is_mask = (
                            arr.dtype == "uint8"
                            or arr.dtype == "int8"
                            or arr.dtype == "int16"
                        ) or arr.max() <= 1
                        fig, ax = plt.subplots(figsize=(8, 8))
                        if is_mask:
                            show(arr != 0, transform=src.transform, ax=ax, cmap="Reds")
                        else:
                            show(arr, transform=src.transform, ax=ax, cmap="Reds")
                        ax.set_title(extract_date_from_filename(p))
                        plt.show()
                except Exception as e:
                    print(f"Failed to open/display {p}: {e}")
        else:
            files = list_vector_files(perim_dir)
            if not files:
                print("No perimeter vector files found in:", perim_dir)
            else:
                fig, ax = plot_perimeters(
                    files, base_ax=ax, base_crs=base_crs, save_path=save_path
                )

    if not raster_path and not perim_dir:
        print("No raster or perimeter directory provided. Nothing to show.")
        return

    plt.show()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Display raster and overlay daily perimeters from a directory"
    )
    p.add_argument(
        "--raster",
        default=None,
        help="Optional raster file path to use as background (GeoTIFF)",
    )
    p.add_argument(
        "--perim_dir",
        default="src/data/CampFire_2018",
        help="Directory containing per-day perimeter vector files (shp/geojson/gpkg). Default: src/data/CampFire_2018",
    )
    p.add_argument(
        "--save", default=None, help="Optional path to save the resulting figure (png)"
    )
    args = p.parse_args()
    main(args.raster, args.perim_dir, args.save)
