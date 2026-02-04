#!/usr/bin/env python3
"""
===============================================================================
WILDFIRE DATA PROCESSING PIPELINE
===============================================================================

This script downloads and processes all data needed for wildfire prediction:
- Fire masks from Google Earth Engine (VIIRS)
- Weather data from GridMET
- SAR satellite data from ASF (NISAR or Sentinel-1)
- Topography from OpenTopography

Each section is clearly separated and modular for easy debugging.

Author: Aditya Rajesh
Date: January 2026
===============================================================================
"""

import os
import shutil
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")

# Core libraries
import numpy as np
import xarray as xr
import rioxarray
import rasterio
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds

# Check for optional libraries
LIBRARIES_AVAILABLE = {
    "ee": False,
    "geemap": False,
    "pygridmet": False,
    "asf_search": False,
    "bmi_topography": False,
}

try:
    import ee

    LIBRARIES_AVAILABLE["ee"] = True
except ImportError:
    pass

try:
    import geemap

    LIBRARIES_AVAILABLE["geemap"] = True
except ImportError:
    pass

try:
    import pygridmet

    LIBRARIES_AVAILABLE["pygridmet"] = True
except ImportError:
    pass

try:
    import asf_search as asf

    LIBRARIES_AVAILABLE["asf_search"] = True
except ImportError:
    pass

try:
    import requests

    LIBRARIES_AVAILABLE["requests"] = True
except ImportError:
    pass

# Optional: load credentials from .env (copy .env.example to .env and fill in)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration constants for the pipeline."""

    # Spatial reference system (UTM Zone 10N for California)
    TARGET_CRS = "EPSG:32610"

    # Grid resolution in meters
    TARGET_RESOLUTION = 500

    # Buffer around fire in meters (to capture spread areas)
    BUFFER_DISTANCE = 5000

    # NISAR availability date (when it started producing science data)
    NISAR_START_DATE = datetime(2025, 11, 1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def check_library(library_name: str, purpose: str) -> bool:
    """
    Check if a library is available and log appropriate message.

    Args:
        library_name: Name of the library (e.g., 'ee', 'pygridmet')
        purpose: What the library is used for

    Returns:
        True if available, False otherwise
    """
    if LIBRARIES_AVAILABLE.get(library_name, False):
        logger.info(f"✓ {library_name} available for {purpose}")
        return True
    else:
        logger.warning(f"✗ {library_name} not available for {purpose}")
        logger.warning(f"  Install with: pip install {library_name}")
        return False


def create_date_list(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Create a list of dates between start and end (inclusive).

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of datetime objects
    """
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def calculate_grid_parameters(
    bounds_wgs84: Tuple[float, float, float, float],
    target_crs: str,
    resolution: int,
    buffer: int,
) -> Tuple[Tuple[float, float, float, float], Tuple[int, int], object]:
    """
    Calculate grid parameters for processing.

    Args:
        bounds_wgs84: (west, south, east, north) in WGS84
        target_crs: Target coordinate reference system
        resolution: Grid resolution in meters
        buffer: Buffer distance in meters

    Returns:
        Tuple of (buffered_bounds, grid_shape, grid_transform)
    """
    west, south, east, north = bounds_wgs84

    # Transform bounds to target CRS
    bounds_target = transform_bounds("EPSG:4326", target_crs, west, south, east, north)

    # Apply buffer
    minx, miny, maxx, maxy = bounds_target
    buffered_bounds = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

    # Calculate grid dimensions
    width = int((buffered_bounds[2] - buffered_bounds[0]) / resolution)
    height = int((buffered_bounds[3] - buffered_bounds[1]) / resolution)

    grid_shape = (height, width)
    grid_transform = from_bounds(*buffered_bounds, width, height)

    return buffered_bounds, grid_shape, grid_transform


# ============================================================================
# FIRE MASK PROCESSING (Google Earth Engine)
# ============================================================================


def download_fire_masks_gee(
    bounds_wgs84: Tuple[float, float, float, float],
    date_list: List[datetime],
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
) -> bool:
    """
    Download daily fire masks from Google Earth Engine VIIRS data.

    This function uses Google Earth Engine to download VIIRS active fire
    detections for each day in the date range.

    Args:
        bounds_wgs84: Bounding box in WGS84 (west, south, east, north)
        date_list: List of dates to process
        grid_shape: Target grid shape (height, width)
        grid_transform: Rasterio transform for target grid
        target_crs: Target coordinate reference system
        output_path: Path to save fire_masks.zarr

    Returns:
        True if successful, False otherwise

    """
    logger.info("Processing fire masks from Google Earth Engine...")

    # Check if libraries are available
    if not check_library("ee", "Google Earth Engine"):
        logger.error("Cannot process fire masks without Earth Engine")
        return False

    if not check_library("geemap", "direct GEE downloads"):
        logger.warning("geemap not available - using placeholder")
        logger.warning("Install geemap for actual downloads: pip install geemap")

    try:
        # Initialize Earth Engine
        ee.Initialize()
        logger.info("✓ Earth Engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {e}")
        logger.error("Run: earthengine authenticate")
        return False

    # Create region of interest
    west, south, east, north = bounds_wgs84
    roi = ee.Geometry.Rectangle([west, south, east, north])

    # Load VIIRS fire dataset
    # Dataset: FIRMS (Fire Information for Resource Management System)
    viirs = ee.ImageCollection("FIRMS").filterBounds(roi)

    fire_masks = []

    for date in date_list:
        date_str = date.strftime("%Y-%m-%d")
        next_date_str = (date + timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info(f"  Processing {date_str}...")

        # Filter to single day
        daily_fires = viirs.filterDate(date_str, next_date_str)

        # Create binary mask (fire detected = 1, no fire = 0)
        fire_mask = daily_fires.select("T21").mosaic().gt(0).unmask(0)

        if LIBRARIES_AVAILABLE["geemap"]:
            import geemap

            fire_array = geemap.ee_to_numpy(fire_mask, region=roi, scale=500)
            # ee_to_numpy returns (height, width) or (height, width, bands); ensure 2D
            if fire_array.ndim > 2:
                fire_array = np.squeeze(fire_array)
            if fire_array.ndim != 2:
                logger.warning(
                    f"  Unexpected shape {fire_array.shape} for {date_str}, using placeholder"
                )
                fire_array = np.zeros(grid_shape, dtype=np.uint8)
            elif fire_array.shape != grid_shape:
                logger.warning(
                    f"  GEE array shape {fire_array.shape} != grid {grid_shape} for {date_str}; using placeholder"
                )
                fire_array = np.zeros(grid_shape, dtype=np.uint8)
            fire_masks.append(fire_array)
        else:
            fire_masks.append(np.zeros(grid_shape, dtype=np.uint8))

    if not fire_masks:
        logger.error("No fire mask data collected")
        return False

    # Stack into xarray DataArray
    fire_da = xr.DataArray(
        np.stack(fire_masks),
        dims=["time", "y", "x"],
        coords={
            "time": [d.strftime("%Y-%m-%d") for d in date_list],
            "y": np.arange(grid_shape[0]),
            "x": np.arange(grid_shape[1]),
        },
    )

    # Add spatial reference
    fire_da.rio.write_crs(target_crs, inplace=True)
    fire_da.rio.write_transform(grid_transform, inplace=True)

    # Save to zarr
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fire_da.to_dataset(name="fire_mask").to_zarr(
        output_path, mode="w", consolidated=True
    )

    logger.info(f"✓ Fire masks saved to {output_path}")
    logger.info(f"  Shape: {fire_da.shape} (time, y, x)")

    return True


# ============================================================================
# WEATHER DATA PROCESSING (GridMET)
# ============================================================================


def download_weather_gridmet(
    bounds_wgs84: Tuple[float, float, float, float],
    start_date: datetime,
    end_date: datetime,
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
) -> bool:
    """
    Download weather data from GridMET.

    GridMET provides daily meteorological data at 4km resolution for the
    contiguous United States.

    Args:
        bounds_wgs84: Bounding box in WGS84
        start_date: Start date
        end_date: End date
        grid_shape: Target grid shape
        grid_transform: Rasterio transform
        target_crs: Target CRS
        output_path: Path to save weather.zarr

    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading weather data from GridMET...")

    if not check_library("pygridmet", "weather data"):
        return False

    # Create bounding box geometry
    from shapely.geometry import box

    west, south, east, north = bounds_wgs84
    geometry = box(west, south, east, north)

    # Variables to download
    variables = [
        "pr",  # Precipitation (mm)
        "tmmn",  # Minimum temperature (K)
        "tmmx",  # Maximum temperature (K)
        "vs",  # Wind speed (m/s)
        "th",  # Wind direction (degrees)
        "rmax",  # Maximum relative humidity (%)
        "rmin",  # Minimum relative humidity (%)
    ]

    try:
        # Format dates for API
        dates = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        logger.info(f"  Requesting {len(variables)} variables...")
        logger.info(f"  Date range: {dates[0]} to {dates[1]}")

        # Download data
        weather_data = pygridmet.get_bygeom(
            geometry,
            dates,
            variables=variables,
            snow=False,  # Don't need snow parameters
        )

        logger.info(f"  Downloaded {len(weather_data.data_vars)} variables")

        # PyGridMET returns xarray with dims (time, lat, lon). Set spatial dims for rioxarray.
        if "lat" in weather_data.dims and "lon" in weather_data.dims:
            weather_data = weather_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        weather_data = weather_data.rio.write_crs("EPSG:4326")

        # Reproject to target CRS and resample to target grid
        weather_data = weather_data.rio.reproject(
            target_crs,
            shape=grid_shape,
            transform=grid_transform,
            resampling=rasterio.enums.Resampling.bilinear,
        )

        # Save to zarr
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weather_data.to_zarr(output_path, mode="w", consolidated=True)

        logger.info(f"✓ Weather data saved to {output_path}")
        logger.info(f"  Variables: {list(weather_data.data_vars)}")
        logger.info(f"  Shape: {dict(weather_data.dims)}")

        return True

    except Exception as e:
        logger.error(f"Failed to download weather data: {e}")
        logger.exception("Full traceback:")
        return False


# ============================================================================
# SATELLITE DATA PROCESSING (ASF - NISAR/Sentinel-1)
# ============================================================================


def should_use_nisar(start_date: datetime) -> bool:
    """
    Determine if NISAR data should be used based on date.

    NISAR became operational in November 2025. For earlier dates,
    use Sentinel-1 instead.

    Args:
        start_date: Start date of fire event

    Returns:
        True if NISAR should be used, False for Sentinel-1
    """
    return start_date >= Config.NISAR_START_DATE


def download_sar_data_asf(
    bounds_wgs84: Tuple[float, float, float, float],
    start_date: datetime,
    end_date: datetime,
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
) -> bool:
    """
    Download SAR satellite data from Alaska Satellite Facility (ASF).

    This function downloads ANALYSIS-READY RTC (Radiometric Terrain Corrected)
    products, which are already processed and ready to use.

    NISAR is used for fires from November 2025 onwards.
    Sentinel-1 is used for earlier fires.

    Args:
        bounds_wgs84: Bounding box in WGS84
        start_date: Start date
        end_date: End date
        grid_shape: Target grid shape
        grid_transform: Rasterio transform
        target_crs: Target CRS
        output_path: Path to save satellite.zarr
        earthdata_username: NASA Earthdata username
        earthdata_password: NASA Earthdata password

    Returns:
        True if successful, False otherwise

    NOTE: This downloads RTC products which are ALREADY PROCESSED.
    No additional SAR processing (calibration, terrain correction) needed!
    """
    logger.info("Downloading SAR satellite data from ASF...")

    if not check_library("asf_search", "SAR data"):
        return False

    # Check credentials
    username = earthdata_username or os.getenv("EARTHDATA_USERNAME")
    password = earthdata_password or os.getenv("EARTHDATA_PASSWORD")

    if not username or not password:
        logger.error("NASA Earthdata credentials not provided")
        logger.error("Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars")
        logger.error("Or pass as arguments")
        logger.error("Register at: https://urs.earthdata.nasa.gov/users/new")
        return False

    # Determine which satellite to use (ASF: dataset + processingLevel per docs)
    use_nisar = should_use_nisar(start_date)

    if use_nisar:
        logger.info("  Using NISAR data (available from Nov 2025+)")
        # NISAR product types: GSLC (Geocoded SLC), GCOV, etc. No RTC in ASF for NISAR yet.
        dataset = asf.DATASET.NISAR
        product_type = asf.PRODUCT_TYPE.GSLC  # Geocoded Single Look Complex
    else:
        logger.info("  Using Sentinel-1 data (NISAR not available for this date)")
        # OPERA RTC-S1: analysis-ready RTC from ASF. Use OPERA_S1 + RTC.
        dataset = asf.DATASET.OPERA_S1
        product_type = asf.PRODUCT_TYPE.RTC  # OPERA RTC for Sentinel-1

    # Create WKT polygon from bounds (ASF requires closed polygon: first vertex = last)
    west, south, east, north = bounds_wgs84
    wkt = f"POLYGON(({west} {south}, {east} {south}, {east} {north}, {west} {north}, {west} {south}))"

    # ASF search accepts start/end as strings (YYYY-MM-DD or natural language)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    temp_dir = output_path.parent / "temp_sar"

    try:
        # Search for SAR data (dataset= preferred over platform= per ASF docs)
        logger.info(f"  Searching ASF for {dataset} {product_type} products...")

        results = asf.search(
            dataset=dataset,
            processingLevel=product_type,
            intersectsWith=wkt,
            start=start_str,
            end=end_str,
        )

        logger.info(f"  Found {len(results)} SAR scenes")

        if len(results) == 0:
            logger.warning("  No SAR data found for this region/time")
            logger.info("  Creating placeholder satellite.zarr...")
            create_placeholder_sar(
                grid_shape, grid_transform, target_crs, output_path, start_date
            )
            return True

        # Authenticate (ASFSession.auth_with_creds per ASF docs)
        session = asf.ASFSession().auth_with_creds(username, password)

        # Download to temporary directory
        temp_dir.mkdir(exist_ok=True)

        logger.info(f"  Downloading {len(results)} scenes...")
        logger.info(f"  This may take a while (scenes are ~500MB each)")

        # Download files
        results.download(
            path=str(temp_dir), session=session, processes=2  # Parallel downloads
        )

        logger.info("  Download complete, processing RTC products...")

        # ================================================================
        # Process downloaded RTC GeoTIFFs
        # RTC products are ALREADY PROCESSED - just need to:
        # 1. Open the GeoTIFFs
        # 2. Reproject to target grid
        # 3. Stack into zarr
        # ================================================================

        sar_arrays = []
        dates = []

        # Find all downloaded GeoTIFFs
        rtc_files = list(temp_dir.glob("*.tif")) + list(temp_dir.glob("*.tiff"))

        if not rtc_files:
            logger.warning("  No GeoTIFF files found in download")
            create_placeholder_sar(
                grid_shape, grid_transform, target_crs, output_path, start_date
            )
            return True

        for rtc_file in rtc_files:
            logger.info(f"    Processing {rtc_file.name}...")

            # Open RTC GeoTIFF
            sar_data = rioxarray.open_rasterio(rtc_file, masked=True)

            # Reproject to target grid
            sar_data = sar_data.rio.reproject(
                target_crs,
                shape=grid_shape,
                transform=grid_transform,
                resampling=rasterio.enums.Resampling.bilinear,
            )

            # Extract date from filename (format varies by product)
            # This is simplified - actual parsing depends on file naming
            date_str = rtc_file.stem[:10]  # Assumes YYYY-MM-DD in filename

            sar_arrays.append(sar_data.values)
            dates.append(date_str)

        # Stack into xarray. Standardize to (time, y, x, band) per pipeline spec.
        sar_stacked = np.stack(sar_arrays)

        if sar_stacked.ndim == 4:  # (time, band, y, x) from rasterio
            sar_stacked = np.transpose(
                sar_stacked, (0, 2, 3, 1)
            )  # -> (time, y, x, band)
        # else (time, y, x): add band dim for consistency
        if sar_stacked.ndim == 3:
            sar_stacked = sar_stacked[:, :, :, np.newaxis]

        n_bands = sar_stacked.shape[-1]
        band_names = (
            ["VV", "VH"][:n_bands]
            if n_bands <= 2
            else [f"band_{i}" for i in range(n_bands)]
        )

        sar_da = xr.DataArray(
            sar_stacked,
            dims=["time", "y", "x", "band"],
            coords={
                "time": dates,
                "y": np.arange(grid_shape[0]),
                "x": np.arange(grid_shape[1]),
                "band": band_names,
            },
        )
        sar_da.rio.write_crs(target_crs, inplace=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sar_da.to_dataset(name="sar_backscatter").to_zarr(
            output_path, mode="w", consolidated=True
        )

        logger.info(f"✓ SAR data saved to {output_path}")
        logger.info(f"  Shape: {sar_da.shape} (time, y, x, band)")
        return True

    except Exception as e:
        logger.error(f"Failed to process SAR data: {e}")
        logger.exception("Full traceback:")
        return False

    finally:
        # Clean up downloaded SAR temp files
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("  Cleaned up temp_sar directory")
            except OSError as e:
                logger.warning(f"  Could not remove temp_sar: {e}")


def create_placeholder_sar(
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
    date: datetime,
):
    """Create placeholder SAR data when no real data is available."""
    logger.info("  Creating placeholder SAR data...")

    placeholder = xr.DataArray(
        np.zeros((1, grid_shape[0], grid_shape[1], 2)),
        dims=["time", "y", "x", "band"],
        coords={
            "time": [date.strftime("%Y-%m-%d")],
            "y": np.arange(grid_shape[0]),
            "x": np.arange(grid_shape[1]),
            "band": ["VV", "VH"],
        },
    )
    placeholder.rio.write_crs(target_crs, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder.to_dataset(name="sar_backscatter").to_zarr(
        output_path, mode="w", consolidated=True
    )


# ============================================================================
# STATIC TOPOGRAPHY PROCESSING (OpenTopography)
# ============================================================================


def download_topography(
    bounds_wgs84: Tuple[float, float, float, float],
    grid_shape: Tuple[int, int],
    grid_transform: object,
    target_crs: str,
    output_path: Path,
    api_key: Optional[str] = None,
) -> bool:
    """
    Download topography data from OpenTopography.

    Downloads DEM (Digital Elevation Model) and calculates slope and aspect.

    Args:
        bounds_wgs84: Bounding box in WGS84
        grid_shape: Target grid shape
        grid_transform: Rasterio transform
        target_crs: Target CRS
        output_path: Path to save static.zarr
        api_key: OpenTopography API key

    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading topography from OpenTopography...")

    # Check API key
    api_key = api_key or os.getenv("OPENTOPOGRAPHY_API_KEY")

    if not api_key:
        logger.error("OpenTopography API key not provided")
        logger.error(
            "Get free key at: https://portal.opentopography.org/requestService"
        )
        logger.error("Set OPENTOPOGRAPHY_API_KEY env var or pass as argument")
        return False

    west, south, east, north = bounds_wgs84

    try:
        # Use HTTP request to OpenTopography API
        if not check_library("requests", "HTTP requests"):
            return False

        import requests

        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            "demtype": "SRTMGL1",  # 30m SRTM DEM
            "south": south,
            "north": north,
            "west": west,
            "east": east,
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }

        logger.info("  Requesting DEM from OpenTopography...")
        response = requests.get(url, params=params, stream=True, timeout=300)
        response.raise_for_status()

        # Save to temporary file
        temp_dem = output_path.parent / "temp_dem.tif"
        temp_dem.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_dem, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("  DEM downloaded, processing...")

        # Open DEM
        dem = rioxarray.open_rasterio(temp_dem, masked=True)

        # Reproject to target grid
        dem = dem.rio.reproject(
            target_crs,
            shape=grid_shape,
            transform=grid_transform,
            resampling=rasterio.enums.Resampling.bilinear,
        )

        # Get elevation values
        elevation = dem.values[0]  # Remove band dimension

        # Calculate slope and aspect
        logger.info("  Calculating slope and aspect...")

        # Calculate gradients (rise over run)
        dy, dx = np.gradient(elevation, Config.TARGET_RESOLUTION)

        # Calculate slope (in degrees)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Calculate aspect (in degrees, 0=North, clockwise)
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = (aspect + 360) % 360

        # Create dataset
        static_ds = xr.Dataset(
            {
                "elevation": (["y", "x"], elevation),
                "slope": (["y", "x"], slope),
                "aspect": (["y", "x"], aspect),
            },
            coords={"y": np.arange(grid_shape[0]), "x": np.arange(grid_shape[1])},
        )

        # Add spatial reference
        static_ds.rio.write_crs(target_crs, inplace=True)
        static_ds.rio.write_transform(grid_transform, inplace=True)

        # Save to zarr
        output_path.parent.mkdir(parents=True, exist_ok=True)
        static_ds.to_zarr(output_path, mode="w", consolidated=True)

        logger.info(f"✓ Static topography saved to {output_path}")
        logger.info(f"  Layers: elevation, slope, aspect")
        logger.info(f"  Shape: {static_ds.dims}")

        # Clean up temp file
        if temp_dem.exists():
            temp_dem.unlink()

        return True

    except Exception as e:
        logger.error(f"Failed to process topography: {e}")
        logger.exception("Full traceback:")
        return False


# ============================================================================
# MAIN PROCESSING CLASS
# ============================================================================


class WildfireDataProcessor:
    """
    Main coordinator for wildfire data processing.

    This class orchestrates the entire pipeline, calling individual
    processing functions in sequence.
    """

    def __init__(
        self,
        fire_name: str,
        start_date: str,
        end_date: str,
        bounds: Tuple[float, float, float, float],
        output_dir: str,
        earthdata_username: Optional[str] = None,
        earthdata_password: Optional[str] = None,
        opentopo_api_key: Optional[str] = None,
    ):
        """
        Initialize the processor.

        Args:
            fire_name: Name of fire event
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            bounds: (west, south, east, north) in WGS84
            output_dir: Directory to save processed data
            earthdata_username: NASA Earthdata username
            earthdata_password: NASA Earthdata password
            opentopo_api_key: OpenTopography API key
        """
        self.fire_name = fire_name
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.bounds_wgs84 = bounds
        self.output_dir = Path(output_dir)

        # Store credentials
        self.earthdata_username = earthdata_username
        self.earthdata_password = earthdata_password
        self.opentopo_api_key = opentopo_api_key

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate grid parameters
        self.bounds_target, self.grid_shape, self.grid_transform = (
            calculate_grid_parameters(
                bounds,
                Config.TARGET_CRS,
                Config.TARGET_RESOLUTION,
                Config.BUFFER_DISTANCE,
            )
        )

        # Create date list
        self.date_list = create_date_list(self.start_date, self.end_date)

        logger.info("=" * 80)
        logger.info(f"Initialized processor for: {fire_name}")
        logger.info(
            f"Date range: {start_date} to {end_date} ({len(self.date_list)} days)"
        )
        logger.info(
            f"Grid: {self.grid_shape[1]} x {self.grid_shape[0]} pixels @ {Config.TARGET_RESOLUTION}m"
        )
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 80)

    def process_all(self):
        """
        Run the complete processing pipeline.

        Calls each processing function in sequence:
        1. Fire masks from GEE
        2. Weather from GridMET
        3. SAR from ASF
        4. Topography from OpenTopography
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE DATA PROCESSING")
        logger.info("=" * 80)

        results = {
            "fire_masks": False,
            "weather": False,
            "satellite": False,
            "static": False,
        }

        # Process fire masks
        logger.info("\n[1/4] FIRE MASKS")
        logger.info("-" * 80)
        try:
            results["fire_masks"] = download_fire_masks_gee(
                self.bounds_wgs84,
                self.date_list,
                self.grid_shape,
                self.grid_transform,
                Config.TARGET_CRS,
                self.output_dir / "fire_masks.zarr",
            )
        except Exception as e:
            logger.error(f"Fire mask processing failed: {e}")

        # Process weather
        logger.info("\n[2/4] WEATHER DATA")
        logger.info("-" * 80)
        try:
            results["weather"] = download_weather_gridmet(
                self.bounds_wgs84,
                self.start_date,
                self.end_date,
                self.grid_shape,
                self.grid_transform,
                Config.TARGET_CRS,
                self.output_dir / "weather.zarr",
            )
        except Exception as e:
            logger.error(f"Weather processing failed: {e}")

        # Process satellite
        logger.info("\n[3/4] SAR SATELLITE DATA")
        logger.info("-" * 80)
        try:
            results["satellite"] = download_sar_data_asf(
                self.bounds_wgs84,
                self.start_date,
                self.end_date,
                self.grid_shape,
                self.grid_transform,
                Config.TARGET_CRS,
                self.output_dir / "satellite.zarr",
                self.earthdata_username,
                self.earthdata_password,
            )
        except Exception as e:
            logger.error(f"Satellite processing failed: {e}")

        # Process static topography
        logger.info("\n[4/4] STATIC TOPOGRAPHY")
        logger.info("-" * 80)
        try:
            results["static"] = download_topography(
                self.bounds_wgs84,
                self.grid_shape,
                self.grid_transform,
                Config.TARGET_CRS,
                self.output_dir / "static.zarr",
                self.opentopo_api_key,
            )
        except Exception as e:
            logger.error(f"Topography processing failed: {e}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info("Results:")
        for component, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"  {component:15s}: {status}")
        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info("=" * 80)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main():
    """Main entry point for command-line usage."""

    parser = argparse.ArgumentParser(
        description="Automated wildfire data processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with environment variables for credentials
  python process_wildfire_data.py \\
      --fire-name "park_fire_2024" \\
      --start-date "2024-07-24" \\
      --end-date "2024-08-10" \\
      --bounds -122.0 39.5 -121.0 40.5 \\
      --output-dir processed/park_fire_2024

  # With explicit credentials
  python process_wildfire_data.py \\
      --fire-name "camp_fire_2018" \\
      --start-date "2018-11-08" \\
      --end-date "2018-11-25" \\
      --bounds -121.9 39.5 -121.0 40.0 \\
      --earthdata-username YOUR_USERNAME \\
      --earthdata-password YOUR_PASSWORD \\
      --opentopo-api-key YOUR_API_KEY \\
      --output-dir processed/camp_fire_2018

Environment Variables (recommended for credentials):
  EARTHDATA_USERNAME      NASA Earthdata username
  EARTHDATA_PASSWORD      NASA Earthdata password
  OPENTOPOGRAPHY_API_KEY  OpenTopography API key

Get credentials:
  - Earthdata: https://urs.earthdata.nasa.gov/users/new
  - OpenTopography: https://portal.opentopography.org/requestService
  - Earth Engine: Run 'earthengine authenticate'
        """,
    )

    # Required arguments
    parser.add_argument(
        "--fire-name", required=True, help='Name of fire event (e.g., "park_fire_2024")'
    )
    parser.add_argument(
        "--start-date", required=True, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date", required=True, help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--bounds",
        required=True,
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in WGS84: west south east north",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for processed data"
    )

    # Optional credentials
    parser.add_argument(
        "--earthdata-username",
        help="NASA Earthdata username (or set EARTHDATA_USERNAME)",
    )
    parser.add_argument(
        "--earthdata-password",
        help="NASA Earthdata password (or set EARTHDATA_PASSWORD)",
    )
    parser.add_argument(
        "--opentopo-api-key",
        help="OpenTopography API key (or set OPENTOPOGRAPHY_API_KEY)",
    )

    args = parser.parse_args()

    # Create and run processor
    processor = WildfireDataProcessor(
        fire_name=args.fire_name,
        start_date=args.start_date,
        end_date=args.end_date,
        bounds=tuple(args.bounds),
        output_dir=args.output_dir,
        earthdata_username=args.earthdata_username,
        earthdata_password=args.earthdata_password,
        opentopo_api_key=args.opentopo_api_key,
    )

    processor.process_all()


if __name__ == "__main__":
    main()
