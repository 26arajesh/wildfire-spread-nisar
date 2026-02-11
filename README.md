# wildfire-spread-nisar

Cleaned, pipeline-first repository for building wildfire event datasets.

## What this project does

Given one wildfire event (name, date range, bounds), `scripts/data_pipeline.py` builds:

- `fire_masks.zarr` (time, y, x)
- `weather.zarr` (time, y, x, channels)
- `satellite.zarr` (time, y, x, band)
- `static.zarr` (y, x, layers)

Each run writes these under a chosen output directory (usually `processed/<fire_name>`).

## Repository layout

- `scripts/data_pipeline.py` - main end-to-end pipeline
- `scripts/get_bounds.py` - helper to compute bbox from TIFF inputs
- `scripts/check_dependencies.py` - quick dependency check
- `src/data/CampFire_2018/` - sample local fire-mask TIFFs (reference/test data)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install earthengine-api geemap asf-search
```

3. Authenticate Earth Engine once:

```bash
earthengine authenticate
```

4. Add secrets to `.env` (not committed):

- `EARTHDATA_TOKEN`
- `OPENTOPOGRAPHY_API_KEY`

## Run (example)

```bash
python scripts/data_pipeline.py \
  --fire-name "camp_fire_2018" \
  --start-date "2018-11-08" \
  --end-date "2018-11-21" \
  --bounds -121.9 39.5 -121.0 40.0 \
  --output-dir processed/camp_fire_2018
```

## Notes

- Outputs are overwritten if you reuse the same `--output-dir`.
- ASF SAR download auth is token-only via `EARTHDATA_TOKEN`.
