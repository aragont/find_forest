#!/usr/bin/env python3
"""
LAS to Forest GeoTIFF Converter

Processes LAS LiDAR files and classifies 5x5m grid cells as forest or non-forest
based on the "high vegetation" classification flag (LAS classification value 5).

Bounds file format (single line, in decimal degrees):
    lon_upper_left lat_upper_left lon_down_right lat_down_right

Where:
    upper_left  = north-west corner
    down_right  = south-east corner

Usage:
    python las2forest.py <las_directory> <bounds_file> <output_tiff>
"""

import sys
import numpy as np
from pathlib import Path

try:
    import laspy
except ImportError:
    print("Error: laspy library required. Install with: pip install laspy")
    sys.exit(1)
try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    print("Error: rasterio library required. Install with: pip install rasterio")
    sys.exit(1)
try:
    import utm
except ImportError:
    print("Error: utm library required. Install with: pip install utm")
    sys.exit(1)


def read_bounds_degrees(path: str) -> tuple:
    """
    Read bounding box in decimal degrees.

    Format: lon_ul lat_ul lon_dr lat_dr

    Returns: (lon_west, lat_north, lon_east, lat_south)
    """
    with open(path, 'r') as f:
        line = f.read().strip()
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Expected 4 values, got {len(parts)}")
        lon_ul, lat_ul, lon_dr, lat_dr = map(float, parts)

    # "Upper left" = NW corner, "down right" = SE corner
    lon_west = min(lon_ul, lon_dr)
    lon_east = max(lon_ul, lon_dr)
    lat_north = max(lat_ul, lat_dr)
    lat_south = min(lat_ul, lat_dr)

    return lon_west, lat_north, lon_east, lat_south


def determine_utm_zone(lon, lat) -> int:
    """Determine UTM zone number from longitude and latitude."""
    return int((lon + 180) / 6) + 1


def convert_to_utm(lon_w, lat_n, lon_e, lat_s):
    """
    Convert geographic bounds to UTM.

    Returns: (easting_min, northing_max, easting_max, northing_min, utm_epsg, zone_info)
    """
    # Use center of bounds to determine UTM zone
    lon_c = (lon_w + lon_e) / 2
    lat_c = (lat_n + lat_s) / 2

    zone_number = determine_utm_zone(lon_c, lat_c)
    northern = lat_c >= 0

    # Convert corners using the same UTM zone
    # utm.from_latlon returns (easting, northing, zone_number, zone_letter)
    e_nw, n_nw, _, _ = utm.from_latlon(lat_n, lon_w, force_zone_number=zone_number)
    e_se, n_se, _, _ = utm.from_latlon(lat_s, lon_e, force_zone_number=zone_number)
    e_ne, n_ne, _, _ = utm.from_latlon(lat_n, lon_e, force_zone_number=zone_number)
    e_sw, n_sw, _, _ = utm.from_latlon(lat_s, lon_w, force_zone_number=zone_number)

    easting_min = min(e_nw, e_ne, e_sw, e_se)
    easting_max = max(e_nw, e_ne, e_sw, e_se)
    northing_min = min(n_nw, n_ne, n_sw, n_se)
    northing_max = max(n_nw, n_ne, n_sw, n_se)

    # EPSG code: 326XX for north, 327XX for south
    utm_epsg = 32600 + zone_number if northern else 32700 + zone_number
    zone_info = f"UTM zone {zone_number}{'N' if northern else 'S'}"

    return easting_min, northing_max, easting_max, northing_min, utm_epsg, zone_info


def read_las_header_bbox(las_path: str) -> dict:
    """
    Read min/max coordinates directly from the LAS file header.

    Returns dict with keys: x_min, y_min, x_max, y_max
    or empty dict if reading fails.
    """
    try:
        with laspy.open(las_path) as f:
            header = f.header
            return {
                'x_min': header.mins[0],
                'y_min': header.mins[1],
                'x_max': header.maxs[0],
                'y_max': header.maxs[1],
            }
    except Exception:
        return {}


def bboxes_intersect(xmin1, ymin1, xmax1, ymax1,
                     xmin2, ymin2, xmax2, ymax2) -> bool:
    """Check if two bounding boxes overlap."""
    if xmax1 < xmin2 or xmax2 < xmin1:
        return False
    if ymax1 < ymin2 or ymax2 < ymin1:
        return False
    return True


def process_las_file(las_path: str,
                     easting_min: float, northing_min: float,
                     cell_size: float, nrows: int, ncols: int,
                     forest_count: np.ndarray, nonforest_count: np.ndarray,
                     point_count: np.ndarray):
    """
    Process a single LAS file and update count arrays.

    Convention: north up, east right.
      row 0 = northing_max (north), row increases → south
      col 0 = easting_min (west),  col increases → east
    """
    try:
        las = laspy.read(las_path)
    except Exception as e:
        print(f"Warning: Could not read {las_path}: {e}")
        return

    easting = las.x
    northing = las.y

    # Filter points within ROI
    mask = (easting >= easting_min) & \
           (easting < easting_min + ncols * cell_size) & \
           (northing >= northing_min) & \
           (northing < northing_min + nrows * cell_size)

    e_filt = easting[mask]
    n_filt = northing[mask]

    if len(e_filt) == 0:
        return

    # Column: easting → east (right)
    col_indices = ((e_filt - easting_min) / cell_size).astype(int)
    # Row: northing → south (down), row 0 = north
    northing_max = northing_min + nrows * cell_size
    row_indices = ((northing_max - n_filt) / cell_size).astype(int)

    col_indices = np.clip(col_indices, 0, ncols - 1)
    row_indices = np.clip(row_indices, 0, nrows - 1)

    # Classification 5 = High Vegetation (ASPRS standard)
    try:
        classification = las.classification
        is_high_vegetation = (classification[mask] == 5)
    except AttributeError:
        try:
            classification = las.raw_classification
            is_high_vegetation = (classification[mask] == 5)
        except AttributeError:
            print(f"Warning: No classification field in {las_path}, skipping")
            return

    veg_mask = is_high_vegetation
    if np.any(veg_mask):
        np.add.at(forest_count,
                  (row_indices[veg_mask], col_indices[veg_mask]), 1)

    nonveg_mask = ~veg_mask
    if np.any(nonveg_mask):
        np.add.at(nonforest_count,
                  (row_indices[nonveg_mask], col_indices[nonveg_mask]), 1)

    np.add.at(point_count, (row_indices, col_indices), 1)

    print(f"  Processed {las_path}: {len(e_filt)} points in ROI")


def main():
    if len(sys.argv) != 4:
        print("Usage: python las2forest.py <las_directory> <bounds_file> <output_tiff>")
        print("\nbounds_file format (decimal degrees):")
        print("  lon_upper_left lat_upper_left lon_down_right lat_down_right")
        sys.exit(1)

    las_dir = sys.argv[1]
    bounds_file = sys.argv[2]
    output_tiff = sys.argv[3]

    CELL_SIZE = 5.0  # meters

    # ---- Read and convert bounds ----
    print("Reading bounding box (degrees)...")
    lon_w, lat_n, lon_e, lat_s = read_bounds_degrees(bounds_file)
    print(f"  NW (upper-left):  lat={lat_n:.6f}, lon={lon_w:.6f}")
    print(f"  SE (down-right):  lat={lat_s:.6f}, lon={lon_e:.6f}")

    print("\nConverting to UTM...")
    e_min, n_max, e_max, n_min, utm_epsg, zone_info = convert_to_utm(
        lon_w, lat_n, lon_e, lat_s
    )
    print(f"  CRS: {zone_info} (EPSG:{utm_epsg})")
    print(f"  Easting:  {e_min:.2f} — {e_max:.2f}  ({e_max - e_min:.0f} m)")
    print(f"  Northing: {n_min:.2f} — {n_max:.2f}  ({n_max - n_min:.0f} m)")

    # ---- Grid dimensions ----
    width = e_max - e_min
    height = n_max - n_min

    ncols = int(np.ceil(width / CELL_SIZE))   # cols along easting (east)
    nrows = int(np.ceil(height / CELL_SIZE))  # rows along northing (south)

    print(f"\nGrid: {nrows} rows × {ncols} cols ({CELL_SIZE} m/cell)")
    print(f"Total cells: {nrows * ncols:,}")

    forest_count = np.zeros((nrows, ncols), dtype=np.float32)
    nonforest_count = np.zeros((nrows, ncols), dtype=np.float32)
    point_count = np.zeros((nrows, ncols), dtype=np.float32)

    # ---- Collect and filter LAS files ----
    las_files = list(Path(las_dir).glob('*.las')) + \
                list(Path(las_dir).glob('*.LAS'))
    las_files = list(set(las_files))

    if not las_files:
        print(f"Error: No LAS files found in {las_dir}")
        sys.exit(1)

    roi_xmin, roi_ymin = e_min, n_min
    roi_xmax, roi_ymax = e_max, n_max

    filtered = []
    skipped = 0

    for las_file in las_files:
        meta = read_las_header_bbox(str(las_file))
        if meta and all(k in meta for k in ('x_min', 'y_min', 'x_max', 'y_max')):
            if bboxes_intersect(roi_xmin, roi_ymin, roi_xmax, roi_ymax,
                                meta['x_min'], meta['y_min'],
                                meta['x_max'], meta['y_max']):
                filtered.append(las_file)
            else:
                skipped += 1
        else:
            filtered.append(las_file)

    filtered.sort()

    print(f"\nFound {len(las_files)} LAS file(s):")
    print(f"  Will process:               {len(filtered)}")
    if skipped:
        print(f"  Skipped (no bbox overlap): {skipped}")

    if not filtered:
        print("Error: No LAS files overlap with the region of interest.")
        sys.exit(1)

    # ---- Process files ----
    for i, las_file in enumerate(filtered, 1):
        print(f"[{i}/{len(filtered)}] Processing {las_file.name}...")
        process_las_file(
            str(las_file),
            easting_min=e_min, northing_min=n_min,
            cell_size=CELL_SIZE, nrows=nrows, ncols=ncols,
            forest_count=forest_count,
            nonforest_count=nonforest_count,
            point_count=point_count
        )

    # ---- Compute forest fraction ----
    print("\nComputing forest fraction...")
    output = np.zeros((nrows, ncols), dtype=np.uint8)

    has_data = point_count > 0
    total = forest_count + nonforest_count

    # Fraction of high-vegetation points among all points in the cell
    # Avoid division by zero — cells without data stay 0
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = np.where(has_data, forest_count / total, 0.0)

    # Map [0, 1] → [1, 255], 0 reserved for no-data
    #   fraction 0.0  → 1   (no forest)
    #   fraction 1.0  → 255 (100% forest)
    output[has_data] = (fraction[has_data] * 254 + 1).astype(np.uint8)

    n_forest = int(np.sum(output == 255))
    n_nonforest = int(np.sum(output == 1))
    n_partial = int(np.sum((output > 1) & (output < 255)))
    n_nodata = int(np.sum(output == 0))

    print(f"\nResults:")
    print(f"  100% forest (255): {n_forest:>10,} cells ({n_forest * 25:.0f} m²)")
    print(f"  Partial forest:    {n_partial:>10,} cells ({n_partial * 25:.0f} m²)")
    print(f"  No forest (1):     {n_nonforest:>10,} cells ({n_nonforest * 25:.0f} m²)")
    print(f"  No data (0):       {n_nodata:>10,} cells")
    print(f"  Total:             {n_forest + n_partial + n_nonforest + n_nodata:>10,} cells")

    # ---- Write GeoTIFF ----
    print(f"\nWriting GeoTIFF to {output_tiff}...")
    # from_origin(easting_west, northing_north, xres, yres)
    # col 0 → west, row 0 → north
    # col increases → east, row increases → south (north-up convention)
    transform = from_origin(e_min, n_max, CELL_SIZE, CELL_SIZE)

    with rasterio.open(
        output_tiff,
        'w',
        driver='GTiff',
        height=nrows,
        width=ncols,
        count=1,
        dtype=rasterio.uint8,
        crs=f'EPSG:{utm_epsg}',
        transform=transform,
        nodata=0,
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(output, 1)

    print("Done!")


if __name__ == '__main__':
    main()
