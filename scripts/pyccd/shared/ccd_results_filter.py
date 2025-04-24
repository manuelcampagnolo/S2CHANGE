import pandas as pd
import geopandas as gpd
import os
import glob
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

def filter_pixel_group(group):
    """
    Filter a group of rows for a single pixel according to the rules:
    - If only one row exists, keep it
    - If multiple rows exist, keep the row with highest tBreak where changeProb != 0
    - If no rows have changeProb != 0, keep the row with highest tBreak
    """
    if len(group) == 1:
        return group.iloc[0]
    else:
        valid_rows = group[group['changeProb'] != 0]
        if len(valid_rows) == 0:
            return group.loc[group['tBreak'].idxmax()]
        else:
            return valid_rows.loc[valid_rows['tBreak'].idxmax()]

def process_parquet_file(file_path):
    """
    Process a single parquet file and return filtered rows
    """
    try:
        df = pd.read_parquet(file_path)
        grouped = df.groupby(['x_coord', 'y_coord'])
        filtered_rows = []
        
        for (x, y), group in grouped:
            filtered_rows.append(filter_pixel_group(group))
        
        return filtered_rows
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def collect_data_from_directory(input_dir):
    """
    Collect and process data from all parquet files in a directory
    """
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return None
    
    print(f"Found {len(parquet_files)} parquet files to process")
    
    all_filtered_rows = []
    
    for i, file_path in enumerate(parquet_files, 1):
        print(f"Processing file {i}/{len(parquet_files)}: {os.path.basename(file_path)}")
        filtered_rows = process_parquet_file(file_path)
        all_filtered_rows.extend(filtered_rows)
    
    if not all_filtered_rows:
        print("No valid data found in any files")
        return None
    
    return pd.DataFrame(all_filtered_rows)

def add_break_date_column(df):
    """
    Add break_date column based on changeProb and tBreak values
    """
    df['break_date'] = df.apply(
        lambda row: row['tBreak'] if row['changeProb'] != 0 else None, # else 1 - to see pixels without break
        axis=1
    )
    return df

def create_geodataframe(df, source_crs="EPSG:32629"):
    """
    Create a GeoDataFrame from the DataFrame keeping it in UTM
    """
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.x_coord, df.y_coord),
        crs=source_crs
    )
    return gdf

def calculate_raster_parameters_utm(gdf):
    """
    Calculate raster dimensions and resolution from GeoDataFrame in UTM
    with fixed 10x10 meter resolution. Assumes coordinates are pixel centers.
    """
    # Assuming gdf is in UTM (EPSG:32629)
    min_x, min_y = gdf['x_coord'].min(), gdf['y_coord'].min()
    max_x, max_y = gdf['x_coord'].max(), gdf['y_coord'].max()
    
    # Fixed 10 meter resolution
    res_x = 10.0
    res_y = 10.0
    
    # Adjust bounds to account for pixel centers (extend by half pixel in each direction)
    min_x_corner = min_x - res_x / 2
    min_y_corner = min_y - res_y / 2
    max_x_corner = max_x + res_x / 2
    max_y_corner = max_y + res_y / 2
    
    # Calculate dimensions
    width = int(np.ceil((max_x_corner - min_x_corner) / res_x))
    height = int(np.ceil((max_y_corner - min_y_corner) / res_y))
    
    # Create transform (origin at top-left corner)
    transform = from_origin(min_x_corner, max_y_corner, res_x, res_y)
    
    return {
        'width': width,
        'height': height,
        'transform': transform,
        'resolution': (res_x, res_y),
        'bounds': (min_x_corner, min_y_corner, max_x_corner, max_y_corner)
    }

def create_raster_array_utm(gdf, raster_params):
    """
    Create a raster array from GeoDataFrame with fixed 10m resolution in UTM.
    Assumes coordinates are pixel centers.
    """
    width = raster_params['width']
    height = raster_params['height']
    min_x, min_y, max_x, max_y = raster_params['bounds']
    res_x, res_y = raster_params['resolution']
    
    break_date_array = np.full((height, width), np.nan, dtype=np.float32)
    
    for idx, row in gdf.iterrows():
        # Calculate indices from pixel center coordinates
        x_idx = int(np.round((row['x_coord'] - min_x) / res_x - 0.5))
        y_idx = int(np.round((max_y - row['y_coord']) / res_y - 0.5))
        
        if 0 <= x_idx < width and 0 <= y_idx < height:
            if row['break_date'] is not None:
                break_date_array[y_idx, x_idx] = row['break_date']
    
    return break_date_array

def save_geotiff(array, output_file, raster_params, source_crs='EPSG:32629', target_crs='EPSG:32629'):
    """
    Save a numpy array as a GeoTIFF file, reprojecting to target CRS
    """
    # If target CRS is different from source, reproject directly
    if source_crs != target_crs:
        # Create a temporary in-memory dataset first
        from rasterio.io import MemoryFile
        
        with MemoryFile() as memfile:
            with memfile.open(
                driver='GTiff',
                height=raster_params['height'],
                width=raster_params['width'],
                count=1,
                dtype=np.float32,
                crs=source_crs,
                transform=raster_params['transform'],
                nodata=np.nan
            ) as src:
                src.write(array, 1)
                
                # Calculate reprojection parameters
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds)
                
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                # Write directly to output file with reprojection
                with rasterio.open(output_file, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest)
    else:
        # If no reprojection needed, just save directly
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=raster_params['height'],
            width=raster_params['width'],
            count=1,
            dtype=np.float32,
            crs=source_crs,
            transform=raster_params['transform'],
            nodata=np.nan
        ) as dst:
            dst.write(array, 1)

def save_vector_points(gdf, output_file, target_crs="EPSG:32629"):
    """
    Save all points from the GeoDataFrame that have valid break dates as a vector file.
    """
    valid_points_gdf = gdf[gdf['break_date'].notna()].copy()

    # Convert break_date from milliseconds to date format
    if not valid_points_gdf.empty:
        # Assuming break_date is in milliseconds since epoch
        valid_points_gdf['break_date'] = pd.to_datetime(valid_points_gdf['break_date'], unit='ms').dt.strftime('%Y-%m-%d')
    
    # Reproject if necessary
    if valid_points_gdf.crs.to_string() != target_crs:
        valid_points_gdf = valid_points_gdf.to_crs(target_crs)
        
    valid_points_gdf.to_file(output_file, driver='ESRI Shapefile')
    
    return len(valid_points_gdf)

def process_directory_to_geotiff_improved(input_dir, output_raster_file, output_vector_file, target_crs="EPSG:32629"):
    """
    Main function to process all parquet files in a directory and save as a single GeoTIFF
    and a vector file of used points.
    Uses UTM coordinates throughout and only reprojects at the end if needed.
    """
    # Create output directories if they don't exist
    for output_file in [output_raster_file, output_vector_file]:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Collect data from all parquet files
    df = collect_data_from_directory(input_dir)
    if df is None:
        print("No data")
        return
    
    # Add break_date column
    df = add_break_date_column(df)
    
    # Create GeoDataFrame
    gdf = create_geodataframe(df)
    
    # Calculate raster parameters
    raster_params = calculate_raster_parameters_utm(gdf)
    
    print(f"Creating raster with dimensions: {raster_params['width']} x {raster_params['height']}")
    print(f"Resolution: {raster_params['resolution'][0]} x {raster_params['resolution'][1]} meters")
    
    # Create raster array
    break_date_array = create_raster_array_utm(gdf, raster_params)
    
    # Save to GeoTIFF (with optional reprojection)
    save_geotiff(break_date_array, output_raster_file, raster_params, source_crs='EPSG:32629', target_crs=target_crs)
    
    # Save vector points
    num_points_saved = save_vector_points(gdf, output_vector_file, target_crs)
    
    print(f"Combined GeoTIFF saved to: {output_raster_file}")
    print(f"Vector points saved to: {output_vector_file}")
    print(f"Total pixels processed: {len(df)}")
    print(f"Pixels with break dates: {df['break_date'].notna().sum()}")
    print(f"Points saved to vector file: {num_points_saved}")

if __name__ == "__main__":
    # Set input directory and output files
    input_directory = "/Users/domwelsh/green_ds/Thesis/BDR_300_artigo" # UPDATE
    output_raster_file = "/Users/domwelsh/green_ds/Thesis/BDR_300_artigo/accuracy_assessment/last_break_dates.tif" # UPDATE
    output_vector_file = "/Users/domwelsh/green_ds/Thesis/BDR_300_artigo/accuracy_assessment/used_points.shp" # UPDATE
    
    process_directory_to_geotiff_improved(input_directory, output_raster_file, output_vector_file) # target_crs='EPSG:4326'
