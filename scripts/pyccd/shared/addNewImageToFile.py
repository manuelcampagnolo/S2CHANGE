import numpy as np
import xarray as xr
import rioxarray
import os
from notebooks.read_files import read_tif_files_theia, read_tif_files_gee, readPoints
from notebooks.processing import processar_centros_pixeis
import warnings
warnings.filterwarnings('ignore')
#%%
def addNewImageToFile(output_file, tiles, var, S2_tile, BDR_DGT, N, random_state_value, desired_bands, FOLDER_OUTPUTS, img_collection, NODATA_VALUE):
    """
    Loads the latest GeoTIFF, processes the geospatial data, and adds it to the existing numpy file.

    Args:
        - output_file (str): Path to the numpy file where data will be added.
        - tiles (str): Directory where the TIFF (raster) files are located.
        - var (str): Variable that indicates the data source, which can be 'THEIA' or 'GEE'.
        - S2_tile (str): Sentinel-2 tile identifier to be processed.
        - BDR_DGT (GeoDataFrame): GeoDataFrame containing the geometries to be processed.
        - N (int): Number of points to be processed.
        - random_state_value (int): Random number generator seed value.
        - desired_bands (list): List of desired bands for processing.
        - FOLDER_OUTPUTS (str): Directory where the results will be saved.
        - img_collection (str): Image collection to be used.
        - NODATA_VALUE (int): Value to represent missing data.

    Returns:
        - updated_data (ndarray): Numpy array updated with the new data.
    """
    
    # Process the centers of the points from each geometry to match the centers of the pixels from the rasters
    raster_path = tiles / 'Theia_T29TNE_20170813-112433.tif'
    gdf_pixel_centers = process_pixel_centers(BDR_DGT, raster_path)
    geospatial_data_meters = readPoints(gdf_pixel_centers, N, random_state_value)
    
    if var == 'THEIA':
        tif_names, tif_dates = read_tif_files_theia(S2_tile, tiles)
    else:
        tif_names, tif_dates = read_tif_files_gee(S2_tile, tiles)
    
    # Select the last tif and dates
    tif_names = [os.path.join(tiles, i) for i in tif_names][-1]
    tif_dates_ord = [d.toordinal() for d in tif_dates][-1]
    
    # Load the last GeoTIFF
    geotiff_da = rioxarray.open_rasterio(tif_names, chunks={'x': -1, 'y': -1}).sel(band=desired_bands)
    geotiff_da = geotiff_da.expand_dims('time').assign_coords(time=[tif_dates_ord])

    # X and Y coordinates of the chosen points
    points_x_int = xr.DataArray(np.round(geospatial_data_meters.geometry.x.values).astype('int'), dims=['location'])
    points_y_int = xr.DataArray(np.round(geospatial_data_meters.geometry.y.values).astype('int'), dims=['location'])

    selection = geotiff_da.sel(x=points_x_int, y=points_y_int, band=desired_bands)
    sel_values = selection.values

    # Load the existing numpy file if it exists
    if os.path.exists(output_file):
        try:
            existing_data = np.load(output_file)
            print("Existing file loaded successfully.")
        except FileNotFoundError:
            print(f"Error: The file '{output_file}.npy' was not found.")
            return None
    else:
        print(f"The file '{output_file}' does not exist. Creating a new file...")
        existing_data = np.array([], dtype=np.float64)  # Initialize an empty array if the file does not exist
    
    # Add the new data to the existing array
    updated_data = np.concatenate((existing_data, sel_values), axis=0)
    
    # Save the updated array back to the file
    np.save(output_file, updated_data)
    print("New image added and data saved successfully.")

    return updated_data

