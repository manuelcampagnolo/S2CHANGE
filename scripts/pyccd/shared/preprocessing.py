# -*- coding: utf-8 -*-
import xarray as xr
import rioxarray
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import geopandas as gpd
import os
import time
from shared.read_files import read_tif_files_theia, read_tif_files_gee
from shared.utils import get_largest_tif_by_pixels
import h5py
#%%
def clip_vector_mask_to_tile(vector_mask_path : str, reference_tif_path: str):
    """
    Clips the vector mask to the extent of the tile, provided a reference tif of that tile.

    Args:
        vector_mask_path (str) : path to the vector geometries to be used as mask for ccd processing.
        reference_tif_path (str) : path to a reference tif of the intended tile.
    
    Returns:
        geodataframe : vector geometries clipped to the tile extent.
    """

    #open vector mask
    vector_mask = gpd.read_file(vector_mask_path)

    #open reference tif belonging to the target tile
    with rasterio.open(reference_tif_path) as src:
        bounds = src.bounds
        crs = src.crs
    # Create a polygon from the bounding box
    polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    # Create a GeoDataFrame
    gdf_tile_extent = gpd.GeoDataFrame({"geometry": [polygon]}, crs=crs)

    #convert vector to same crs as the tif, if needed
    if vector_mask.crs.to_epsg() != crs.to_epsg():
        vector_mask = vector_mask.to_crs(crs.to_epsg())
    
    
    return gpd.clip(vector_mask, gdf_tile_extent)
#%%
def rasterize_vector_mask(clipped_vector_mask, reference_tif):
    """
    Converts a vector mask (polygon geometries) into a raster mask using the spatial properties 
    of a reference raster file.
    
    Args:
        clipped_vector_mask (GeoDataFrame): A GeoDataFrame containing polygon geometries to be rasterized.
        reference_tif (str): Path to the reference raster file, used to define spatial resolution, extent, and metadata.
    
    Returns:
        np.ndarray: A boolean 2D array where masked areas (polygons) are `True` and background areas are `False`.
    """
    #get metadata from the reference tif (transform, height, width etc)
    with rasterio.open(reference_tif) as src:
        meta = src.meta.copy()
        meta.update({"count":1}) #single band

    #create polygon's attribute to become raster value
    clipped_vector_mask["raster_value"] = 1 #binary raster (masked=1, unmasked=0)

    # Convert polygons to raster features
    shapes = [(geom, value) for geom, value in zip(clipped_vector_mask.geometry, clipped_vector_mask["raster_value"])]

    # Rasterize
    raster = rasterize(
        shapes=shapes,
        out_shape=(meta['height'], meta['width']),
        transform=meta['transform'],
        fill=0,  # Background value
        dtype="uint8"
    )
    
    return raster.astype(bool)#.flatten(order='F')
#%%
def getTimeSeriesForMask(tif_names, tif_dates_ord, bandas_desejadas, vector_mask_path, output_file):
    """
    Saves h5py files with time series values and location (x, y) of pixels inside a ROI (mask).
    It should be executed at the tile level.

    Args:
        tif_names : list of tif names (with path) of the target tile.
        tif_dates_ord : list of ordinal tif dates.
        bandas_desejadas: bands for which the time series will be collected.
        vector_mask_path : path to the vector file containing the region of interest/mask.
        output_file : base name (with path) of the npy file to which the information will be saved.

    Returns:
        N : number of pixels/points to be processed 
    """
    #identify a tif to be used as reference for determining tile extent
    ref_tif = get_largest_tif_by_pixels(tif_names)
    #clip vector geometries to tile extent
    clipped_vector_mask = clip_vector_mask_to_tile(vector_mask_path, ref_tif)
    #rasterize clipped vector
    rasterized = rasterize_vector_mask(clipped_vector_mask, ref_tif)

    #use xarray to handle time series
    time_var = xr.Variable('time',tif_dates_ord)
    # Load in and concatenate all individual GeoTIFFs
    tifs_xr = [rioxarray.open_rasterio(i, chunks={'x':-1, 'y':-1}) for i in tif_names] #old chunks={'x':-1, 'y':-1}
    geotiffs_da = xr.concat(tifs_xr, dim=time_var).sel(band=bandas_desejadas)#.astype('uint16')

    #stack x and y to allow selecting by index
    geotiffs_da_stacked = geotiffs_da.stack(pixel=('x','y'))
    #select by index
 
    total_selected_pixels = rasterized.sum()
    n_time = time_var.shape[0]
    n_bands = len(bandas_desejadas)

    mask_y, mask_x = np.where(np.flip(rasterized,0)) #we have to use np.flip because the y dimension is in inverse order in xarray (given crs 32629)

    x_sel = geotiffs_da.x.values[mask_x]
    y_sel = geotiffs_da.y.values[mask_y] #>>>>>>>>>>>> y is in different order

    with h5py.File(output_file, 'w') as hf:
        chunk_size = 1
        chunk_size = min(chunk_size, total_selected_pixels)

        data = hf.create_dataset(
            "values",
            shape=(n_time, n_bands, total_selected_pixels),
            dtype='uint16',
            compression='lzf', #supposed to be the best compression for fast read/write
            chunks=(n_time, n_bands, chunk_size) #experiment with chunks indicated this was the best setting for balancing creation time and read (access) time
        )
        xs = hf.create_dataset("xs", shape=(total_selected_pixels,), dtype='int32', compression='gzip', compression_opts=9)
        ys = hf.create_dataset("ys", shape=(total_selected_pixels,), dtype='int32', compression='gzip', compression_opts=9)

        #store coordinates only once
        xs[:] = x_sel
        ys[:] = y_sel

        ts = time.time()

        for t_idx in range(n_time):
            print("processing time step {}/{}".format(t_idx+1, n_time), end="\r")

            selection_values = geotiffs_da.isel(time=t_idx).fillna(65535).values[:, mask_y, mask_x]
        
            data[t_idx] = selection_values

        print('\nFinished h5 creation. Duration: {}min'.format(round((time.time()-ts)/60,2)))   

    return total_selected_pixels
#%%
def check_or_initialize_file(output_file, tiles, var, S2_tile, min_year, max_date, vector_mask_path, bandas_desejadas):
    """
    Checks for the existence of a specific h5py file and, depending on this verification, 
    performs different operations to process geospatial data.
    
    Args:
        - output_file (str): Path of the NumPy file to be checked and, if necessary, created.
        - tiles (str): Directory where the TIFF (raster) files are located.
        - var (str): Variable indicating the data source, which can be 'THEIA' or 'GEE'.
        - S2_tile (str): Identifier of the Sentinel-2 tile to be processed.
        - max_date (datetime.date): Maximum date limit for processing TIFF files. Only files with dates up to and including `max_date` will be considered.
        - vector_mask_path (GeoDataFrame): GeoDataFrame containing the geometries to be processed (polygons).
        - bandas_desejadas (list): List of desired bands for processing.
        - FOLDER_OUTPUTS (str): Directory where the results will be saved.
        - img_collection (str): Image collection to be used.
        - NODATA_VALUE (int): Value to be used to represent missing data.
        - raster_path (Path): Path of the raster file to be used for processing.
    
    Returns:
        - tif_dates_ord (list): List of dates in ordinal format.
        - N: Number of pixels/points to be processed.
    """
        
    if os.path.exists(output_file):
        print(f"The file '{output_file}' already exists. Loading and processing the existing data...")
        with h5py.File(output_file, 'r') as hf:
            N = hf["xs"].shape[0]
        if os.path.exists(output_file.parent / 'tif_dates_ord.npy'):
            print("Collecting dates from existing tif_dates_ord.npy file")
            tif_dates_ord = list(np.load(output_file.parent / 'tif_dates_ord.npy'))
        else:
            print('Collecting name and date of the TIFFs...')
            if var == 'THEIA':
                tif_names, tif_dates = read_tif_files_theia(S2_tile, tiles, min_year, max_date)
            else:
                tif_names, tif_dates = read_tif_files_gee(S2_tile, tiles, max_date)
            tif_names = [os.path.join(tiles, i) for i in tif_names]
            tif_dates_ord = [d.toordinal() for d in tif_dates]

    else:
        # If the numpy file does not exist, execute the entire initial process of creating the numpy file
        print(f"The file '{output_file}' does not exist. Starting processing...")
            
        print('Collecting name and date of the TIFFs...')
        if var == 'THEIA':
            tif_names, tif_dates = read_tif_files_theia(S2_tile, tiles, min_year, max_date)
        else:
            tif_names, tif_dates = read_tif_files_gee(S2_tile, tiles, max_date)
        tif_names = [os.path.join(tiles, i) for i in tif_names]
        tif_dates_ord = [d.toordinal() for d in tif_dates]

        # Save tif_dates_ord as a numpy file for future use
        np.save(output_file.parent / 'tif_dates_ord.npy', np.array(tif_dates_ord))
        
        print(f'Processing {var} data... ({tiles})')
        start_time = time.time()

        print('Opening TIFFs with xarray and loading the time series...')
        N = getTimeSeriesForMask(tif_names, tif_dates_ord, bandas_desejadas, vector_mask_path, output_file)
        
        end_time = time.time()
        execution_time_seconds = end_time - start_time
        execution_time_minutes = execution_time_seconds / 60
        print(f"Reading {var} data for a total of {N} pixels:", execution_time_minutes, "minutes")

    return tif_dates_ord, N
