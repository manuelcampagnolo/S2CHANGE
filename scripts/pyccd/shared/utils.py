from shared.read_files import read_tif_files_theia, read_tif_files_gee
import numpy as np
import rasterio
import datetime as dt
import os

def fromParamsReturnName(col_name, ccd_params, tifs_info, roi_name, min_year, max_date, output_folder):
    """
    Returns file name based on execution parameters.

    Args:
        col_name (str) :  image collection name (Theia, GEE).
        ccd_params (dict) : parameters found on ccd parameters.py file.
        tifs_info (tuple) : tile name and path of tifs in image collection (in the form of (tile_name, tile_path)).
        roi_name (str) : name to identify the ROI (polygons) used in the run.
        min_year (int) : first year of the time series (e.g. 2017).
        max_date (datetime) : datetime object corresponding to the last date of the time series.
        output_folder (str) : path to the numpy output folder, needed to bypass reading img collection in case a tif_dates_ord.npy file exists. 

    Returns:
        name : file name.

    NOTE: currently not implemented to return names of bands used for change detection and tmask.
    """

    #get chi
    chi = str(ccd_params['CHISQUAREPROB'])
    chi = chi[chi.find('.')+1:]
    #get minYears
    minYears = str(ccd_params['MIN_YEARS']).replace('.','')
    #get num obs
    n_obs = str(ccd_params['PEEK_SIZE'])
    #get lambda
    lam = str(ccd_params['ALPHA'])
    #get max iter
    maxIter = str(ccd_params['LASSO_MAX_ITER'])

    #get detection bands
    ### TODO (because using ndvi is currently a temporary solution - replacing blue)
    #get tmask bands
    ### TODO

    #get start and end dates
    tile_name, tile_path = tifs_info
    
    # Extract the dataset type from col_name
    suffix = col_name.split('_')[-1]
    
    # Load dataset and retrieve dates
    if suffix == 'THEIA':
        _ , dates = read_tif_files_theia(tile_name, tile_path, min_year, max_date)
    else:
        _ , dates = read_tif_files_gee(tile_name, tile_path, max_date)

    if len(dates) == 0:
        if os.path.exists(output_folder / 'tif_dates_ord.npy'):
            print("No images found in the {} folder".format(tile_path))
            print("    - Reading dates from existing tif_dates_ord.npy file")
            dates_ord = np.load(output_folder / 'tif_dates_ord.npy')
            dates = [dt.datetime.fromordinal(x) for x in dates_ord]
        else:
            raise Exception("No images found in the {} folder. Cannot return filename.".format(tile_path))

    start_date = min(dates).strftime("%Y%m%d")
    end_date = max(dates).strftime("%Y%m%d")

    name = "{0}-NDVI_XX{1}YM{2}NOBS{3}LDA{4}ITER{5}_START{6}_END{7}_ROI{8}".format(col_name, chi, minYears, n_obs, lam, maxIter, start_date, end_date, roi_name)

    return name

def getNumberOfPixelsFromNpy(npy_path):
    """
    Returns the number of pixels based on the npy file.

    Args:
        npy_path : path to npy file.
    
    Returns:
        number of pixels.
    """

    aux = np.load(str(npy_path.with_suffix('')) + '_xs.npy') #opens xs because it is lighter

    return aux.shape[0]


def get_largest_tif_by_pixels(tif_paths):
    """
    Given a list of GeoTIFF file paths, returns the path to the file with the largest number of pixels.

    Args:
        tif_paths (list of str): List of paths to GeoTIFF files.

    Returns:
        str: Path to the largest GeoTIFF file in terms of pixel count.
    """
    largest_tif = None
    max_pixels = 0

    for path in tif_paths:
        try:
            with rasterio.open(path) as src:
                pixel_count = src.width * src.height  # Total number of pixels

                if pixel_count > max_pixels:
                    max_pixels = pixel_count
                    largest_tif = path

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return largest_tif

def getStrDateFromOrdinal(dates_ord):
    """
    Given a list of ordinal dates, converts to strings in the format yyyymmdd.

    Args:
        dates_ord (list) : list of dates in ordinal (integers).

    Returns:

    """

    dates_str = [dt.datetime.fromordinal(x).strftime('%Y%m%d') for x in dates_ord]

    return dates_str