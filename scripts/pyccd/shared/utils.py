from notebooks.read_files import read_tif_files_theia, read_tif_files_gee
import numpy as np
from pathlib import Path

def fromParamsReturnName(col_name, ccd_params, tifs_info, roi_name, min_year, max_date):
    """
    Returns file name based on execution parameters.

    Args:
        col_name:  image collection name (Theia, GEE).
        ccd_params : parameters found on ccd parameters.py file.
        tifs_info : path and tile name of tifs in image collection (in the for of (S2_tiles, tiles)).
        roi_name : name to identify the ROI (polygons) used in the run.

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
    S2_tile, tiles = tifs_info
    
    # Extract the dataset type from col_name
    suffix = col_name.split('_')[-1]
    
    # Load dataset and retrieve dates
    if suffix == 'THEIA':
        _ , dates = read_tif_files_theia(S2_tile,tiles, min_year, max_date)
    else:
        _ , dates = read_tif_files_gee(S2_tile,tiles, max_date)
    
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
