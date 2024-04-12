from read_files import read_tif_files


def fromParamsReturnName(col_name, ccd_params, tifs_info, n_sample, random_state_value):
    """
    Returns file name based on execution parameters:
    col_name: image collection name (Theia, GEE).
    ccd_params: parameters found on ccd parameters.py file.
    tifs_info: path and tile name of tifs in image collection (in the for of (S2_tiles, tiles)).
    n_sample: number of samples used when sampling input points.
    random_state_value: seed used for sampling input points.

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
    _ , tif_dates = read_tif_files(S2_tile,tiles)
    start_date = min(tif_dates).strftime("%Y%m%d")
    end_date = max(tif_dates).strftime("%Y%m%d")


    name = "{0}-NDVI_XX{1}YM{2}NOBS{3}LDA{4}ITER{5}_START{6}_END{7}_N{8}_RS{9}".format(col_name, chi, minYears, n_obs, lam, maxIter, start_date, end_date, n_sample, random_state_value)

    return name





