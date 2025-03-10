# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
import pandas as pd
import ccd
from shapely.geometry import Point
import geopandas as gpd
import os
import time
from shared.read_files import read_tif_files_theia, read_tif_files_gee
from shared.utils import get_largest_tif_by_pixels
from pyproj import CRS
#%%
def create_geodataframe_from_parquet(filename, epsg_input, epsg_output, S2_tile, parquet_dir, shapefile_dir):
    """
    Creates a GeoDataFrame from a Parquet file containing geographic coordinates, reprojects it,
    and adds a temporal break column (tBreak). It then saves the GeoDataFrame as a Shapefile
    in a dynamically created folder based on the S2_tile.

    Args:
        - filename (str): Name of the Parquet file without the extension.
        - epsg_input (int): EPSG code of the input spatial reference system.
        - epsg_output (int): EPSG code of the output spatial reference system.
        - S2_tile (str): Identifier of the S2 tile used to create the subfolder.
        - parquet_dir (Path): Path to the folder where the Parquet file is located.
        - shapefile_dir (Path): Path to the folder where the Shapefile will be saved.

    Returns:
        - GeoDataFrame: A GeoDataFrame with the reprojected geometry and the added tBreak column.
    """
    # Build the full path to the Parquet file  
    parquet_path = Path(parquet_dir) / f"{filename}.parquet"
    
    # Load the Parquet file into a pandas DataFrame
    df = pd.read_parquet(parquet_path)
    
    # Create a 'geometry' column with Point objects based on Lat and Lon
    geometry = [Point(lon, lat) for lon, lat in zip(df['Lon'], df['Lat'])]
    
    # Reproject to the specified EPSG coordinate system
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS.from_epsg(epsg_input))
    gdf = gdf.to_crs(epsg=epsg_output)
    
    # Add the tBreak column to the GeoDataFrame
    gdf['tBreak'] = df['tBreak']

    # Save the GeoDataFrame as a Shapefile
    shapefile_path = Path(shapefile_dir) / f"{filename}.shp"
    gdf.to_file(shapefile_path, driver='ESRI Shapefile')
    
    return gdf
#%%
def processPointData(args):
    """
    Processes the data of a specific point, including filtering out NODATA_VALUE and calculating the NDVI.

    Args:
        - i (int): Index of the point of interest.
        - sel_values (3D array): 3D array of selected values (number of images x number of bands x batch_size).
        - dates (ndarray): Array of dates.
        - xs (ndarray): Array of x coordinates.
        - ys (ndarray): Array of y coordinates.
        - NODATA_VALUE (float): Value representing missing data.
        - FOLDER_OUTPUTS (str): Directory to save the results.
        - img_collection (ndarray): Sentinel-2 image collection.

    Returns:
        - dates (ndarray): Array of filtered dates.
        - ndvis (ndarray): Array of NDVI values.
        - greens (ndarray): Array of green band values.
        - reds (ndarray): Array of red band values.
        - nirs (ndarray): Array of NIR band values.
        - swir2s (ndarray): Array of SWIR2 band values.
        - pixel (tuple): Coordinates of the pixel (x, y).
    """
    i, sel_values, dates, xs, ys, NODATA_VALUE, MAX_VALUE_NDVI, FOLDER_OUTPUTS, CRS_THEIA, CRS_WGS84, img_collection = args

    # Extract the point of interest
    ponto = sel_values[:, :, i]

    # Coordinates of the point
    ponto_desejado = xs[i], ys[i]

    # Combine dates and point values into a matrix
    ponto_with_dates = np.column_stack((dates, ponto[:, 0], ponto[:, 1:]))

    # Apply a mask to remove NODATA_VALUE
    mask = (ponto_with_dates != NODATA_VALUE).all(axis=1)
    ponto_with_dates_filtered = ponto_with_dates[mask].transpose()

    # Separate the bands and dates
    dates, greens, reds, nirs, swir2s = ponto_with_dates_filtered

    # Calculate NDVI
    ndvis = np.where((nirs + reds) > 0, MAX_VALUE_NDVI * (nirs - reds) / (nirs + reds), NODATA_VALUE)
    
    # Calcular o NBR
    # nbrs = np.where((nirs + swir2s) > 0, MAX_VALUE_NDVI * (nirs - swir2s) / (nirs + swir2s), NODATA_VALUE)
    
    # Create a new array with NDVI in position 1
    ponto_with_dates_updated = np.vstack((dates, ndvis, greens, reds, nirs, swir2s))
    # ponto_with_dates_updated = np.vstack((dates, ndvis, greens, reds, nirs, swir2s, nbrs))
    
    
    # Filter again to remove NODATA_VALUE
    ponto_with_dates_updated1 = ponto_with_dates_updated.transpose()
    ponto_with_dates_updated2 = ponto_with_dates_updated1[~np.any(ponto_with_dates_updated1 == NODATA_VALUE, axis=1)]
    ponto_with_dates_final = ponto_with_dates_updated2.transpose()
    

    # Separate the bands and dates again after filtering
    dates, ndvis, greens, reds, nirs, swir2s = ponto_with_dates_final
    # dates, ndvis, greens, reds, nirs, swir2s, nbrs = ponto_with_dates_final


    return dates, ndvis, greens, reds, nirs, swir2s, ponto_desejado, NODATA_VALUE
    #return dates, ndvis, greens, reds, nirs, swir2s, nbrs, ponto_desejado, NODATA_VALUE, CRS_THEIA, CRS_WGS84
#%%
def runDetectionForPoint(args):
    """
    Executes the CCD for a specific point.

    Args:
        - i (int): Index of the point of interest.
        - sel_values (ndarray): Array of selected values.
        - dates (ndarray): Array of dates.
        - xs (ndarray): Array of x coordinates.
        - ys (ndarray): Array of y coordinates.
        - NODATA_VALUE (float): Value representing missing data.
        - FOLDER_OUTPUTS (str): Directory to save the results.
        - img_collection (ndarray): Sentinel-2 image collection.

    Returns:
        - df (DataFrame): DataFrame with the CCD results.
    """
    # Process the point data
    dates, ndvis, greens, reds, nirs, swir2s, ponto_desejado, NODATA_VALUE = processPointData(args)

    # Execute change detection
    results = ccd.detect(dates, ndvis, greens, swir2s)
    # results = ccd.detect(dates, ndvis, greens, swir2s, nbrs)

    df = process_detection_results(results, ponto_desejado, NODATA_VALUE)

    return df
#%%
def process_detection_results(results, ponto_desejado, NODATA_VALUE):
    """
    Processes the change detection results for a specific point (1 pixel; all segments).

    Args:
        - results (dict): Change detection results, expected as a dictionary with 'change_models' and 'processing_mask'.
        - dates (ndarray): Array of dates.
        - ndvis (ndarray): Array of NDVI values.
        - desired_point (tuple): Coordinates of the desired point, tuple (x, y) 'crs': 'EPSG:32629'.
        - NODATA_VALUE (float): Value representing missing data.

    Returns:
        - df (DataFrame): DataFrame containing the processed change detection results.
    """
    #predicted_values = []
    #prediction_dates = []
    break_dates = []
    start_dates = []
    end_dates = []
    coeficientes = []
    prob = []
    intercept_values = []
    ndvi_magnitudes=[]
    
    for num, result in enumerate(results['change_models']):
        #days = np.arange(result['start_day'], result['end_day'] + 1)
        #prediction_dates.append(days)
        break_dates.append(result['break_day'])
        start_dates.append(result['start_day'])
        end_dates.append(result['end_day'])
        prob.append(int(result['change_probability'] * 100))
        
        # intercept = result['ndvi']['intercept']
        intercept_values.append(result['ndvi']['intercept'])
        
        # coef = result['ndvi']['coefficients']
        coeficientes.append(result['ndvi']['coefficients'])

        # predicted_values.append(intercept + coef[0] * days +
        #                         coef[1]*np.cos(days*1*2*np.pi/365.25) + coef[2]*np.sin(days*1*2*np.pi/365.25) +
        #                         coef[3]*np.cos(days*2*2*np.pi/365.25) + coef[4]*np.sin(days*2*2*np.pi/365.25) +
        #                         coef[5]*np.cos(days*3*2*np.pi/365.25) + coef[6]*np.sin(days*3*2*np.pi/365.25))
    
    # ndvi_magnitudes_1 = [int(predicted_values[num][-1] - predicted_values[num + 1][0]) for num in range(len(predicted_values) - 1)]
    # ndvi_magnitudes.extend(ndvi_magnitudes_1)
    # # Se não houver mais segmentos a seguir adiciona NODATA_VALUE se só existir um segmento adiciona -65535
    # ndvi_magnitudes.append(NODATA_VALUE if ndvi_magnitudes and any(ndvi_magnitudes) else -NODATA_VALUE)
    
    datas = [datetime.fromordinal(data) for data in break_dates]
    break_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in start_dates]
    start_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in end_dates]
    end_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    ponto_desejado_x, ponto_desejado_y = ponto_desejado
    ponto_desejado_x = int(ponto_desejado_x)
    ponto_desejado_y = int(ponto_desejado_y)

    # mask_array = np.array(results['processing_mask'], dtype='bool')
    # mask_len, mask_num_false = (len(mask_array), np.uint16(np.sum(~mask_array)))
    
    # If you remove the last element from tbreak when running the validation, an error occurs because the columns do not have the same size.
    dados = [{
        'tBreak': break_dates_epoch, 
        'tEnd': end_dates_epoch,
        'tStart': start_dates_epoch, 
        'changeProb': prob,
        'x_coord': ponto_desejado_x, 
        'y_coord': ponto_desejado_y,
        # 'ndvi_magnitude': ndvi_magnitudes,
        # 'prediction_dates': [d.tolist() for d in prediction_dates],
        # 'predicted_values': [[int(value) for value in sublist] for sublist in predicted_values], 
        'coeficientes': coeficientes, 
        'intercept_values': intercept_values
        #'mask_len': mask_len,
        #'mask_num_false': mask_num_false
        }]
    
    df = pd.DataFrame(dados)
    
    # Reorganize columns
    ordem_colunas = ['tBreak','tEnd', 'tStart', 'changeProb', 'x_coord', 'y_coord', #'ndvi_magnitude', 
                     'coeficientes', 'intercept_values']
                      #'prediction_dates', 'predicted_values', 'coeficientes', 'intercept_values']
                      #'mask_len', 'mask_num_false']
    df = df[ordem_colunas]
    return df
