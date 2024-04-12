import xarray as xr
import rioxarray
import numpy as np
from datetime import datetime, timezone
import pandas as pd
from notebooks.read_files import convertPointToCrs
import ccd


def getTimeSeriesForPoints(tif_names, tif_dates_ord, bandas_desejadas,dados_geoespaciais_metros):

    time_var = xr.Variable('time',tif_dates_ord)
    # Load in and concatenate all individual GeoTIFFs
    tifs_xr = [rioxarray.open_rasterio(i, chunks={'x':10924, 'y':10900}) for i in tif_names]
    geotiffs_da = xr.concat(tifs_xr, dim=time_var).sel(band=bandas_desejadas)
    #%% 
    # COORDENADAS X E Y DOS 10 000 PONTOS ESCOLHIDOS
    points_x_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.x.values).astype('int'), dims=['location'])
    points_y_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.y.values).astype('int'), dims=['location'])
    #%%
    selection = geotiffs_da.sel(x=points_x_int, y=points_y_int, band=bandas_desejadas)
    dates = selection.time
    xs = selection.x
    ys = selection.y
    sel_values = selection.values

    return sel_values, dates, xs, ys


def runDetectionForPoint(args):
    i,sel_values, dates, xs, ys, NODATA_VALUE = args

    ponto = sel_values[:,:,i]

    ponto_desejado=xs[i],ys[i]
    
    ponto_with_dates = np.column_stack((dates, ponto[:, 0], ponto[:, 1:]))
    
    mask = (ponto_with_dates != NODATA_VALUE).all(axis=1)
    ponto_with_dates_filtered = ponto_with_dates[mask].transpose()
    
    dates, blues, greens, reds, nirs, swir1s, swir2s = ponto_with_dates_filtered
    
    ndvis = np.where((nirs + reds) > 0, 10000 * (nirs - reds) / (nirs + reds), NODATA_VALUE)
    
    ponto_with_dates_filtered[1]=ndvis
    
    ponto_with_dates_filtered1=ponto_with_dates_filtered.transpose()
    
    ponto_with_dates_filtered2 = ponto_with_dates_filtered1[~np.any(ponto_with_dates_filtered1 == NODATA_VALUE, axis=1)]
    
    ponto_with_dates_filtered3=ponto_with_dates_filtered2.transpose()
    
    dates, ndvis, greens, reds, nirs, swir1s, swir2s = ponto_with_dates_filtered3
    
    results = ccd.detect(dates, ndvis, greens, reds, nirs, swir1s, swir2s)
    
    
    predicted_values = []
    prediction_dates = []
    break_dates = []
    start_dates = []
    end_dates=[]
    # coeficientes=[]
    prob=[]
    
    for num, result in enumerate(results['change_models']):
        days = np.arange(result['start_day'], result['end_day'] + 1)
        prediction_dates.append(days)
        break_dates.append(result['break_day'])
        start_dates.append(result['start_day'])
        end_dates.append(result['end_day'])
        prob.append(result['change_probability'])
        
        intercept = result['blue']['intercept']
        coef = result['blue']['coefficients']
        
        predicted_values.append(intercept + coef[0] * days +
                                coef[1]*np.cos(days*1*2*np.pi/365.25) + coef[2]*np.sin(days*1*2*np.pi/365.25) +
                                coef[3]*np.cos(days*2*2*np.pi/365.25) + coef[4]*np.sin(days*2*2*np.pi/365.25) +
                                coef[5]*np.cos(days*3*2*np.pi/365.25) + coef[6]*np.sin(days*3*2*np.pi/365.25))
    
    ndvi_magnitudes = [predicted_values[num][-1] - predicted_values[num + 1][0] for num in range(len(predicted_values) - 1)]
    
    # Se não houver mais segmentos a seguir adiciona NODATA_VALUE se só existir um segmento adiciona 0
    ndvi_magnitudes.append(65535 if ndvi_magnitudes and any(ndvi_magnitudes) else 0)
    
    datas = [datetime.fromordinal(data) for data in break_dates]
    break_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in start_dates]
    start_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in end_dates]
    end_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    ponto_desejado_wgs = convertPointToCrs(ponto_desejado, 32629, 4326)
    
    ponto_desejado_wgs_x, ponto_desejado_wgs_y = ponto_desejado_wgs
    
    dados = [
        {'tBreak': break_dates_epoch,'tEnd': end_dates_epoch,'tStart':start_dates_epoch,'changeProb':prob, 'Lat': ponto_desejado_wgs_y,'Lon': ponto_desejado_wgs_x, 'ndvi_magnitude' : ndvi_magnitudes}
    ]
    
    df = pd.DataFrame(dados)
    
    # Reorganizar colunas
    ordem_colunas = ['tBreak', 'tEnd', 'tStart', 'changeProb', 'Lat', 'Lon', 'ndvi_magnitude']
    df=df[ordem_colunas]

    return df