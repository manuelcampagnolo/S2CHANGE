import xarray as xr
import rioxarray
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd
from notebooks.read_files import convertPointToCrs
import ccd
from rasterio.features import geometry_window
from shapely.geometry import Point
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import dask
import dask.array as da
from rasterio.windows import Window
#%%
def processar_centros_pixeis(shapefile_path, raster_path):
    # Carregar o shapefile
    poligonos = gpd.read_file(shapefile_path)
    caminho_raster = raster_path

    # Lista para armazenar os centros dos pixels para cada geometria
    todos_centros_pixeis = []
    poligonos = poligonos[poligonos.is_valid]

    for index, row in poligonos.iterrows():
        
        # Obter a geometria do polígono
        geometry = row['geometry']

        # Carregar o raster
        with rasterio.open(caminho_raster) as src:
            window = geometry_window(src, [geometry])

            transform = src.window_transform(window)

            # Obter o tamanho do pixel
            x_res = transform.a
            y_res = transform.e

            # Calcular o deslocamento do centro do pixel
            x_offset = x_res / 2.0
            y_offset = y_res / 2.0

            pixel_centers = []

            # Calcular o centro do pixel para cada pixel na janela
            for y in range(window.height):
                for x in range(window.width):
                    # Calcular as coordenadas do centro do pixel
                    pixel_center_x = transform.c + (x * x_res) + x_offset
                    pixel_center_y = transform.f + (y * y_res) + y_offset
                    
                    # Verificar se o ponto do centro do pixel está dentro do polígono
                    if Point(pixel_center_x, pixel_center_y).within(geometry):
                        # Armazenar as coordenadas do centro do pixel na lista
                        pixel_centers.append((pixel_center_x, pixel_center_y))
        
        # Adicionar os centros dos pixels desta geometria à lista geral
        todos_centros_pixeis.append(pixel_centers)
        
    pontos_shapely = [Point(centro) for sublist in todos_centros_pixeis for centro in sublist]

    # Criar um GeoDataFrame a partir da lista de pontos
    gdf_centros_pixeis = gpd.GeoDataFrame(geometry=pontos_shapely)
    
    return gdf_centros_pixeis
#%%
# def getTimeSeriesForPoints(tif_names, tif_dates_ord, bandas_desejadas, dados_geoespaciais_metros):

#     time_var = xr.Variable('time',tif_dates_ord)
#     # Load in and concatenate all individual GeoTIFFs
#     tifs_xr = [rioxarray.open_rasterio(i, chunks={'x':10924, 'y':10900}) for i in tif_names]
#     geotiffs_da = xr.concat(tifs_xr, dim=time_var).sel(band=bandas_desejadas)

#     # COORDENADAS X E Y DOS 10 000 PONTOS ESCOLHIDOS
#     points_x_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.x.values).astype('int'), dims=['location'])
#     points_y_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.y.values).astype('int'), dims=['location'])

#     selection = geotiffs_da.sel(x=points_x_int, y=points_y_int, band=bandas_desejadas)
#     dates = selection.time
#     xs = selection.x
#     ys = selection.y
#     sel_values = selection.values

#     return sel_values, dates, xs, ys

def getTimeSeriesForPoints(tif_names, tif_dates_ord, bandas_desejadas, dados_geoespaciais_metros, output_file):
    '''
    inputs:
    tif_names: list..
    '''
    time_var = xr.Variable('time',tif_dates_ord)
    # Load in and concatenate all individual GeoTIFFs
    tifs_xr = [rioxarray.open_rasterio(i, chunks={'x':-1, 'y':-1}) for i in tif_names]
    geotiffs_da = xr.concat(tifs_xr, dim=time_var).sel(band=bandas_desejadas)

    # COORDENADAS X E Y DOS 10 000 PONTOS ESCOLHIDOS
    points_x_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.x.values).astype('int'), dims=['location'])
    points_y_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.y.values).astype('int'), dims=['location'])

    selection = geotiffs_da.sel(x=points_x_int, y=points_y_int, band=bandas_desejadas)
    dates = selection.time
    xs = selection.x
    ys = selection.y
    sel_values = selection.values
    
    np.save(output_file + '_xs.npy', xs)
    np.save(output_file + '_ys.npy', ys)
    
    np.save(output_file, sel_values)

    return dates
#%%
def runDetectionForPoint(args, plot_flag=False): # se plot_flag =  False não faz gráficos se True faz gráficos
    i,sel_values, dates, xs, ys, NODATA_VALUE, FOLDER_OUTPUTS, img_collection = args

    ponto = sel_values[:,:,i]

    ponto_desejado=xs[i],ys[i]
    
    ponto_with_dates = np.column_stack((dates, ponto[:, 0], ponto[:, 1:]))
    
    mask = (ponto_with_dates != NODATA_VALUE).all(axis=1)
    ponto_with_dates_filtered = ponto_with_dates[mask].transpose()
    
    dates, blues, greens, reds, nirs, swir1s, swir2s = ponto_with_dates_filtered
    
    # Calcular o NDVI
    ndvis = np.where((nirs + reds) > 0, 10000 * (nirs - reds) / (nirs + reds), NODATA_VALUE)
    
    ponto_with_dates_filtered[1]=ndvis
    
    ponto_with_dates_filtered1=ponto_with_dates_filtered.transpose()
    
    ponto_with_dates_filtered2 = ponto_with_dates_filtered1[~np.any(ponto_with_dates_filtered1 == NODATA_VALUE, axis=1)]
    
    ponto_with_dates_filtered3=ponto_with_dates_filtered2.transpose()
    
    dates, ndvis, greens, reds, nirs, swir1s, swir2s = ponto_with_dates_filtered3
    
    # results = ccd.detect(dates, ndvis, greens, reds, nirs, swir1s, swir2s)
    results = ccd.detect(dates, ndvis, greens, swir2s)
    
    
    predicted_values = []
    prediction_dates = []
    break_dates = []
    start_dates = []
    end_dates=[]
    coeficientes=[]
    prob=[]
    
    for num, result in enumerate(results['change_models']):
        days = np.arange(result['start_day'], result['end_day'] + 1)
        prediction_dates.append(days)
        break_dates.append(result['break_day'])
        start_dates.append(result['start_day'])
        end_dates.append(result['end_day'])
        prob.append(result['change_probability'])
        
        intercept = result['ndvi']['intercept']
        coef = result['ndvi']['coefficients']
        coeficientes.append(coef)
        
        coef_str = f"({coef[0]:.2f}, {coef[1]:.2f}, {coef[2]:.2f}, {coef[3]:.2f}, {coef[4]:.2f}, {coef[5]:.2f}, {coef[6]:.2f})"
        
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

    # Se plot_flag = True faz gráficos
    if plot_flag:
        # BANDA QUE QUEREMOS PLOTAR NO GRÁFICO
        variavel_grafico = ndvis
    
        mask = np.array(results['processing_mask'], dtype='bool')
        date_objects1 = [datetime.fromordinal(int(ordinal)) for ordinal in dates]
        
        plt.style.use('ggplot')
        fg = plt.figure(figsize=(14, 4), dpi=90)
        
        limite_inicial = datetime.strptime('2018-01-01', '%Y-%m-%d')
        limite_final = datetime.strptime('2021-12-31', '%Y-%m-%d')
        
        a1 = fg.add_subplot(1, 1, 1, xlim=(limite_inicial, limite_final))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        
        a1.xaxis.set_major_locator(mdates.YearLocator(1))
        a1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        
        colors = ['orange', 'purple', 'brown']
        
        # Predicted curves
        for idx, (_preddate, _predvalue, _coef) in enumerate(zip(prediction_dates, predicted_values, coeficientes)):
            # Converter números ordinais de volta para objetos de data
            _preddate = [datetime.fromordinal(int(ordinal)) for ordinal in _preddate]
            color = colors[idx % len(colors)]
            coef_str = f"({', '.join([f'{c:.2f}' for c in _coef])})"
            label = f'Predicted values {idx + 1} (Coefs: {coef_str})'
            a1.plot(_preddate, _predvalue, color, linewidth=1, label=label)
        
        a1.plot(np.array(date_objects1)[mask], np.array(variavel_grafico)[mask], 'g+',label='Observed values')  # Observed values
        a1.plot(np.array(date_objects1)[~mask], np.array(variavel_grafico)[~mask], 'g+')  # Observed values masked out
    
        ticks = [min(date_objects1) + timedelta(days=i*365) for i in range(10) if min(date_objects1) + timedelta(days=i*365) <= datetime(2021, 12, 31)]
        plt.xticks(ticks)
        plt.title('Lat:' + str(round(ponto_desejado_wgs_x, 5)) + ' Lon:' + str(round(ponto_desejado_wgs_y, 5)))
        
        a1.plot([], [], color='r', linestyle='--', label='Start dates')
        a1.plot([], [], color='brown', linestyle='--', label='End Dates')
        a1.plot([], [], color='b', linestyle='--', label='Break dates')
        # a1.plot([], [], color='black', linestyle='--', label='DGT Dates')
        
        for b in break_dates:
            b_date = datetime.fromordinal(b)
            a1.axvline(b_date, color='b', linestyle='--')
            a1.text(mdates.date2num(b_date)+1, a1.get_ylim()[1], b_date.strftime('%d-%m-%Y'), rotation=90, ha='right',weight='bold', va='top', color='b',size=8)
        
        # Linhas verticais para datas de início (color='r')
        for s in start_dates:
            s_date = datetime.fromordinal(s)
            a1.axvline(s_date, color='r', linestyle='--')
            a1.text(mdates.date2num(s_date) + 1, a1.get_ylim()[0], s_date.strftime('%d-%m-%Y'), rotation=90, ha='right',weight='bold', va='bottom', color='r',size=8)
        
        for e in end_dates:
            e_date = datetime.fromordinal(e)
            a1.axvline(e_date, color='brown', linestyle='--')
            a1.text(mdates.date2num(e_date) + 1, a1.get_ylim()[0], e_date.strftime('%d-%m-%Y'), rotation=90, ha='right',weight='bold', va='bottom', color='brown',size=8,alpha=0.6)
 
        reference_start_date = datetime.strptime('2018-09-12', '%Y-%m-%d')
        reference_end_date = datetime.strptime('2021-09-30', '%Y-%m-%d')
        a1.axvspan(reference_start_date, reference_end_date, facecolor='pink', alpha=0.3,label='Período de Referência')

        plt.ylabel('NDVI')
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
        plt.tight_layout()
        caminho_graficos=os.path.join(FOLDER_OUTPUTS / 'plots' / f'{img_collection}_ccdc_ponto_{i}_{start_dates[0]}_{end_dates[-1]}.png')
        plt.savefig(caminho_graficos)
        plt.close()

    return df
