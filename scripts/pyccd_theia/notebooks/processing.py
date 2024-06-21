import xarray as xr
import rioxarray
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd
import ccd
from rasterio.features import geometry_window
from shapely.geometry import Point
import rasterio
import geopandas as gpd
import os
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from notebooks.read_files import read_tif_files_theia, read_tif_files_gee, readPoints, convertPointToCrs
#%%
def processar_centros_pixeis(shapefile_path, raster_path):
    """
    A função processar_centros_pixeis é responsável por calcular os centros dos pixels dentro das geometrias
    de um shapefile específico, com base num raster fornecido. Para cada polígono no shapefile, a função 
    calcula as coordenadas dos centros dos pixels dentro dessa área, retornando um GeoDataFrame que contem esses pontos.
    
    Inputs:
    - shapefile_path (str): O caminho do arquivo shapefile que contém as geometrias dos polígonos.
    - raster_path (str): O caminho do arquivo raster que será usado para calcular os centros dos pixels.
    
    Outputs:
    - gdf_centros_pixeis (GeoDataFrame): Um GeoDataFrame que contem as coordenadas dos centros dos pixels 
      dentro das geometrias do shapefile.
    """
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
    
    np.save(str(output_file.with_suffix('')) + '_xs.npy', xs)
    np.save(str(output_file.with_suffix('')) + '_ys.npy', ys)
    
    np.save(output_file, sel_values)

    return dates
#%%
def check_or_initialize_file(output_file, tiles, var, S2_tile, BDR_DGT, N, random_state_value, bandas_desejadas, FOLDER_OUTPUTS, img_collection, NODATA_VALUE):
    """
    Verifica a existência de um arquivo numpy específico e, dependendo dessa verificação,
    realiza diferentes operações para processar dados geoespaciais.

    Inputs:
    - output_file (str): O caminho do arquivo numpy que será verificado e, se necessário, criado.
    - tiles (str): O diretório onde os arquivos TIFF (raster) estão localizados.
    - var (str): A variável que indica a origem dos dados, podendo ser 'THEIA' ou 'GEE'.
    - S2_tile (str): Identificador da tile de Sentinel-2 a ser processado.
    - BDR_DGT (GeoDataFrame): Um GeoDataFrame contendo as geometrias que serão processados.
    - N (int): O número de pontos a serem processados.
    - random_state_value (int): O valor usado para inicializar o gerador de números aleatórios.
    - bandas_desejadas (list): A lista de bandas desejadas para o processamento.
    - FOLDER_OUTPUTS (str): O diretório onde os resultados serão salvos.
    - img_collection (str): A coleção de imagens a ser utilizada.
    - NODATA_VALUE (int): O valor a ser usado para representar dados ausentes.

    Output:
    - tif_dates_ord (list): Uma lista de datas no formato ordinal, que pode ser usada para análises temporais subsequentes.
    """
    
    if os.path.exists(output_file):
        # Se o arquivo numpy existe, apenas carregar e processar os dados
        print(f"O arquivo '{output_file}' já existe. Carregando e processando os dados existentes...")
        # Recolher nome e data dos tifs
        print('A recolher nome e data dos tifs...')
        if var == 'THEIA':
            tif_names, tif_dates = read_tif_files_theia(S2_tile, tiles)
        else:
            tif_names, tif_dates = read_tif_files_gee(S2_tile, tiles)
        tif_names = [os.path.join(tiles, i) for i in tif_names]
        tif_dates_ord = [d.toordinal() for d in tif_dates]
    else:
        # Se o arquivo numpy não existe, executar todo o processamento inicial de criar o ficheiro numpy
        print(f"O arquivo '{output_file}' não existe. Iniciando processamento...")
        
        # Processar centros dos pontos de cada geometria para corresponder aos centros dos pixels dos rasters
        raster_path = tiles / 'Theia_T29TNE_20170813-112433.tif'
        print('Processar centros dos pontos de cada geometria para corresponder aos centros dos pixels dos rasters...')
        start_time = time.time()
        gdf_centros_pixeis = processar_centros_pixeis(BDR_DGT, raster_path)
        end_time = time.time()
        execution_time_seconds = end_time - start_time
        execution_time_minutes = execution_time_seconds / 60
        print("Processar centros dos pixels:", execution_time_minutes, "minutos")
        dados_geoespaciais_metros = readPoints(gdf_centros_pixeis, N, random_state_value)
        # Recolher nome e data dos tifs
        print('A recolher nome e data dos tifs...')
        if var == 'THEIA':
            tif_names, tif_dates = read_tif_files_theia(S2_tile, tiles)
        else:
            tif_names, tif_dates = read_tif_files_gee(S2_tile, tiles)
        tif_names = [os.path.join(tiles, i) for i in tif_names]
        tif_dates_ord = [d.toordinal() for d in tif_dates]
        print(f'Processando dados {var}... ({tiles})')
        start_time = time.time()
        # Abrir tifs com xarray e carregar série temporal
        print('A abrir tifs com xarray e carregar série temporal...')
        getTimeSeriesForPoints(tif_names, tif_dates_ord, bandas_desejadas, dados_geoespaciais_metros, output_file)
        
        end_time = time.time()
        execution_time_seconds = end_time - start_time
        execution_time_minutes = execution_time_seconds / 60
        print(f"Ler dados {var} para um total de {N} pixels:", execution_time_minutes, "minutos")

    return tif_dates_ord
#%%
def runDetectionForPoint(args): 
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
        
        # coef_str = f"({coef[0]:.2f}, {coef[1]:.2f}, {coef[2]:.2f}, {coef[3]:.2f}, {coef[4]:.2f}, {coef[5]:.2f}, {coef[6]:.2f})"
        
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
    
    # dados = [
    #     {'tBreak': break_dates_epoch,'tEnd': end_dates_epoch,'tStart':start_dates_epoch,'changeProb':prob, 'Lat': ponto_desejado_wgs_y,'Lon': ponto_desejado_wgs_x, 'ndvi_magnitude' : ndvi_magnitudes}
    # ]
    
    # df = pd.DataFrame(dados)
    
    # # Reorganizar colunas
    # ordem_colunas = ['tBreak', 'tEnd', 'tStart', 'changeProb', 'Lat', 'Lon', 'ndvi_magnitude']
    # df=df[ordem_colunas]
    
    dados = [
        {'tBreak': break_dates_epoch, 'tEnd': end_dates_epoch, 'tStart': start_dates_epoch, 'changeProb': prob,
         'Lat': ponto_desejado_wgs_y, 'Lon': ponto_desejado_wgs_x, 'ndvi_magnitude': ndvi_magnitudes,
         'ndvis': ndvis.tolist(), 'dates': dates.tolist(), 'prediction_dates': [d.tolist() for d in prediction_dates],
         'predicted_values': [v.tolist() for v in predicted_values], 'coeficientes': coeficientes, 
         'mask': np.array(results['processing_mask'], dtype='bool').tolist()}
    ]
    
    df = pd.DataFrame(dados)
    
    # Reorganizar colunas
    ordem_colunas = ['tBreak', 'tEnd', 'tStart', 'changeProb', 'Lat', 'Lon', 'ndvi_magnitude', 'ndvis', 'dates', 
                      'prediction_dates', 'predicted_values', 'coeficientes', 'mask']
    
    df=df[ordem_colunas]
    return df
