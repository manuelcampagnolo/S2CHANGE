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
from shared.read_files import read_tif_files_theia, read_tif_files_gee, readPoints, convertPointToCrs
from shared.utils import get_largest_tif_by_pixels
from pyproj import CRS
#%%
def create_geodataframe_from_parquet(filename, epsg_input, epsg_output, S2_tile, parquet_dir, shapefile_dir):
    """
    Cria um GeoDataFrame a partir de um arquivo Parquet contendo coordenadas geográficas, reprojeta-o 
    e adiciona uma coluna de quebra temporal (tBreak). Em seguida, salva o GeoDataFrame como um Shapefile
    numa pasta criada dinamicamente com base no S2_tile.

    Args:
        filename (str): Nome do arquivo Parquet sem extensão.
        epsg_input (int): Código EPSG do sistema de referência espacial de entrada.
        epsg_output (int): Código EPSG do sistema de referência espacial de saída.
        S2_tile (str): Identificador do tile S2 usado para criar a subpasta.
        parquet_dir (Path): Caminho para a pasta onde o arquivo Parquet está localizado.
        shapefile_dir (Path): Caminho para a pasta onde o shapefile será salvo.

    Returns:
        GeoDataFrame: Um GeoDataFrame com a geometria reprojetada e a coluna tBreak adicionada.
    """
    # Construir o caminho completo para o arquivo Parquet
    parquet_path = Path(parquet_dir) / f"{filename}.parquet"
    
    # Carregar o arquivo Parquet para um DataFrame do pandas
    df = pd.read_parquet(parquet_path)
    
    # Criar uma coluna 'geometry' com objetos Point baseados em Lat e Lon
    geometry = [Point(lon, lat) for lon, lat in zip(df['Lon'], df['Lat'])]
    
    # Criar um GeoDataFrame a partir do DataFrame original e da geometria
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS.from_epsg(epsg_input))
    
    # Reprojetar para o sistema de coordenadas EPSG especificado
    gdf = gdf.to_crs(epsg=epsg_output)
    
    # Adicionar a coluna tBreak ao GeoDataFrame
    gdf['tBreak'] = df['tBreak']
    
    shapefile_path = Path(shapefile_dir) / f"{filename}.shp"
    
    # Salvar o GeoDataFrame como um Shapefile
    gdf.to_file(shapefile_path, driver='ESRI Shapefile')
    
    return gdf
#%%
def processPointData(args):
    """
    Processa os dados de um ponto específico, incluindo a filtragem de NODATA_VALUE e o cálculo do NDVI.
    
    Args:
        - i (int): Índice do ponto de interesse.
        - sel_values (3D array): 3D-Array de valores selecionados (numero de imagens x numero de bandas x batch_size).
        - dates (ndarray): Array de datas.
        - xs (ndarray): Array de coordenadas x.
        - ys (ndarray): Array de coordenadas y.
        - NODATA_VALUE (float): Valor que representa dados ausentes.
        - FOLDER_OUTPUTS (str): Diretório para salvar os resultados.
        - img_collection (ndarray): Coleção de imagens Sentinel-2.
    
    Returns:
        - dates (ndarray): Array de datas filtradas.
        - ndvis (ndarray): Array de valores NDVI.
        - greens (ndarray): Array de valores da banda verde.
        - reds (ndarray): Array de valores da banda vermelha.
        - nirs (ndarray): Array de valores da banda NIR.
        - swir2s (ndarray): Array de valores da banda SWIR2.
        - ponto_desejado (tuple): Coordenadas do ponto desejado (x, y).
    """
    i, sel_values, dates, xs, ys, NODATA_VALUE, MAX_VALUE_NDVI, FOLDER_OUTPUTS, CRS_THEIA, CRS_WGS84, img_collection = args

    # Extrair o ponto de interesse
    ponto = sel_values[:, :, i]

    # Coordenadas do ponto desejado
    ponto_desejado = xs[i], ys[i]

    # Combinar datas e valores do ponto numa matriz
    ponto_with_dates = np.column_stack((dates, ponto[:, 0], ponto[:, 1:]))

    # Aplicar máscara para remover NODATA_VALUE
    mask = (ponto_with_dates != NODATA_VALUE).all(axis=1)
    ponto_with_dates_filtered = ponto_with_dates[mask].transpose()

    # Separar as bandas e as datas
    dates, greens, reds, nirs, swir2s = ponto_with_dates_filtered

    # Calcular o NDVI
    ndvis = np.where((nirs + reds) > 0, MAX_VALUE_NDVI * (nirs - reds) / (nirs + reds), NODATA_VALUE)
    
    # Calcular o NBR
    # nbrs = np.where((nirs + swir2s) > 0, MAX_VALUE_NDVI * (nirs - swir2s) / (nirs + swir2s), NODATA_VALUE)
    
    # Criar um novo array com NDVI na posição 1
    ponto_with_dates_updated = np.vstack((dates, ndvis, greens, reds, nirs, swir2s))
    # ponto_with_dates_updated = np.vstack((dates, ndvis, greens, reds, nirs, swir2s, nbrs))
    
    
    # Filtrar novamente para remover NODATA_VALUE
    ponto_with_dates_updated1 = ponto_with_dates_updated.transpose()
    ponto_with_dates_updated2 = ponto_with_dates_updated1[~np.any(ponto_with_dates_updated1 == NODATA_VALUE, axis=1)]
    ponto_with_dates_final = ponto_with_dates_updated2.transpose()
    

    # Separar as bandas e as datas novamente após a filtragem
    dates, ndvis, greens, reds, nirs, swir2s = ponto_with_dates_final
    # dates, ndvis, greens, reds, nirs, swir2s, nbrs = ponto_with_dates_final


    return dates, ndvis, greens, reds, nirs, swir2s, ponto_desejado, NODATA_VALUE, CRS_THEIA, CRS_WGS84
    #return dates, ndvis, greens, reds, nirs, swir2s, nbrs, ponto_desejado, NODATA_VALUE, CRS_THEIA, CRS_WGS84
#%%
def runDetectionForPoint(args):
    """
    Executa o CCD para um ponto específico.
    
    Args:
        - i (int): Índice do ponto de interesse.
        - sel_values (ndarray): Array de valores selecionados.
        - dates (ndarray): Array de datas.
        - xs (ndarray): Array de coordenadas x.
        - ys (ndarray): Array de coordenadas y.
        - NODATA_VALUE (float): Valor que representa dados ausentes.
        - FOLDER_OUTPUTS (str): Diretório para salvar os resultados.
        - img_collection (ndarray): Coleção de imagens Sentinel-2.

    Returns:
        - df (DataFrame): DataFrame com os resultados do CCD.
    """
    # Processar os dados do ponto
    dates, ndvis, greens, reds, nirs, swir2s, ponto_desejado, NODATA_VALUE, CRS_THEIA, CRS_WGS84 = processPointData(args)

    # Executar a detecção de mudanças
    results = ccd.detect(dates, ndvis, greens, swir2s)
    # results = ccd.detect(dates, ndvis, greens, swir2s, nbrs)

    # Chamar a função auxiliar para processar os resultados
    df = process_detection_results(results, ponto_desejado, NODATA_VALUE, CRS_THEIA, CRS_WGS84)

    return df
#%%
def process_detection_results(results, ponto_desejado, NODATA_VALUE, CRS_THEIA, CRS_WGS84):
    """
    Processa os resultados da detecção de mudanças para um ponto específico.(1 pixel; all segments)

    Args:
        - results (dict): Resultados da detecção de mudanças, esperado como um dicionário com 'change_models' e 'processing_mask'.
        - dates (ndarray): Array de datas.
        - ndvis (ndarray): Array de valores NDVI.
        - ponto_desejado (tuple): Coordenadas do ponto desejado tuple (x,y) 'crs': 'EPSG:32629',
        - NODATA_VALUE (float): Valor que representa dados ausentes.
        - CRS_THEIA (str): Sistema de coordenadas de referência (THEIA).
        - CRS_WGS84 (str): Sistema de coordenadas de referência (WGS84).

    Returns:
        - df (DataFrame): DataFrame que contém os resultados processados da detecção de mudanças.
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
    
    # se remover o ultimo elemento do tbreak ao correr a validação dá erro porque as colunas não tem o mesmo tamanho
    dados = [{
        'tBreak': break_dates_epoch[:-1], 
        'tEnd': end_dates_epoch,
        'tStart': start_dates_epoch, 
        'changeProb': prob,
        'x_coord': ponto_desejado_x, 
        'y_coord': ponto_desejado_y,
        'ndvi_magnitude': ndvi_magnitudes,
        # 'prediction_dates': [d.tolist() for d in prediction_dates],
        # 'predicted_values': [[int(value) for value in sublist] for sublist in predicted_values], 
        'coeficientes': coeficientes, 
        'intercept_values': intercept_values
        #'mask_len': mask_len,
        #'mask_num_false': mask_num_false
        }]
    
    df = pd.DataFrame(dados)
    
    # Reorganizar colunas
    ordem_colunas = ['tBreak','tEnd', 'tStart', 'changeProb', 'x_coord', 'y_coord', 'ndvi_magnitude', 
                     'coeficientes', 'intercept_values']
                      #'prediction_dates', 'predicted_values', 'coeficientes', 'intercept_values']
                      #'mask_len', 'mask_num_false']
    df = df[ordem_colunas]
    return df
