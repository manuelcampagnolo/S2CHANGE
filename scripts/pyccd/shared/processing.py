import xarray as xr
import rioxarray
from pathlib import Path
import numpy as np
from datetime import datetime, timezone
import pandas as pd
import ccd
from rasterio.features import geometry_window
from shapely.geometry import Point
import rasterio
import geopandas as gpd
import os
import time
from notebooks.read_files import read_tif_files_theia, read_tif_files_gee, readPoints, convertPointToCrs
from pyproj import CRS
#%%
def create_geodataframe_from_csv(filename, epsg_input, epsg_output, S2_tile, csv_dir, shapefile_dir):
    """
    Cria um GeoDataFrame a partir de um arquivo CSV contendo coordenadas geográficas, reprojeta-o 
    e adiciona uma coluna de quebra temporal (tBreak). Em seguida, salva o GeoDataFrame como um Shapefile
    numa pasta criada dinamicamente com base no S2_tile.

    Args:
        filename (str): Nome do arquivo CSV sem extensão.
        folder_outputs (Path): Caminho para a pasta onde o arquivo CSV está localizado.
        epsg_input (int): Código EPSG do sistema de referência espacial de entrada.
        epsg_output (int): Código EPSG do sistema de referência espacial de saída.
        S2_tile (str): Identificador do tile S2 usado para criar a subpasta.

    Returns:
        GeoDataFrame: Um GeoDataFrame com a geometria reprojetada e a coluna tBreak adicionada.
    """
    # Construir o caminho completo para o arquivo CSV
    csv_path = Path(csv_dir) / f"{filename}.csv"
    
    # Carregar o arquivo CSV para um DataFrame do pandas
    df = pd.read_csv(csv_path)
    
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
def processar_centros_pixeis(shapefile_path, raster_path):
    """
    Função para calcular os centros dos pixels dentro das geometrias de um shapefile com base num raster.
    input: polygon geopackage; raster
    output: point geopackage, where each point is the center of the pixel, for pixels within the input polygon
    """
    start_time = time.time()
    print('Processar centros dos pontos de cada geometria para corresponder aos centros dos pixels dos rasters...')
    # Carregar o shapefile
    poligonos = gpd.read_file(shapefile_path)
    poligonos = poligonos[poligonos.is_valid]

    # Lista para armazenar os centros dos pixels
    todos_centros_pixeis = []

    # Carregar o raster
    with rasterio.open(raster_path) as src:
        # Obter CRS do raster
        raster_crs = src.crs

        # Garantir que o shapefile esteja no mesmo CRS do raster
        if poligonos.crs != raster_crs:
            poligonos = poligonos.to_crs(raster_crs)

        for index, row in poligonos.iterrows():
            # Obter a geometria do polígono
            geometry = row['geometry']

            # Calcular a janela da geometria no raster
            try:
                window = geometry_window(src, [geometry])
            except Exception as e:
                print(f"Erro ao calcular a janela para a geometria {index}: {e}")
                continue

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
            todos_centros_pixeis.extend(pixel_centers)
        
    # Criar um GeoDataFrame a partir da lista de pontos
    pontos_shapely = [Point(centro) for centro in todos_centros_pixeis]
    gdf_centros_pixeis = gpd.GeoDataFrame(geometry=pontos_shapely, crs=raster_crs)
    
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    execution_time_minutes = execution_time_seconds / 60
    print("Processar centros dos pixels:", execution_time_minutes, "minutos. Número de pixels:", len(gdf_centros_pixeis))
    
    return gdf_centros_pixeis
#%%
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
def check_or_initialize_file(output_file, tiles, var, S2_tile, min_year, max_date, gdf_centros_pixeis, N, random_state_value, bandas_desejadas, FOLDER_OUTPUTS, img_collection, NODATA_VALUE, raster_path):
    """
    Verifica a existência de um arquivo numpy específico e, dependendo dessa verificação,
    realiza diferentes operações para processar dados geoespaciais.

    Args:
        - output_file (str): caminho do arquivo numpy que será verificado e, se necessário, criado.
        - tiles (str): diretório onde os arquivos TIFF (raster) estão localizados.
        - var (str): variável que indica a origem dos dados, podendo ser 'THEIA' ou 'GEE'.
        - S2_tile (str): Identificador da tile de Sentinel-2 a ser processado.
        - max_date (datetime.date): Data máxima limite para o processamento dos arquivos TIFF. Apenas arquivos com datas até e incluindo `max_date` serão considerados.
        - gdf_centros_pixeis (GeoDataFrame): GeoDataFrame contendo as geometrias que serão processados.
        - N (int): número de pontos a serem processados.
        - random_state_value (int): valor usado para inicializar o gerador de números aleatórios.
        - bandas_desejadas (list): lista de bandas desejadas para o processamento.
        - FOLDER_OUTPUTS (str): diretório onde os resultados serão salvos.
        - img_collection (str): coleção de imagens a ser utilizada.
        - NODATA_VALUE (int): valor a ser usado para representar dados ausentes.
        - raster_path (Path): caminho do arquivo raster que será utilizado para processamento.

    Returns:
        - tif_dates_ord (list): lista de datas no formato ordinal.
    """
    
    if os.path.exists(output_file):
        # Se o arquivo numpy existe, apenas carregar e processar os dados
        print(f"O arquivo '{output_file}' já existe. Carregando e processando os dados existentes...")
        # Recolher nome e data dos tifs
        print('A recolher nome e data dos tifs...')
        if var == 'Theia':
            tif_names, tif_dates = read_tif_files_theia(S2_tile, tiles, min_year, max_date)
        else:
            tif_names, tif_dates = read_tif_files_gee(S2_tile, tiles, max_date)
        tif_names = [os.path.join(tiles, i) for i in tif_names]
        tif_dates_ord = [d.toordinal() for d in tif_dates]

    else:
        # Se o arquivo numpy não existe, executar todo o processamento inicial de criar o ficheiro numpy
        print(f"O arquivo '{output_file}' não existe. Iniciando processamento...")
            
        # Recolher nome e data dos tifs
        print('A recolher nome e data dos tifs...')
        if var == 'Theia':
            tif_names, tif_dates = read_tif_files_theia(S2_tile, tiles, min_year, max_date)
        else:
            tif_names, tif_dates = read_tif_files_gee(S2_tile, tiles, max_date)
        tif_names = [os.path.join(tiles, i) for i in tif_names]
        tif_dates_ord = [d.toordinal() for d in tif_dates]
        print(f'Processando dados {var}... ({tiles})')
        start_time = time.time()
        
        # # Selecionar uma amostra aleat�ria de N pontos de gdf_centros_pixeis
        # print(f'Selecionando uma amostra aleatoria de {N} pontos...')
        # if len(gdf_centros_pixeis) > N:
        #     # Garantir que a amostragem � feita somente se existirem mais pontos que N
        #     gdf_centros_pixeis = gdf_centros_pixeis.sample(n=N, random_state=random_state_value)

        # Abrir tifs com xarray e carregar série temporal
        print('A abrir tifs com xarray e carregar série temporal...')
        getTimeSeriesForPoints(tif_names, tif_dates_ord, bandas_desejadas, gdf_centros_pixeis, output_file)
        
        end_time = time.time()
        execution_time_seconds = end_time - start_time
        execution_time_minutes = execution_time_seconds / 60
        print(f"Ler dados {var} para um total de {N} pixels:", execution_time_minutes, "minutos")

    return tif_dates_ord
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
    
    # Criar um novo array com NDVI na posição 1
    ponto_with_dates_updated = np.vstack((dates, ndvis, greens, reds, nirs, swir2s))
    
    
    # Filtrar novamente para remover NODATA_VALUE
    ponto_with_dates_updated1 = ponto_with_dates_updated.transpose()
    ponto_with_dates_updated2 = ponto_with_dates_updated1[~np.any(ponto_with_dates_updated1 == NODATA_VALUE, axis=1)]
    ponto_with_dates_final = ponto_with_dates_updated2.transpose()
    

    # Separar as bandas e as datas novamente após a filtragem
    dates, ndvis, greens, reds, nirs, swir2s = ponto_with_dates_final


    return dates, ndvis, greens, reds, nirs, swir2s, ponto_desejado, NODATA_VALUE, CRS_THEIA, CRS_WGS84
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

    # Chamar a função auxiliar para processar os resultados
    df = process_detection_results(results, dates, ndvis, ponto_desejado, NODATA_VALUE, CRS_THEIA, CRS_WGS84)

    return df
#%%
def process_detection_results(results, dates, ndvis, ponto_desejado, NODATA_VALUE, CRS_THEIA, CRS_WGS84):
    """
    Processa os resultados da detecção de mudanças para um ponto específico.

    Args:
        - results (dict): Resultados da detecção de mudanças, esperado como um dicionário com 'change_models' e 'processing_mask'.
        - dates (ndarray): Array de datas.
        - ndvis (ndarray): Array de valores NDVI.
        - ponto_desejado (tuple): Coordenadas do ponto desejado.
        - NODATA_VALUE (float): Valor que representa dados ausentes.
        - CRS_THEIA (str): Sistema de coordenadas de referência (THEIA).
        - CRS_WGS84 (str): Sistema de coordenadas de referência (WGS84).

    Returns:
        - df (DataFrame): DataFrame que contém os resultados processados da detecção de mudanças.
    """
    predicted_values = []
    prediction_dates = []
    break_dates = []
    start_dates = []
    end_dates = []
    coeficientes = []
    prob = []
    
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

        predicted_values.append(intercept + coef[0] * days +
                                coef[1]*np.cos(days*1*2*np.pi/365.25) + coef[2]*np.sin(days*1*2*np.pi/365.25) +
                                coef[3]*np.cos(days*2*2*np.pi/365.25) + coef[4]*np.sin(days*2*2*np.pi/365.25) +
                                coef[5]*np.cos(days*3*2*np.pi/365.25) + coef[6]*np.sin(days*3*2*np.pi/365.25))
    
    ndvi_magnitudes = [predicted_values[num][-1] - predicted_values[num + 1][0] for num in range(len(predicted_values) - 1)]
    
    # Se não houver mais segmentos a seguir adiciona NODATA_VALUE se só existir um segmento adiciona 0
    ndvi_magnitudes.append(NODATA_VALUE if ndvi_magnitudes and any(ndvi_magnitudes) else 0)
    
    datas = [datetime.fromordinal(data) for data in break_dates]
    break_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in start_dates]
    start_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in end_dates]
    end_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    def ms_to_date_str(ms):
        return datetime.utcfromtimestamp(ms / 1000).strftime('%d%m%Y')

    # Converter timestamps em formato ddmmyyyy
    break_dates_ddmmyyyy = [ms_to_date_str(ts) for ts in break_dates_epoch]
    end_dates_ddmmyyyy = [ms_to_date_str(ts) for ts in end_dates_epoch]
        
    ponto_desejado_wgs = convertPointToCrs(ponto_desejado, CRS_THEIA, CRS_WGS84)
    ponto_desejado_wgs_x, ponto_desejado_wgs_y = ponto_desejado_wgs
    
    # se remover o ultimo elemento do tbreak ao correr a validação dá erro porque as colunas não tem o mesmo tamanho
    dados = [{'tBreak': break_dates_epoch[:-1], 'tBreak_ddmmyyyy':break_dates_ddmmyyyy[:-1],'tEnd': end_dates_epoch,'tEnd_ddmmyyyy':end_dates_ddmmyyyy,
              'tStart': start_dates_epoch, 'changeProb': prob,
              'Lat': ponto_desejado_wgs_y, 'Lon': ponto_desejado_wgs_x, 'ndvi_magnitude': ndvi_magnitudes,
                'ndvis': ndvis.tolist(), 'dates': dates.tolist(), 'prediction_dates': [d.tolist() for d in prediction_dates],
                'predicted_values': [v.tolist() for v in predicted_values], 'coeficientes': coeficientes, 
                'mask': np.array(results['processing_mask'], dtype='bool').tolist()}]
    
    df = pd.DataFrame(dados)
    
    # Reorganizar colunas
    ordem_colunas = ['tBreak', 'tBreak_ddmmyyyy','tEnd','tEnd_ddmmyyyy', 'tStart', 'changeProb', 'Lat', 'Lon', 'ndvi_magnitude', 'ndvis', 'dates', 
                      'prediction_dates', 'predicted_values', 'coeficientes', 'mask']
    
    df = df[ordem_colunas]
    return df
