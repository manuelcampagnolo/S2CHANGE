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
from rasterio.features import rasterize
from shapely.geometry import box
import geopandas as gpd
import os
import time
from shared.read_files import read_tif_files_theia, read_tif_files_gee, readPoints, convertPointToCrs
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

def rasterize_vector_mask(clipped_vector_mask, reference_tif):

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
    
    return raster.astype(bool).flatten(order='F')


def getTimeSeriesForMask(tif_names, tif_dates_ord, bandas_desejadas, vector_mask_path, output_file):
    """
    Saves npy files with time series values and location (x, y) of pixels inside a ROI (mask).
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
    #clip vector geometries to tile extent
    clipped_vector_mask = clip_vector_mask_to_tile(vector_mask_path, tif_names[0])
    #rasterize clipped vector
    rasterized = rasterize_vector_mask(clipped_vector_mask, tif_names[0])

    #use xarray to handle time series
    time_var = xr.Variable('time',tif_dates_ord)
    # Load in and concatenate all individual GeoTIFFs
    tifs_xr = [rioxarray.open_rasterio(i, chunks={'x':-1, 'y':-1}) for i in tif_names]
    geotiffs_da = xr.concat(tifs_xr, dim=time_var).sel(band=bandas_desejadas).astype('uint16')

    #stack x and y to allow selecting by index
    geotiffs_da_stacked = geotiffs_da.stack(pixel=('x','y'))
    #select by index
    selection = geotiffs_da_stacked[:,:,rasterized]

    dates = selection.time
    xs = selection.x
    ys = selection.y
    sel_values = selection.values

    np.save(str(output_file.with_suffix('')) + '_xs.npy', xs)
    np.save(str(output_file.with_suffix('')) + '_ys.npy', ys)
    
    np.save(output_file, sel_values)

    return xs.shape[0]

    
#%%
def check_or_initialize_file(output_file, tiles, var, S2_tile, min_year, max_date, vector_mask_path, bandas_desejadas, FOLDER_OUTPUTS, img_collection, NODATA_VALUE, raster_path):
    """
    Verifica a existência de um arquivo numpy específico e, dependendo dessa verificação,
    realiza diferentes operações para processar dados geoespaciais.

    Args:
        - output_file (str) : caminho do arquivo numpy que será verificado e, se necessário, criado.
        - tiles (str) : diretório onde os arquivos TIFF (raster) estão localizados.
        - var (str) : variável que indica a origem dos dados, podendo ser 'THEIA' ou 'GEE'.
        - S2_tile (str) : Identificador da tile de Sentinel-2 a ser processado.
        - max_date (datetime.date) : Data máxima limite para o processamento dos arquivos TIFF. Apenas arquivos com datas até e incluindo `max_date` serão considerados.
        - vector_mask_path (GeoDataFrame) : GeoDataFrame contendo as geometrias que serão processados (poligonos).
        - bandas_desejadas (list) : lista de bandas desejadas para o processamento.
        - FOLDER_OUTPUTS (str) : diretório onde os resultados serão salvos.
        - img_collection (str) : coleção de imagens a ser utilizada.
        - NODATA_VALUE (int) : valor a ser usado para representar dados ausentes.
        - raster_path (Path) : caminho do arquivo raster que será utilizado para processamento.

    Returns:
        - tif_dates_ord (list) : lista de datas no formato ordinal.
        - N : number of pixels/points to be processed.
    """
    
    if os.path.exists(output_file):
        # Se o arquivo numpy existe, apenas carregar e processar os dados
        print(f"O arquivo '{output_file}' já existe. Carregando e processando os dados existentes...")
        # Recolher nome e data dos tifs
        print('A recolher nome e data dos tifs...')
        if var == 'THEIA':
            tif_names, tif_dates = read_tif_files_theia(S2_tile, tiles, min_year, max_date)
        else:
            tif_names, tif_dates = read_tif_files_gee(S2_tile, tiles, max_date)
        tif_names = [os.path.join(tiles, i) for i in tif_names]
        tif_dates_ord = [d.toordinal() for d in tif_dates]
        aux = np.load(str(output_file.with_suffix('')) + '_xs.npy')
        N = aux.shape[0] #gets number of pixels/points

    else:
        # Se o arquivo numpy não existe, executar todo o processamento inicial de criar o ficheiro numpy
        print(f"O arquivo '{output_file}' não existe. Iniciando processamento...")
            
        # Recolher nome e data dos tifs
        print('A recolher nome e data dos tifs...')
        if var == 'THEIA':
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
        N = getTimeSeriesForMask(tif_names, tif_dates_ord, bandas_desejadas, vector_mask_path, output_file)
        
        end_time = time.time()
        execution_time_seconds = end_time - start_time
        execution_time_minutes = execution_time_seconds / 60
        print(f"Ler dados {var} para um total de {N} pixels:", execution_time_minutes, "minutos")

    return tif_dates_ord, N
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
