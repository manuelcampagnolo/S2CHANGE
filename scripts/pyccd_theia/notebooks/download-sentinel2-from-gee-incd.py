import ee
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from multiprocessing.pool import ThreadPool as Pool
import os
import requests
# ---------------------------------
#             INPUTS
# ---------------------------------
# Configuracoes e execucao principal
file_path_original = r'/users1/cpca070342024/scaetano/CCD_yml_win/C:/Users/scaetano/Downloads/nvg_2018_ccd.gpkg'
file_path_tiles = r'/users1/cpca070342024/scaetano/CCD_yml_win/C:/Users/scaetano/Downloads/sentinel2_tiles_PT_terra_tm06.shp'
file_path = r'/users1/cpca070342024/scaetano/CCD_yml_win/C:/Users/scaetano/Downloads/nvg_2018_ccd_with_tiles.gpkg' # Nome dos tiles nas geometrias da BDR
tile_to_test = 'T29SPC'  # Escolher o tile 
date_start = '2023-08-01' # Escolher a data inicial para fazer o download das imagens S2
date_end = '2023-08-10' # Escolher a data final para fazer o download das imagens S2
# bandas = ['ndvi', 'B2', 'B3', 'B4', 'B8']
bandas = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'] # 12 bandas S2

# Caso especial para o tile 'T29SNB', adicionar a variï¿½vel de divisï¿½o
tile_to_split = 'T29TPE'
geometry_part = 'second'  # 'first' ou 'second' para escolher a parte da geometria
personal_project_gee = 'ee-testeccd1234'
#%%

# ---------------------------------
# Carregar o arquivo GeoPackage e adicionar a coluna 'tile'
# ---------------------------------
# Verificar se o arquivo jï¿½ existe
if not os.path.exists(file_path):
    print("Arquivo nï¿½o encontrado. Criando o arquivo com os tiles...")

    # Carregar o arquivo GeoPackage original
    gdf = gpd.read_file(file_path_original)

    # Carregar o shapefile dos tiles
    tiles_gdf = gpd.read_file(file_path_tiles)

    # Verificar e alinhar CRS (Sistema de Coordenadas)
    print("CRS do gdf:", gdf.crs)
    print("CRS dos tiles_gdf:", tiles_gdf.crs)

    # Se os CRS forem diferentes, reprojetar para o mesmo CRS
    if gdf.crs != tiles_gdf.crs:
        gdf = gdf.to_crs(tiles_gdf.crs)

    # Adicionar uma coluna 'tile' ao gdf
    gdf['tile'] = None

    # Iterar sobre cada geometria em gdf
    for idx, geom in gdf.iterrows():
        # Verificar os tiles que contï¿½m completamente a geometria
        containing_tiles = tiles_gdf[tiles_gdf.contains(geom.geometry)]
        
        if not containing_tiles.empty:
            # Se houver um tile que contï¿½m a geometria completamente, usa esse tile
            tile_id = containing_tiles.iloc[0]['Name']  
            gdf.at[idx, 'tile'] = tile_id
        else:
            # Se nï¿½o houver um tile que contenha completamente, verificar interseï¿½ï¿½o
            intersecting_tiles = tiles_gdf[tiles_gdf.intersects(geom.geometry)]
            if not intersecting_tiles.empty:
                # Se houver interseï¿½ï¿½es, assume o primeiro tile que intersecta
                tile_id = intersecting_tiles.iloc[0]['Name']
                gdf.at[idx, 'tile'] = tile_id
                
    # Salvar o resultado em um novo arquivo
    gdf.to_file(file_path, driver='GPKG')

    print(f"Coluna 'tile' adicionada e resultados salvos em '{file_path}'.")
else:
    print(f"Arquivo '{file_path}' jï¿½ existe. Carregando dados...")

    # Carregar o arquivo existente
    gdf = gpd.read_file(file_path)

    print(f"Arquivo '{file_path}' carregado com sucesso.")
    
# ---------------------------------
# SCRIPT PARA DOWNLOAD DAS IMAGENS SENTINEL-2 DIRETAMENTE DO GEE
# ---------------------------------

# Inicializar o Earth Engine
ee.Initialize(project=personal_project_gee)

# ---------------------------------
#       Funcoes Auxiliares
# ---------------------------------
# # Fun  o para adicionar banda NDVI s/ transformacao da escala
# def addNDVI(image):
#     ndvi = image.normalizedDifference(['B8', 'B4']).multiply(10000).int16()
#     return image.addBands(ndvi.rename('ndvi'))

# Funcao para adicionar banda NDVI c/ transformacao da escala
def addNDVI(image):
    # Calcula o NDVI
    ndvi = image.normalizedDifference(['B8', 'B4']).multiply(10000).int16()
    
    # Aplica  o da transforma  o:
    # Se NDVI >= 5000 -> NDVI = 5000
    # Se NDVI < 0 -> NDVI = 0
    # Se 0 <= NDVI < 5000 -> mant m o valor original
    ndvi_clipped = ndvi.expression(
        '((ndvi < 0) ? 0 : (ndvi >= 5000 ? 5000 : ndvi))',
        {'ndvi': ndvi}
    ).int16()
    
    return image.addBands(ndvi_clipped.rename('ndvi'))

# Funï¿½ï¿½o para filtrar imagens com base em nuvens usando a coleï¿½ï¿½o S2 Cloud Probability
def filterS2cloudless(S2SRCol, S2CloudCol):
    CLOUD_FILTER = 60
    CLD_PRB_THRESH = 50
    NIR_DRK_THRESH = 0.2
    CLD_PRJ_DIST = 1
    BUFFER = 50

    S2SRCol = S2SRCol.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))

    joined = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(
        primary=S2SRCol,
        secondary=S2CloudCol,
        condition=ee.Filter.equals(
            leftField='system:index',
            rightField='system:index'
        )))

    def add_cloud_bands(img):
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
        is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def add_shadow_bands(img):
        not_water = img.select('SCL').neq(6)
        SR_BAND_SCALE = 1e4
        dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
        shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
        cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
            .reproject(crs=img.select(0).projection(), scale=100)
            .select('distance')
            .mask()
            .rename('cloud_transform'))
        shadows = cld_proj.multiply(dark_pixels).rename('shadows')
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(img):
        img_cloud = add_cloud_bands(img)
        img_cloud_shadow = add_shadow_bands(img_cloud)
        is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
        is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
            .reproject(crs=img.select([0]).projection(), scale= 20)
            .rename('cloudmask'))
        return img_cloud_shadow.addBands(is_cld_shdw)

    def apply_cld_shdw_mask(img):
        not_cld_shdw = img.select('cloudmask').Not()
        return img.updateMask(not_cld_shdw)

    s2_sr = joined.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)

    return s2_sr

# Funï¿½ao para obter a coleï¿½ao de imagens filtrada
def getImageCollection(params_ImgCol):
    S2 = ee.ImageCollection(params_ImgCol['nameImage']).filterDate(params_ImgCol['date_start'], params_ImgCol['date_end'])

    if params_ImgCol['cloudFilter'] == 's2cloudless':
        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                            .filterDate(params_ImgCol['date_start'], params_ImgCol['date_end']))
        S2filtered = filterS2cloudless(S2, s2_cloudless_col)
    elif params_ImgCol['cloudFilter'] == 'NoFilter':
        S2filtered = S2

    if 'ndvi' in map(str.lower, params_ImgCol['indices']):
        S2filtered = S2filtered.map(addNDVI)

    return S2filtered

# Funï¿½ao para adicionar banda de data em milissegundos
def addDateBand(image):
    dateBand = ee.Image(ee.Date(image.date()).millis()).rename('image_date')
    return image.addBands(dateBand.toInt64())

def convert_band_types(image, target_type='uint16'):
    if target_type == 'uint16':
        # Converter todas as bandas para UInt16
        image = image.toUint16()
    elif target_type == 'int16':
        # Converter todas as bandas para Int16
        image = image.toInt16()
    return image
        
def exportImage(image, i, tile, geometry, folder_name):
    try:
        print(f'Exporting image: {i} for tile: {tile}')
        
        # Obter o nome do arquivo com base na data da imagem
        dateInMillis = image.date().getInfo()['value']
        fileName = f'S2SR_image_{dateInMillis}_tile_{tile}.tif'
        
        # Converter a imagem para o tipo correto (caso necessario)
        image = convert_band_types(image)
        
        # Aplicar a geometria como mascara
        masked_image = image.clip(geometry).unmask(65535)

        # Obter o URL de download da imagem
        url = masked_image.getDownloadURL({
            'scale': 10,
            'crs': 'EPSG:32629',
            'format': 'GeoTIFF',
	    'formatOptions': {'noData': 65535},
            'region': geometry,
        })
        
        # Definir o caminho para salvar o arquivo no disco local
        save_path = os.path.join(folder_name, fileName)
        
        # Criar o diretorio se nao existir
        os.makedirs(folder_name, exist_ok=True)
        
        # Baixar a imagem e salvar localmente
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f'Image saved as {save_path}')
        else:
            print(f'Failed to download image {i} for tile {tile}. HTTP status code: {response.status_code}')
    
    except Exception as e:
        print(f'Error exporting image {i} for tile {tile}: {e}')

# Carregar e processar o arquivo GeoPackage
def load_geometry_from_geopackage(file_path, tile):
    # Carregar o arquivo GeoPackage
    gdf = gpd.read_file(file_path)


    # Reprojetar o GeoDataFrame para EPSG:4326
    gdf = gdf.to_crs(epsg=4326)

    # Verificar se o tile existe no GeoDataFrame
    if tile not in gdf['tile'].values:
        raise ValueError(f"Tile {tile} nÃ£o encontrado no GeoPackage.")

    # Filtrar pelo tile especificado
    gdf_tile = gdf[gdf['tile'] == tile]
    
    # Unir geometria
    if not gdf_tile.empty:
        unified_polygon = gdf_tile.geometry.unary_union
        if isinstance(unified_polygon, (Polygon, MultiPolygon)):
            geojson = unified_polygon.__geo_interface__
            return ee.Geometry(geojson)
    return None

def load_geometry_from_geopackage_parts(file_path, tile, half=0.5, part='first'):
    """
    Carrega uma fraï¿½ï¿½o das geometrias para o tile especificado e retorna a geometria unida.
    
    Args:
    - file_path (str): Caminho para o arquivo GeoPackage.
    - tile (str): Identificador do tile.
    - half (float): Fraï¿½ï¿½o das geometrias a serem carregadas.
    - part (str): Indica se deve carregar a 'primeira' ou 'segunda' metade das geometrias.
    
    Returns:
    - ee.Geometry: Geometria unida correspondente ï¿½ fraï¿½ï¿½o e parte especificadas.
    """
    # Carregar o arquivo GeoPackage
    gdf = gpd.read_file(file_path)

    # Reprojetar o GeoDataFrame para EPSG:4326
    gdf = gdf.to_crs(epsg=4326)

    # Verificar se o tile existe no GeoDataFrame
    if tile not in gdf['tile'].values:
        raise ValueError(f"Tile {tile} nï¿½o encontrado no GeoPackage.")

    # Filtrar pelo tile especificado
    gdf_tile = gdf[gdf['tile'] == tile]
    
    # Dividir as geometrias
    if not gdf_tile.empty:
        # Selecionar a fraï¿½ï¿½o especificada
        split_index = int(len(gdf_tile) * half)
        if part == 'first':
            gdf_tile_part = gdf_tile.iloc[:split_index]
        elif part == 'second':
            gdf_tile_part = gdf_tile.iloc[split_index:]
        else:
            raise ValueError("O parï¿½metro 'part' deve ser 'first' ou 'second'.")
        
        # Listar os nomes das geometrias
        geometry_names = gdf_tile_part['geometry'].index.tolist()
        print(f"Geometries in the {part} half: {geometry_names}")
        
        # Unir geometria
        unified_polygon = gdf_tile_part.geometry.unary_union
        if isinstance(unified_polygon, (Polygon, MultiPolygon)):
            geojson = unified_polygon.__geo_interface__
            return ee.Geometry(geojson)
    return None

# ---------------------------------
#   Inicializaï¿½ï¿½o do Earth Engine
# ---------------------------------
try:
    # Verifica se o tile ï¿½ o especial T29SNB
    if tile_to_test == tile_to_split:
        # Usar divisï¿½o de geometria para este tile
        geometry = load_geometry_from_geopackage_parts(file_path, tile_to_test, half=0.5, part=geometry_part)
        folder_name = tile_to_test if geometry_part == 'first' else f'{tile_to_test}(2)'
    else:
        # Carregar geometria completa para outros tiles
        geometry = load_geometry_from_geopackage(file_path, tile_to_test)
        folder_name = tile_to_test

    if geometry:
        params_ImgCol = {
            'nameImage': "COPERNICUS/S2_SR_HARMONIZED",
            'date_start': date_start,
            'date_end': date_end,
            'indices': ['ndvi'],
            'cloudFilter': 's2cloudless',
            'bandas': bandas,
            'banda': 'ndvi'
        }

        # Obtem a colecao de imagens filtrada
        s2_col_filtered = getImageCollection(params_ImgCol)

        # Adiciona a banda de data em milissegundos e ordena pela data
        s2_col_with_date = s2_col_filtered.map(addDateBand)
        s2_col_sorted = s2_col_with_date.sort('image_date')

        # Seleciona as bandas de interesse
        s2_col_selection = s2_col_sorted.select(bandas)

        # Filtra as imagens para o tile especificado
        s2_col_selection = s2_col_selection.filterMetadata('MGRS_TILE', 'EQUALS', tile_to_test[1:])

        print('Number of images selected for tile {}: {}'.format(tile_to_test, s2_col_selection.size().getInfo()))
        print('First image info for tile {}: {}'.format(tile_to_test, s2_col_selection.first().getInfo()))

        # Converte a colecao de imagens para uma lista
        imageList = s2_col_selection.toList(s2_col_selection.size())

        # Exporta as imagens
        pool = Pool(processes=4)
        pool.starmap(exportImage, [(ee.Image(imageList.get(i)), i, tile_to_test, geometry, folder_name) for i in range(imageList.size().getInfo())])
    else:
        print(f'Geometria para o tile {tile_to_test} nï¿½o encontrada.')

except Exception as e:
    print(f'Erro: {e}')