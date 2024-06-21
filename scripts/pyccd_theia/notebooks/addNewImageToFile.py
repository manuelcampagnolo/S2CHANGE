import numpy as np
import xarray as xr
import rioxarray
import os
user_profile = os.environ['USERPROFILE']
directory_path = os.path.join(user_profile, 'Desktop', 'CCD_yml_win')
os.chdir(directory_path)
import os
import sys
from pathlib import Path
# chamar python a partir da pasta 'CCD'
module_path= Path(__name__ ).parent.absolute() / 'S2CHANGE' / 'scripts' / 'pyccd_theia' #  / 'CCD' / 'S2CHANGE' / 'scripts' / 
base_path= Path(__name__ ).parent.absolute()  # dir do script; # dir referência (acima): 'DGT-S2CHANGE_2023'
if module_path not in sys.path:
    sys.path.append(str(module_path))
import ccd
from notebooks.read_files import read_tif_files_theia, read_tif_files_gee, readPoints
from notebooks.processing import processar_centros_pixeis
from notebooks.utils import fromParamsReturnName
import warnings
warnings.filterwarnings('ignore')
#%%
############ INPUTS ######################
# Caminho onde estão os dados todos
public_documents = Path('C:/Users/Public/Documents/')
# Caminhos para a base de dados de validação
# -> BDR DGT:
BDR_DGT = public_documents / 'BDR_300_artigo' / 'BDR_CCDC_TNE_Adjusted.shp'
# -> BDR NAVIGATOR:
# BDR_NAVIGATOR = .... caminho ainda não definido

# -> IMAGENS SENTINEL:
FOLDER_THEIA = public_documents / 'imagens_Theia' # Caminho dados THEIA
FOLDER_GEE = public_documents / 'imagens_GEE' # Caminho dados GEE

S2_tile = 'T29TNE'
var = 'THEIA' # choose variable: THEIA or GEE

if var == 'THEIA':
    tiles = FOLDER_THEIA / S2_tile
else:
    tiles = FOLDER_GEE / S2_tile
img_collection = tiles.parts[-2]

############ PARAMETROS PRÉ PROCESSAMENTO ########################
# N=241941 # numero total de pontos presentes na BDR
N = 10000 # número de pontos aleatórios
random_state_value = 42

bandas_desejadas=[1, 2, 3, 7, 10]
NODATA_VALUE= 65535

############ PARAMETROS CCD ########################
alpha = ccd.parameters.defaults['ALPHA'] # Looks for alpha in the parameters.py file
ccd_params = ccd.parameters.defaults

############ PARAMETROS DA VALIDAÇÃO ########################
# datas do filtro das datas da análise (DGT 300)
########### Não alterar ################
dt_ini = '2018-09-12' # data inicial
dt_end = '2021-09-30' # data final
# Margem de tolerância entre a quebra do Modelo e do Analista
theta = 60 # +/- theta dias de diferenca
# bandar a filtrar com base na magnitude
bandFilter = None #não implementado ainda - não mexer

######### NOME BASE DOS FICHEIROS A SEREM GERADOS #########
filename = fromParamsReturnName(img_collection, ccd_params, (S2_tile, tiles), N, random_state_value)

############ OUTPUTS ######################
FOLDER_OUTPUTS = public_documents / 'output_BDR300'
output_file = FOLDER_OUTPUTS / 'numpy' / "{}.npy".format(filename) # ficheiro numpy (matriz) dos dados (nr de imagens x nr de bandas x nr de pontos)
print(output_file)
#%%
def addNewImageToFile(output_file, tiles, var, S2_tile, BDR_DGT, N, random_state_value, bandas_desejadas, FOLDER_OUTPUTS, img_collection, NODATA_VALUE):
    """
    Carrega o último GeoTIFF, processa os dados geoespaciais e adiciona ao arquivo numpy existente.

    Args:
        - output_file (str): caminho do arquivo numpy onde os dados serão adicionados.
        - tiles (str): diretório onde os arquivos TIFF (raster) estão localizados.
        - var (str): variável que indica a origem dos dados, podendo ser 'THEIA' ou 'GEE'.
        - S2_tile (str): Identificador da tile de Sentinel-2 a ser processado.
        - BDR_DGT (GeoDataFrame): GeoDataFrame contendo as geometrias que serão processadas.
        - N (int): número de pontos a serem processados.
        - random_state_value (int): valor do gerador de números aleatórios.
        - bandas_desejadas (list): lista de bandas desejadas para o processamento.
        - FOLDER_OUTPUTS (str): diretório onde os resultados serão salvos.
        - img_collection (str): coleção de imagens a ser utilizada.
        - NODATA_VALUE (int): valor a ser usado para representar dados ausentes.

    Returns:
        - updated_data (ndarray): array numpy atualizado com os novos dados.
    """
    
    # Processar centros dos pontos de cada geometria para corresponder aos centros dos pixels dos rasters
    raster_path = tiles / 'Theia_T29TNE_20170813-112433.tif'
    gdf_centros_pixeis = processar_centros_pixeis(BDR_DGT, raster_path)
    dados_geoespaciais_metros = readPoints(gdf_centros_pixeis, N, random_state_value)
    
    if var == 'THEIA':
        tif_names, tif_dates = read_tif_files_theia(S2_tile, tiles)
    else:
        tif_names, tif_dates = read_tif_files_gee(S2_tile, tiles)
    
    # Selecionar o último tif e as datas
    tif_names = [os.path.join(tiles, i) for i in tif_names][-1]
    tif_dates_ord = [d.toordinal() for d in tif_dates][-1]
    
    # Carregar o último GeoTIFF
    geotiff_da = rioxarray.open_rasterio(tif_names, chunks={'x': -1, 'y': -1}).sel(band=bandas_desejadas)
    geotiff_da = geotiff_da.expand_dims('time').assign_coords(time=[tif_dates_ord])

    # Coordenadas X e Y dos pontos escolhidos
    points_x_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.x.values).astype('int'), dims=['location'])
    points_y_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.y.values).astype('int'), dims=['location'])

    selection = geotiff_da.sel(x=points_x_int, y=points_y_int, band=bandas_desejadas)
    sel_values = selection.values

    # Carregar o arquivo numpy existente se existir
    if os.path.exists(output_file):
        try:
            existing_data = np.load(output_file)
            print("Arquivo existente carregado com sucesso.")
        except FileNotFoundError:
            print(f"Erro: O arquivo '{output_file}.npy' não foi encontrado.")
            return None
    else:
        print(f"O arquivo '{output_file}' não existe. Criando novo arquivo...")
        existing_data = np.array([], dtype=np.float64)  # Inicializa um array vazio caso não exista
    
    # Adicionar os novos dados ao array existente
    updated_data = np.concatenate((existing_data, sel_values), axis=0)
    
    # Salvar o array atualizado de volta ao arquivo
    np.save(output_file, updated_data)
    print("Nova imagem adicionada e dados salvos com sucesso.")

    return updated_data
#%%
numpy_file = addNewImageToFile(output_file, tiles, var, S2_tile, BDR_DGT, N, random_state_value, bandas_desejadas, FOLDER_OUTPUTS, img_collection, NODATA_VALUE)

