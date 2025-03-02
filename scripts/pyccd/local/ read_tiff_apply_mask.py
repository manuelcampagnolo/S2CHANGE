# PyCCD v01 - 24/02/2025 / Contrato N.ยบ 3044 DGT/ISA/CEXC/2152/2023
import os
import platform

# Verifica o sistema operacional
# Windows
if platform.system() == "Windows":
    user_profile = os.environ['USERPROFILE']
    directory_path = os.path.join(user_profile, 'Desktop', 'S2CHANGE')
else:  # Linux
    user_home = os.path.expanduser("~")
    directory_path = os.path.join(user_home, 'CCD_yml_win')
os.chdir(directory_path)   
import pandas as pd
import rasterio
import os
import sys
from pathlib import Path
# Assumir onde esta a pasta dos scripts do PyCCD
PASTA_DE_SCRIPTS = Path(__name__ ).parent.absolute() / 'scripts' / 'pyccd' 

if PASTA_DE_SCRIPTS not in sys.path:
    sys.path.append(str(PASTA_DE_SCRIPTS))
import ccd
from shared.avaliacao_exatidao_pyccd import runValidation
from datetime import datetime
from shared.processing import runDetectionForPoint, create_geodataframe_from_parquet
from shared.preprocessing import check_or_initialize_file
from shared.utils import fromParamsReturnName, getNumberOfPixelsFromNpy
from shared.plot import plotFromCSV
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')
import os
import h5py
# Working directory (DADOS):
# |----FOLDER PUBLIC DOCUMENTS
#    |---- SUBFOLDER BDR_300 (DGT)
#         |---- file.shp
#    |---- SUBFOLDER BDR_Navigator
#         |---- file.gpkg
#    |---- SUBFOLDER IMAGENS GEE
#         |---- folder TILES
#              |---- files.tif
#    |---- SUBFOLDER IMAGENS THEIA
#         |---- folder TILES
#              |---- files.tif
#    |---- SUBFOLDER output_BDR300
#         |---- folder numpy
#              |---- files.npy
#         |---- folder plots
#              |---- plots.png
#         |---- folder tabular (csv e validaรงรฃo)
#              |---- files.csv
#    |---- SUBFOLDER output_NAV
#         |---- folder numpy
#              |---- files.npy
#         |---- folder plots
#              |---- plots.png
#         |---- folder tabular (csv)
#              |---- files.csv

# Working directory (PyCCD):
# |----FOLDER CCD_yml_win
#    |---- SUBFOLDER S2CHANGE
#       |---- SUBFOLDER scripts
#         |---- SUBFOLDER pyccd_theia
#           |---- SUBFOLDER ccd
#              |---- SUBFOLDER models
#                   |---- __init__.py
#                   |---- lasso.py
#                   |---- robust_fit.py
#                   |---- tmask.py
#              |---- __init__.py
#              |---- app.py
#              |---- change.py
#              |---- math_utils.py
#              |---- parameters.py
#              |---- procedures.py
#              |---- qa.py
#              |---- version.py
#           |---- SUBFOLDER notebooks 
#              |---- addNewImageToFile.py
#              |---- avaliacao_exatidao_pyccd.py
#              |---- main.py (** ficheiro principal **)
#              |---- plot.py
#              |---- processing.py
#              |---- read_files.py
#              |---- utils.py
#%%
# ---------------------------------
#             INPUTS
# ---------------------------------
var = 'GEE' # choose variable: THEIA or GEE
S2_tile = 'T29SPB' # escolher o tile S2

# Caminho onde estรฃo os inputs todos
PASTA_DE_INPUTS = Path('D:/')
# -> Shapefile ou Geopackage que contem a regiao de interesse
# REGIAO_INTERESSE =  PASTA_DE_INPUTS / 'BDR_Navigator' / 'nvg_2018_ccd.gpkg'
REGIAO_INTERESSE = Path('C:/Users/scaetano/Downloads/CCDC_Mask_dissolve.gpkg')
# Nome para identificar a BDR
BDR = "NAV"
# -> IMAGENS SENTINEL:
IMAGENS_S2 = PASTA_DE_INPUTS / 's2_images'

tiles = IMAGENS_S2 / S2_tile
# ---------------------------------
#   PARAMETROS PRE PROCESSAMENTO
# ---------------------------------
min_year =  2017 # ano inicial da corrida do CCD
max_date = datetime(2024, 12, 31) # data atรฉ onde se corre o ccd
#bandas_desejadas = [1, 2, 3, 7, 10] # bandas usadas para o pré-processamento
input_bands = ['B3', 'B4', 'B8', 'B12']
bands_dict = {1:'B3', 2:'B4', 3:'B8', 4:'B12'} # a banda NDVI só é adicionada mais à frente na funcao processPointData
bandas_desejadas = list(bands_dict.keys())

NODATA_VALUE = 65535
MAX_VALUE_NDVI = 10000

EXECUTAR_PLOT = False # (false para nรฃo fazer; true para fazer)
ROW_INDEX = 8 # plot para uma linha do CSV (escolher a linha no row_index)

batch_size = 10#000 # Número de pixels para cada lote

img_collection = tiles.parts[-2]

CRS_THEIA = 32629
CRS_WGS84 = 4326
# ---------------------------------
#            OUTPUTS
# ---------------------------------
PASTA_DE_OUTPUTS = Path('C:/Users/Public/Documents/outputs_RI')
FOLDER_NPY = PASTA_DE_OUTPUTS / 'numpy' / S2_tile
FOLDER_PLOTS = PASTA_DE_OUTPUTS / 'plots' / S2_tile
FOLDER_PARQUET = PASTA_DE_OUTPUTS / 'tabular' / S2_tile
FOLDER_SHP = PASTA_DE_OUTPUTS / 'shapefiles' / S2_tile

# Funรงรฃo para criar diretรณrios se nรฃo existirem
def create_directory_if_not_exists(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Diretorio criado: {path}")
    else:
        print(f"Diretorio ja existe: {path}")

# Criar os diretรณrios
create_directory_if_not_exists(FOLDER_NPY)
create_directory_if_not_exists(FOLDER_PLOTS)
create_directory_if_not_exists(FOLDER_PARQUET)
create_directory_if_not_exists(FOLDER_SHP)

# ---------------------------------
#      PARAMETROS PROCESSAMENTO
# ---------------------------------
# Listar todos os ficheiros na pasta e ordenar pelo tamanho (maior primeiro)
raster_files = sorted(tiles.glob('*.*'), key=lambda f: f.stat().st_size, reverse=True)

# Se a lista de arquivos não estiver vazia, escolher o maior arquivo
raster_path = None
if raster_files:
    largest_file = raster_files[0]
    try:
        with rasterio.open(largest_file) as src:
            if src.read(1).size > 0:  # Verificar se a imagem tem dados validos
                raster_path = largest_file
    except:
        raster_path = None  # Se houver erro ao abrir, nada é selecionado

# Imprimir o tiff selecionado
if raster_path:
    print("Imagem selecionada:", raster_path)
else:
    print("Nenhuma imagem valida foi encontrada.")
# ---------------------------------
#          PARAMETROS CCD
# ---------------------------------
alpha = ccd.parameters.defaults['ALPHA'] # Looks for alpha in the parameters.py file
ccd_params = ccd.parameters.defaults
######### NOME BASE DOS FICHEIROS A SEREM GERADOS #########
filename = fromParamsReturnName(img_collection, ccd_params, (S2_tile, tiles), BDR, min_year, max_date)
############ OUTPUTS ######################
output_file = FOLDER_NPY / "{}.h5".format(filename) # ficheiro numpy (matriz) dos dados (nr de imagens x nr de bandas x nr total de pontos)

# ---------------------------------
#      PARAMETROS DA VALIDAรรO
# ---------------------------------
# datas do filtro das datas da anรกlise (DGT 300)
########### Nรฃo alterar ################
dt_ini = '2018-09-12' # data inicial
dt_end = '2021-09-30' # data final
# Margem de tolerรขncia entre a quebra do Modelo e do Analista
theta = 60 # +/- theta dias de diferenca
# banda a filtrar com base na magnitude
bandFilter = None #nรฃo implementado ainda - nรฃo mexer
#%%
# Função auxiliar para processar um batch
def process_batch(args):
    """
    Processes a batch of points for change detection.

    This function takes a batch of selected values and processes each point in the batch 
    by calling the `runDetectionForPoint` function. It creates a list of arguments for each 
    point in the batch and returns the results for all points.

    Args:
        - args (tuple): A tuple containing the following elements:
            - i (int): Index of the point being processed within the batch. Used to iterate through each point in the batch.
            - sel_values_block (ndarray): 3D array of selected values for the batch (number of images x number of bands x batch_size).
            - xs_slice (ndarray): Array of x coordinates for the batch.
            - ys_slice (ndarray): Array of y coordinates for the batch.
            - tif_dates_ord (ndarray): Array of ordered dates corresponding to the images.
            - NODATA_VALUE (float): Value representing missing data in the dataset.
            - MAX_VALUE_NDVI (float): Maximum possible NDVI value used for scaling.
            - PASTA_DE_OUTPUTS (str): Directory to save the output results.
            - CRS_THEIA (str): Coordinate Reference System (CRS) for the Theia dataset.
            - CRS_WGS84 (str): Coordinate Reference System (CRS) for the WGS84 standard.
            - img_collection (ndarray): Collection of Sentinel-2 images.

    Returns:
        - list: A list of results from `runDetectionForPoint` for each point in the batch.
    """
    sel_values_block, xs_slice, ys_slice, tif_dates_ord = args
    arg_list = [
        (i, sel_values_block, tif_dates_ord, xs_slice, ys_slice, NODATA_VALUE, MAX_VALUE_NDVI, PASTA_DE_OUTPUTS, CRS_THEIA, CRS_WGS84, img_collection)
        for i in range(sel_values_block.shape[2])]
    # Retorna resultados para todos os pontos no batch
    return [runDetectionForPoint(arg) for arg in arg_list]
#%%
def main(batch_size):
    # Verificar a existência do arquivo .npy e inicializar ou carregar os dados
    tif_dates_ord, N = check_or_initialize_file(output_file, tiles, var, S2_tile, min_year, max_date, REGIAO_INTERESSE, 
                                                bandas_desejadas, PASTA_DE_OUTPUTS, img_collection, NODATA_VALUE, raster_path)
    
    # Carregar os dados numpy para o processamento em lotes
    h5_file = h5py.File(output_file, 'r')
    sel_values = h5_file['values']
    xs = h5_file['xs']
    ys = h5_file['ys']
    
    # Criar os batches
    batches = [
        (sel_values[:, :, start:end], xs[start:end], ys[start:end], tif_dates_ord)
        for start, end in zip(range(0, N, batch_size), range(batch_size, N + batch_size, batch_size))
    ]
    
    dfs = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        tqdm_bar = tqdm(total=len(batches))
        
        # Processar os batches paralelamente
        for batch_results in executor.map(process_batch, batches):
            dfs.extend(batch_results)  # Adiciona os resultados processados
            tqdm_bar.update(1)
        
        tqdm_bar.close()
    
    # Concatenar os resultados de todos os lotes num único DataFrame
    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)
        for col in result_df.columns:
            if result_df[col].apply(lambda x: isinstance(x, list)).any():
                result_df = result_df.explode(col)
        print(f"Salvando o ficheiro parquet com {len(result_df)} registos.")
        result_df.to_parquet(FOLDER_PARQUET / '{}.parquet'.format(filename), index=False)
    
    if EXECUTAR_PLOT:
        plotFromCSV(FOLDER_PARQUET / '{}.parquet'.format(filename), ROW_INDEX, save_dir=FOLDER_PLOTS / '{}_RowIndex{}.png'.format(filename, ROW_INDEX))
#%%
# Executar o código
if __name__ == '__main__':
    main(batch_size)
    # create_geodataframe_from_csv(filename, CRS_WGS84, CRS_THEIA, S2_tile, FOLDER_CSV, FOLDER_SHP)
    # if BDR == 'DGT':
    #     runValidation(filename, FOLDER_CSV, REGIAO_INTERESSE, dt_ini, dt_end, bandFilter, theta)
