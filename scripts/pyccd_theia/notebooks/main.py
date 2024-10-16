# PyCCD v01 - 16/10/2024 / Contrato N.º 3044 DGT/ISA/CEXC/2152/2023
import os
import platform

# Verifica o sistema operacional
if platform.system() == "Windows":
    user_profile = os.environ['USERPROFILE']
    directory_path = os.path.join(user_profile, 'Desktop', 'CCD_yml_win')
else:  # Assume que é Linux
    user_home = os.path.expanduser("~")
    directory_path = os.path.join(user_home, 'CCD_yml_win')
os.chdir(directory_path)

import pandas as pd
import os
import sys
from pathlib import Path
# Assumir onde está a pasta dos scripts do PyCCD
PASTA_DE_SCRIPTS = Path(__name__ ).parent.absolute() / 'S2CHANGE' / 'scripts' / 'pyccd_theia' 

if PASTA_DE_SCRIPTS not in sys.path:
    sys.path.append(str(PASTA_DE_SCRIPTS))
import ccd
from notebooks.avaliacao_exatidao_pyccd import runValidation
from datetime import datetime
from notebooks.processing import check_or_initialize_file, runDetectionForPoint, processar_centros_pixeis, create_geodataframe_from_csv
from notebooks.utils import fromParamsReturnName
from notebooks.plot import plotFromCSV
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import os
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
#         |---- folder tabular (csv e validação)
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
var = 'Theia' # choose variable: Theia or GEE
S2_tile = 'T29TNE' # escolher o tile S2

# Caminho onde estão os inputs todos
PASTA_DE_INPUTS = Path('C:/Users/Public/Documents/')
# -> Shapefile ou Geopackage que contem a regiao de interesse
REGIAO_INTERESSE =  PASTA_DE_INPUTS / 'BDR_300_artigo' / 'BDR_CCDC_TNE_Adjusted.shp'

# -> IMAGENS SENTINEL:
IMAGENS_S2 = PASTA_DE_INPUTS / f'imagens_{str(var)}'

tiles = IMAGENS_S2 / S2_tile

# ---------------------------------
#   PARAMETROS PRÉ PROCESSAMENTO
# ---------------------------------
min_year =  2017 # ano inicial da corrida do CCD
max_date = datetime(2023, 12, 31) # data até onde se corre o ccd
bandas_desejadas = [1, 2, 3, 7, 10] # bandas usadas para o pré-processamento

NODATA_VALUE = 65535
MAX_VALUE_NDVI = 10000

EXECUTAR_PLOT = False # (false para não fazer; true para fazer)
ROW_INDEX = 8 # plot para uma linha do CSV (escolher a linha no row_index)

# ---------------------------------
#            OUTPUTS
# ---------------------------------
PASTA_DE_OUTPUTS = Path('C:/Users/Public/Documents/outputs_RI')

FOLDER_NPY = PASTA_DE_OUTPUTS / 'numpy' / S2_tile
FOLDER_PLOTS = PASTA_DE_OUTPUTS / 'plots' / S2_tile
FOLDER_CSV = PASTA_DE_OUTPUTS / 'tabular' / S2_tile
FOLDER_SHP = PASTA_DE_OUTPUTS / 'shapefiles' / S2_tile

# Função para criar diretórios se não existirem
def create_directory_if_not_exists(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Diretório criado: {path}")
    else:
        print(f"Diretório já existe: {path}")

# Criar os diretórios
create_directory_if_not_exists(FOLDER_NPY)
create_directory_if_not_exists(FOLDER_PLOTS)
create_directory_if_not_exists(FOLDER_CSV)
create_directory_if_not_exists(FOLDER_SHP)

# ---------------------------------
#      PARAMETROS PROCESSAMENTO
# ---------------------------------
raster_path = next(tiles.glob('*.*'), None)

gdf_centros_pixeis = processar_centros_pixeis(REGIAO_INTERESSE, raster_path)

N = len(gdf_centros_pixeis) # Número total de pixels
random_state_value = 42
batch_size = 10000  # Tamanho do lote
num_batches = math.ceil(N / batch_size)  # Número de lotes necessários

img_collection = tiles.parts[-2]

CRS_THEIA = 32629
CRS_WGS84 = 4326

# ---------------------------------
#          PARAMETROS CCD
# ---------------------------------
alpha = ccd.parameters.defaults['ALPHA'] # Looks for alpha in the parameters.py file
ccd_params = ccd.parameters.defaults
######### NOME BASE DOS FICHEIROS A SEREM GERADOS #########
filename = fromParamsReturnName(img_collection, ccd_params, (S2_tile, tiles), N, random_state_value, min_year, max_date)
############ OUTPUTS ######################
output_file = FOLDER_NPY / "{}.npy".format(filename) # ficheiro numpy (matriz) dos dados (nr de imagens x nr de bandas x nr total de pontos)

# ---------------------------------
#      PARAMETROS DA VALIDAÇÃO
# ---------------------------------
# datas do filtro das datas da análise (DGT 300)
########### Não alterar ################
dt_ini = '2018-09-12' # data inicial
dt_end = '2021-09-30' # data final
# Margem de tolerância entre a quebra do Modelo e do Analista
theta = 60 # +/- theta dias de diferenca
# bandar a filtrar com base na magnitude
bandFilter = None #não implementado ainda - não mexer
#%%
def main(batch_size=None):
    # Verificar a existência do arquivo .npy e inicializar ou carregar os dados
    # tif_dates_ord = check_or_initialize_file(output_file, tiles, var, S2_tile, max_date, BDR_FILE, N, random_state_value, bandas_desejadas, FOLDER_OUTPUTS, img_collection, NODATA_VALUE, raster_path)
    tif_dates_ord = check_or_initialize_file(output_file, tiles, var, S2_tile, min_year, max_date, gdf_centros_pixeis, N, random_state_value, bandas_desejadas, PASTA_DE_OUTPUTS, img_collection, NODATA_VALUE, raster_path)
    
    # Carregar os dados numpy para o processamento em lotes
    sel_values = np.load(output_file, mmap_mode='r')

    xs = np.load(str(output_file.with_suffix('')) + '_xs.npy', mmap_mode='r')
    ys = np.load(str(output_file.with_suffix('')) + '_ys.npy', mmap_mode='r')
    
    # Executar o processamento em lotes com ProcessPoolExecutor
    dfs = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        tqdm_bar = tqdm(total=num_batches)
        
        for batch_index, start_index in enumerate(range(0, N, batch_size)):
            end_index = min(start_index + batch_size, N)
            
            # Carregar apenas o bloco atual de pontos
            sel_values_block = sel_values[:, :, start_index:end_index]
            xs_slice = xs[start_index:end_index]
            ys_slice = ys[start_index:end_index]
            
            # Criar argumentos para cada lote de 10 000 pontos
            arg_list = [(i, sel_values_block, tif_dates_ord, xs_slice, ys_slice, NODATA_VALUE, MAX_VALUE_NDVI, PASTA_DE_OUTPUTS, CRS_THEIA, CRS_WGS84, img_collection) for i in range(sel_values_block.shape[2])]
            
            # Mapear os resultados usando executor.map
            for result_df in executor.map(runDetectionForPoint, arg_list):
                dfs.append(result_df)
                tqdm_bar.update(1)
        
        tqdm_bar.close()
    
    # Concatenar os resultados de todos os lotes em um único DataFrame
    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)
        result_df.to_csv(FOLDER_CSV / '{}.csv'.format(filename), index=False)
    
    if EXECUTAR_PLOT:
        plotFromCSV(FOLDER_CSV / '{}.csv'.format(filename), ROW_INDEX, save_dir=FOLDER_PLOTS / '{}_RowIndex{}.png'.format(filename, ROW_INDEX))
#%%
if __name__ == '__main__':
    main(batch_size)
    # if BDR == 'DGT':
    #     runValidation(filename, FOLDER_CSV, REGIAO_INTERESSE, dt_ini, dt_end, bandFilter, theta)
    create_geodataframe_from_csv(filename, CRS_WGS84, CRS_THEIA, S2_tile, FOLDER_CSV, FOLDER_SHP)