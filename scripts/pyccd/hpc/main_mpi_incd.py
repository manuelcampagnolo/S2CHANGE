# -*- coding: utf-8 -*-
import os
import platform

# Verifica o sistema operacional
# Windows
if platform.system() == "Windows":
    user_profile = os.environ['USERPROFILE']
    directory_path = os.path.join(user_profile, 'Desktop', 'S2CHANGE')
else:  # Linux
    user_home = os.path.expanduser("~")
    directory_path = os.path.join(user_home, 'S2CHANGE')
os.chdir(directory_path)
import pandas as pd
import rasterio
import sys
from pathlib import Path
# Assumir onde esta a pasta dos scripts do PyCCD
PASTA_DE_SCRIPTS = Path(__name__ ).parent.absolute() / 'scripts' / 'pyccd' 

if PASTA_DE_SCRIPTS not in sys.path:
    sys.path.append(str(PASTA_DE_SCRIPTS))
import ccd
from datetime import datetime
from shared.processing import check_or_initialize_file, runDetectionForPoint, create_geodataframe_from_parquet
from shared.utils import fromParamsReturnName, getNumberOfPixelsFromNpy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from mpi4py import MPI
import platform
from multiprocessing import Manager
from pathlib import Path
import h5py
cpus_slurm = int(os.getenv('SLURM_NTASKS', os.cpu_count()))

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
#         |---- folder tabular (csv e validaÃ§Ã£o)
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
#    |---- SUBFOLDER scripts
#    |---- SUBFOLDER pyccd_theia
#         |---- SUBFOLDER ccd
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
#         |---- SUBFOLDER notebooks 
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
var = 'THEIA' # choose variable: THEIA or GEE
BDR = 'DGT' # choose variable: DGT or NAV
S2_tile = 'T29TNE' # escolher o tile S2
# Caminho onde estão os dados todos
public_documents = Path('/projects/F202410004CPCAA1/')
# Caminhos para a base de dados de validação
# -> BDR DGT:
BDR_DGT = public_documents / 'BDR_300_artigo' / 'BDR_CCDC_TNE_Adjusted.shp'
# -> BDR NAVIGATOR:
BDR_NAVIGATOR =  public_documents / 'BDR_Navigator' / 'nvg_2018_ccd.gpkg'

# -> IMAGENS SENTINEL:
FOLDER_THEIA = public_documents / 'imagens_Theia' # Caminho dados THEIA
FOLDER_GEE = public_documents / 's2_images' # Caminho dados GEE

if var == 'THEIA':
    tiles = FOLDER_THEIA / S2_tile
else:
    tiles = FOLDER_GEE / S2_tile
#%% 
# ---------------------------------
#   PARAMETROS PRE PROCESSAMENTO
# ---------------------------------
min_year =  2017 # ano inicial da corrida do CCD
max_date = datetime(2023, 12, 31) # data até onde se corre o ccd

input_bands=['B3', 'B4', 'B8', 'B12']
bands_dict={1: 'NDVI', 2: 'B3', 3:'B4', 4: 'B8', 5:'B12'}
bandas_desejadas = bands_dict.keys() # to check

NODATA_VALUE = 65535
MAX_VALUE_NDVI = 10000

EXECUTAR_PLOT = False # (false para não fazer; true para fazer)
ROW_INDEX = 8 # plot para uma linha do CSV (escolher a linha no row_index)

BATCH_SIZE = 1000  # Ajustar o tamanho do lote para processamento em paralelo

img_collection = tiles.parts[-2]

CRS_THEIA = 32629
CRS_WGS84 = 4326
# ---------------------------------
#            OUTPUTS
# ---------------------------------
if BDR == 'DGT':
    BDR_FILE = BDR_DGT
    FOLDER_OUTPUTS = public_documents / 'output_BDR300'
else:
    BDR_FILE = BDR_NAVIGATOR
    FOLDER_OUTPUTS = public_documents / 'output_BDR-NAV'

FOLDER_NPY = FOLDER_OUTPUTS / 'numpy' / S2_tile
FOLDER_PLOTS = FOLDER_OUTPUTS / 'plots' / S2_tile
FOLDER_PARQUET = FOLDER_OUTPUTS / 'tabular' / S2_tile
FOLDER_SHP = FOLDER_OUTPUTS / 'shapefiles' / S2_tile

# Função para criar diretórios se não existirem
def create_directory_if_not_exists(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# Criar os diretórios
create_directory_if_not_exists(FOLDER_NPY)
create_directory_if_not_exists(FOLDER_PLOTS)
create_directory_if_not_exists(FOLDER_PARQUET)
create_directory_if_not_exists(FOLDER_SHP)

# ---------------------------------
#      PARAMETROS PROCESSAMENTO
# ---------------------------------
raster_files = sorted(tiles.glob('*.*'), key=lambda f: f.stat().st_size, reverse=True)

raster_path = None
if raster_files:
    largest_file = raster_files[0] 
    try:
        with rasterio.open(largest_file) as src:
            if src.read(1).size > 0:  # Verificar se a imagem tem dados válidos
                raster_path = largest_file
    except:
        raster_path = None  # Se houver erro ao abrir, nada é selecionado

# Imprimir o tiff selecionado
if raster_path:
    print("Imagem selecionada:", raster_path)
else:
    print("Nenhuma imagem válida foi encontrada.")
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
#      PARAMETROS DA VALIDAÇÃO
# ---------------------------------
# datas do filtro das datas da análise (DGT 300)
########### Não alterar ################
dt_ini = '2018-09-12' # data inicial
dt_end = '2021-09-30' # data final
# Margem de tolerância entre a quebra do Modelo e do Analista
theta = 60 # +/- theta dias de diferença
# bandar a filtrar com base na magnitude
bandFilter = None #não implementado ainda - não mexer
#%% Configurações MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#%%
# Função para processar um lote
def process_batch(args):
    sel_values_block, xs_slice, ys_slice, tif_dates_ord = args
    arg_list = [
        (i, sel_values_block, tif_dates_ord, xs_slice, ys_slice, NODATA_VALUE, MAX_VALUE_NDVI, FOLDER_OUTPUTS, CRS_THEIA, CRS_WGS84, img_collection)
        for i in range(sel_values_block.shape[2])
    ]
    return [runDetectionForPoint(arg) for arg in arg_list]

# Função para processar um lote
def process_single_batch(batch, sel_values_path, tif_dates_ord, progress):
    start, end = batch

    # Carregar apenas o bloco específico para o lote
    h5_file = h5py.File(sel_values_path, 'r')
    sel_values_block = h5_file['values'][:, :, start:end]
    xs_slice = h5_file['xs'][start:end]
    ys_slice = h5_file['ys'][start:end]

    # Processar o bloco específico
    result = process_batch((sel_values_block, xs_slice, ys_slice, tif_dates_ord))

    # Atualizar a barra de progresso compartilhada
    progress.value += 1
    return result

def main(batch_size=None):
    if rank == 0:
        # Processo mestre
        tif_dates_ord, N = check_or_initialize_file(
            output_file, tiles, var, S2_tile, min_year, max_date, BDR_FILE,
            bandas_desejadas, FOLDER_OUTPUTS, img_collection, NODATA_VALUE, raster_path
        )

        # Dividir os índices de dados
        indices = list(range(0, N, batch_size))
        batches = [
            (start, min(start + batch_size, N)) for start in indices
        ]
        print(f"[Rank {rank}] Total de batches criados: {len(batches)}")

        # Dividir lotes entre ranks
        batches_per_rank = [batches[i::size] for i in range(size)]
    else:
        tif_dates_ord = None
        batches_per_rank = None

    # Compartilhar dados entre processos
    tif_dates_ord = comm.bcast(tif_dates_ord, root=0)
    my_batches = comm.scatter(batches_per_rank, root=0)

    # Criar uma barra de progresso compartilhada
    with Manager() as manager:
        progress = manager.Value('i', 0)  # Contador compartilhado
        total_batches = len(my_batches)  # Número total de lotes

        with tqdm(total=total_batches, desc=f"Processo {rank}") as pbar:
            def update_progress(_):
                pbar.n = progress.value  # Atualiza a barra de progresso com o valor compartilhado
                pbar.refresh()

            # Processar os lotes atribuídos usando ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=cpus_slurm) as executor:
                futures = [
                    executor.submit(
                        process_single_batch,
                        batch,
                        output_file,  # Caminho do sel_values
                        tif_dates_ord,
                        progress
                    )
                    for batch in my_batches
                ]
                for future in futures:
                    future.add_done_callback(update_progress)

        local_results = [future.result() for future in futures]

    # Salvar os resultados localmente por processo
    dfs = [df for results in local_results for df in results]
    if dfs:
        # Cria um Parquet para cada processo
        rank_parquet_filename = FOLDER_PARQUET / f'{filename}_rank_{rank}.parquet'
        result_df = pd.concat(dfs, ignore_index=True)
        for col in result_df.columns:
            if result_df[col].apply(lambda x: isinstance(x, list)).any():
                result_df = result_df.explode(col)
        result_df.to_parquet(rank_parquet_filename, index=False)

    comm.Barrier()  # Sincronizar todos os ranks antes de continuar

    if rank == 0:
        all_parquet_files = list(FOLDER_PARQUET.glob(f'{filename}_rank_*.parquet'))
        
        if not all_parquet_files:
            raise FileNotFoundError(f"Nenhum arquivo encontrado correspondente ao padrao {filename}_rank_*.parquet em {FOLDER_PARQUET}")
        
        for parquet_filename in all_parquet_files:
            try:
                parquet_file = parquet_filename.stem
                # Função para criar o shapefile para cada Parquet de cada processo
                create_geodataframe_from_parquet(
                    parquet_file, CRS_WGS84, CRS_THEIA, S2_tile, FOLDER_PARQUET, FOLDER_SHP
                )
                
            except Exception as e:
                print(f"Erro ao processar o arquivo {parquet_file}: {e}")
        
        print(f"Todos os shapefiles individuais foram criados em {FOLDER_SHP}.")

if __name__ == '__main__':
    n = getNumberOfPixelsFromNpy(output_file)
    if rank == 0:
        print(f"Numero total de pixels processados: {n}")
        print(f"Executando com batch_size = {BATCH_SIZE} e n = {n}")
        print(f'Numero de CPUs para o ProcessPoolExecutor: {os.cpu_count()}')
    main(BATCH_SIZE)
