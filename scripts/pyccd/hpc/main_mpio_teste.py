# -*- coding: utf-8 -*-
# TESTE HPC MPI COM HDF5 ---> INCD
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
from shared.preprocessing import check_or_initialize_file
from shared.processing import runDetectionForPoint#, create_geodataframe_from_parquet
from shared.utils import fromParamsReturnName#, getNumberOfPixelsFromNpy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from mpi4py import MPI
import platform
from multiprocessing import Manager
from pathlib import Path
import h5py
#%%
# ---------------------------------
#             INPUTS
# ---------------------------------
var = 'GEE' # choose variable: THEIA or GEE
BDR = 'NAV' # choose variable: DGT or NAV
S2_tile = 'T29SPB' # escolher o tile S2
# Caminho onde estao os dados todos
public_documents = Path('/users1/cpca070342024/scaetano')
# Caminhos para a base de dados de validacao
BDR_FILE = Path('/users1/cpca070342024/scaetano/CCDC_Mask_dissolve.gpkg')
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
max_date = datetime(2024, 12, 31) # data atÃ© onde se corre o ccd

input_bands=['B3', 'B4', 'B8', 'B12']
bands_dict={1: 'NDVI', 2: 'B3', 3:'B4', 4: 'B8', 5:'B12'}
bandas_desejadas = bands_dict.keys() # to check

NODATA_VALUE = 65535
MAX_VALUE_NDVI = 10000

EXECUTAR_PLOT = False # (false para nÃ£o fazer; true para fazer)
ROW_INDEX = 8 # plot para uma linha do CSV (escolher a linha no row_index)

BATCH_SIZE = 1000  # Ajustar o tamanho do lote para processamento em paralelo

img_collection = tiles.parts[-2]

CRS_THEIA = 32629
CRS_WGS84 = 4326
# ---------------------------------
#            OUTPUTS
# ---------------------------------
FOLDER_OUTPUTS = Path('/users1/cpca070342024/scaetano/outputs_RI')

FOLDER_NPY = FOLDER_OUTPUTS / 'numpy' / S2_tile
FOLDER_PLOTS = FOLDER_OUTPUTS / 'plots' / S2_tile
FOLDER_PARQUET = FOLDER_OUTPUTS / 'tabular' / S2_tile
FOLDER_SHP = FOLDER_OUTPUTS / 'shapefiles' / S2_tile

# FunÃ§Ã£o para criar diretÃ³rios se nÃ£o existirem
def create_directory_if_not_exists(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# Criar os diretÃ³rios
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
            if src.read(1).size > 0:  # Verificar se a imagem tem dados vÃ¡lidos
                raster_path = largest_file
    except:
        raster_path = None  # Se houver erro ao abrir, nada Ã© selecionado
# ---------------------------------
#          PARAMETROS CCD
# ---------------------------------
alpha = ccd.parameters.defaults['ALPHA'] # Looks for alpha in the parameters.py file
ccd_params = ccd.parameters.defaults
######### NOME BASE DOS FICHEIROS A SEREM GERADOS #########
filename = fromParamsReturnName(img_collection, ccd_params, (S2_tile, tiles), BDR, min_year, max_date)
############ OUTPUTS ######################
output_file = FOLDER_NPY / "{}.h5".format(filename) # ficheiro numpy (matriz) dos dados (nr de imagens x nr de bandas x nr total de pontos)
#%%
def main(batch_size=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # root rank
        tif_dates_ord, N = check_or_initialize_file(
            output_file, tiles, var, S2_tile, min_year, max_date, BDR_FILE,
            bandas_desejadas, FOLDER_OUTPUTS, img_collection, NODATA_VALUE, raster_path
        )
        
        # Split data indexes
        indices = list(range(0, N, batch_size))
        batches = [(start, min(start + batch_size, N)) for start in indices]
        print(f"[Rank {rank}] Total batches created: {len(batches)}")

        # Split batches between rans
        batches_per_rank = [batches[i::size] for i in range(size)]
    else:
        tif_dates_ord = None
        batches_per_rank = None

    # Share data between processes
    tif_dates_ord = comm.bcast(tif_dates_ord, root=0)
    my_batches = comm.scatter(batches_per_rank, root=0)

    # Create shared progress bar
    with Manager() as manager:
        progress = manager.Value('i', 0)  # Shared counter
        total_batches = len(my_batches)  # Total number of lots

        local_results = []

        with tqdm(total=total_batches, desc=f"Processo {rank}") as pbar:
            def update_progress(_):
                pbar.n = progress.value  # Update progress bar with shared value
                pbar.refresh()

            for batch in my_batches:
                result = process_batch(batch, output_file, tif_dates_ord, progress)
                local_results.extend(result)
                with progress.get_lock():
                    progress.value += 1
                update_progress(None)

    # Saving results locally per process
    if local_results:
        result_df = pd.concat(local_results, ignore_index=True)
        result_df.to_parquet(FOLDER_PARQUET / f'{filename}_rank_{rank}.parquet', index=False)

    # Sync all ranks before continuing
    comm.Barrier()
    
    if rank == 0:
        print(f"All batches were processed by {size} ranks.")

# Function to process a single batch
def process_batch(batch, output_file, tif_dates_ord):
    start, end = batch
    h5_file = h5py.File(output_file, 'r')
    # Load only the specific block for the batch
    sel_values_block = h5_file['values'][:, :, start:end]
    xs_slice = h5_file['xs'][start:end]
    ys_slice = h5_file['ys'][start:end]

    # Create the list of arguments for processing
    arg_list = [
        (i, sel_values_block, tif_dates_ord, xs_slice, ys_slice, NODATA_VALUE, MAX_VALUE_NDVI, FOLDER_OUTPUTS, CRS_THEIA, CRS_WGS84, img_collection)
        for i in range(sel_values_block.shape[2])
    ]
    
    # Process the specific batch
    return [runDetectionForPoint(arg) for arg in arg_list]

if __name__ == '__main__':
    # Call the main function with the batch size
    main(BATCH_SIZE)
