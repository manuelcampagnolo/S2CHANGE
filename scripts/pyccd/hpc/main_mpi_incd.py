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
import os
import sys
from pathlib import Path
# Assumir onde esta a pasta dos scripts do PyCCD
PASTA_DE_SCRIPTS = Path(__name__ ).parent.absolute() / 'scripts' / 'pyccd' 

if PASTA_DE_SCRIPTS not in sys.path:
    sys.path.append(str(PASTA_DE_SCRIPTS))
import ccd
from datetime import datetime
from shared.processing import check_or_initialize_file, runDetectionForPoint, create_geodataframe_from_csv
from shared.utils import fromParamsReturnName, getNumberOfPixelsFromNpy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
from mpi4py import MPI
import os
import platform
from multiprocessing import Manager
import os
from pathlib import Path

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
import os
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
#%% 
# ---------------------------------
#   PARAMETROS PRÃ‰ PROCESSAMENTO
# ---------------------------------
var = 'THEIA' # choose variable: THEIA or GEE
BDR = 'DGT' # choose variable: DGT or NAV
S2_tile = 'T29TNE' # escolher o tile S2
min_year =  2017 # ano inicial da corrida do CCD
max_date = datetime(2023, 12, 31) # data até onde se corre o ccd
input_bands=['B3', 'B4', 'B8', 'B12']
bands_dict={1: 'NDVI', 2: 'B3', 3:'B4', 4: 'B8', 5:'B12'}
bandas_desejadas = bands_dict.keys() # to check

NODATA_VALUE = 65535
MAX_VALUE_NDVI = 10000

EXECUTAR_PLOT = False # (false para não fazer; true para fazer)
ROW_INDEX = 8 # plot para uma linha do CSV (escolher a linha no row_index)

# ---------------------------------
#             INPUTS
# ---------------------------------
# Caminho onde estão os dados todos
public_documents = Path('C:/Users/Public/Documents/')
# Caminhos para a base de dados de validação
# -> BDR DGT:
BDR_DGT = public_documents / 'BDR_300_artigo' / 'BDR_CCDC_TNE_Adjusted.shp'
# -> BDR NAVIGATOR:
BDR_NAVIGATOR =  public_documents / 'BDR_Navigator' / 'nvg_2018_ccd.gpkg'

# -> IMAGENS SENTINEL:
FOLDER_THEIA = public_documents / 'imagens_Theia' # Caminho dados THEIA
FOLDER_GEE = public_documents / 'imagens_GEE' # Caminho dados GEE

if var == 'THEIA':
    tiles = FOLDER_THEIA / S2_tile
else:
    tiles = FOLDER_GEE / S2_tile

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
FOLDER_CSV = FOLDER_OUTPUTS / 'tabular' / S2_tile
FOLDER_SHP = FOLDER_OUTPUTS / 'shapefiles' / S2_tile

# Função para criar diretórios se não existirem
def create_directory_if_not_exists(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# Criar os diretórios
create_directory_if_not_exists(FOLDER_NPY)
create_directory_if_not_exists(FOLDER_PLOTS)
create_directory_if_not_exists(FOLDER_CSV)
create_directory_if_not_exists(FOLDER_SHP)

# ---------------------------------
#      PARAMETROS PROCESSAMENTO
# ---------------------------------
# Escolher um raster com tamanho suficiente grande de forma apanhar todos os pixels:
# Converter 800MB para bytes (1MB = 1_048_576 bytes)
min_size = 800 * 1_048_576  

# Listar todos os ficheiros na pasta
raster_files = sorted(tiles.glob('*.*'))

# Filtrar ficheiros maiores que 800MB
raster_files = [f for f in raster_files if f.stat().st_size > min_size]

# Escolher a primeira imagem válida
raster_path = None
for f in raster_files:
    try:
        with rasterio.open(f) as src:
            if src.read(1).size > 0:
                raster_path = f
                break  
    except:
        continue  

print("Imagem selecionada:", raster_path)



img_collection = tiles.parts[-2]

CRS_THEIA = 32629
CRS_WGS84 = 4326

# ---------------------------------
#          PARAMETROS CCD
# ---------------------------------
alpha = ccd.parameters.defaults['ALPHA'] # Looks for alpha in the parameters.py file
ccd_params = ccd.parameters.defaults
######### NOME BASE DOS FICHEIROS A SEREM GERADOS #########
filename = fromParamsReturnName(img_collection, ccd_params, (S2_tile, tiles), BDR, min_year, max_date)
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
def process_single_batch(batch, sel_values_path, xs_path, ys_path, tif_dates_ord, progress):
    start, end = batch

    # Carregar apenas o bloco específico para o lote
    sel_values_block = np.load(sel_values_path, mmap_mode='r')[:, :, start:end]
    xs_slice = np.load(xs_path, mmap_mode='r')[start:end]
    ys_slice = np.load(ys_path, mmap_mode='r')[start:end]

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
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [
                    executor.submit(
                        process_single_batch,
                        batch,
                        output_file,  # Caminho do sel_values
                        str(output_file.with_suffix('')) + '_xs.npy',  # Caminho do xs
                        str(output_file.with_suffix('')) + '_ys.npy',  # Caminho do ys
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
        # Cria um CSV para cada processo
        rank_csv_filename = FOLDER_CSV / f'{filename}_rank_{rank}.csv'
        result_df = pd.concat(dfs, ignore_index=True)

        result_df.to_csv(rank_csv_filename, index=False)

    comm.Barrier()  # Sincronizar todos os ranks antes de continuar

    if rank == 0:
        all_csv_files = list(FOLDER_CSV.glob(f'{filename}_rank_*.csv'))
        
        if not all_csv_files:
            raise FileNotFoundError(f"Nenhum arquivo encontrado correspondente ao padrao {filename}_rank_*.csv em {FOLDER_CSV}")
        
        for csv_filename in all_csv_files:
            try:
                csv_file = csv_filename.stem
                # Função para criar o shapefile para cada CSV de cada processo
                create_geodataframe_from_csv(
                    csv_file, CRS_WGS84, CRS_THEIA, S2_tile, FOLDER_CSV, FOLDER_SHP
                )
                
            except Exception as e:
                print(f"Erro ao processar o arquivo {csv_file}: {e}")
        
        print(f"Todos os shapefiles individuais foram criados em {FOLDER_SHP}.")

if __name__ == '__main__':
    batch_size = 1000  # Ajustar o tamanho do lote
    n = getNumberOfPixelsFromNpy(output_file)
    if rank == 0:
        print(f"Numero total de pixels processados: {n}")
        print(f"Executando com batch_size = {batch_size} e n = {n}")
        print(f'Numero de CPUs para o ProcessPoolExecutor: {os.cpu_count()}')
    main(batch_size)
