import os
user_profile = os.environ['USERPROFILE']
directory_path = os.path.join(user_profile, 'Desktop', 'CCD_yml_win')
os.chdir(directory_path)
import geopandas as gpd
import pandas as pd
import os
import sys
from pathlib import Path
# chamar python a partir da pasta 'CCD'
module_path= Path(__name__ ).parent.absolute() / 'S2CHANGE' / 'scripts' / 'pyccd_theia' #  / 'CCD' / 'S2CHANGE' / 'scripts' / 
base_path= Path(__name__ ).parent.absolute()  # dir do script; # dir referência (acima): 'DGT-S2CHANGE_2023'
if module_path not in sys.path:
    sys.path.append(str(module_path))
import ccd
from notebooks.avaliacao_exatidao_pyccd import runValidation
from notebooks.processing import check_or_initialize_file, runDetectionForPoint
from notebooks.utils import fromParamsReturnName
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import os
#%%
############ INPUTS ######################
public_documents = Path('C:/Users/Public/Documents/')
# -> BDR DGT:
BDR_DGT = public_documents / 'BDR_300_artigo' / 'BDR_CCDC_TNE_Adjusted.shp' # Caminho para a base de dados de validação
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

num_pixels = N  # Número total de pixels
batch_size = 10000  # Tamanho do lote
num_batches = math.ceil(num_pixels / batch_size)  # Número de lotes necessários

bandas_desejadas = [1, 2, 3, 7, 9, 10]
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
output_file = f"{var}_sel_values_N{N}_RS{random_state_value}.npy" # ficheiro numpy (matriz) dos dados (nr de imagens x nr de bandas x nr de pontos)
#%%
def main(batch_size=None):
    # Definir o nome do arquivo numpy e outras variáveis necessárias
    output_file = f"{var}_sel_values_N{N}_RS{random_state_value}.npy"  # Nome do arquivo numpy
    
    # Verificar a existência do arquivo e inicializar ou carregar os dados
    tif_dates_ord = check_or_initialize_file(output_file, tiles, var, S2_tile, BDR_DGT, N, random_state_value, bandas_desejadas, FOLDER_OUTPUTS, img_collection, NODATA_VALUE)
    
    # Carregar os dados numpy para o processamento em lotes
    sel_values = np.load(output_file, mmap_mode='r')
    xs = np.load(output_file + '_xs.npy', mmap_mode='r')
    ys = np.load(output_file + '_ys.npy', mmap_mode='r')
    
    # Executar o processamento em lotes com ProcessPoolExecutor
    dfs = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        tqdm_bar = tqdm(total=num_batches)
        
        for batch_index, start_index in enumerate(range(0, num_pixels, batch_size)):
            end_index = min(start_index + batch_size, num_pixels)
            
            # Carregar apenas o bloco atual de pontos
            sel_values_block = sel_values[:, :, start_index:end_index]
            xs_slice = xs[start_index:end_index]
            ys_slice = ys[start_index:end_index]
            
            # Criar argumentos para cada lote de pontos
            arg_list = [(i, sel_values_block, tif_dates_ord, xs_slice, ys_slice, NODATA_VALUE, FOLDER_OUTPUTS, img_collection) for i in range(sel_values_block.shape[2])]
            
            # Mapear os resultados usando executor.map
            for result_df in executor.map(runDetectionForPoint, arg_list):
                dfs.append(result_df)
                tqdm_bar.update(1)
        
        tqdm_bar.close()
    
    # Concatenar os resultados de todos os lotes em um único DataFrame
    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)
        result_df.to_csv(FOLDER_OUTPUTS / 'tabular' / '{}.csv'.format(filename), index=False)
#%%
# def main(batch_size=None):
#     #abre geopackage com pontos
#     # print('A abrir o geopackage com pontos...')
#     raster_path = tiles / 'Theia_T29TNE_20170813-112433.tif'
    
#     print('Processar centros dos pontos de cada geometria para corresponder aos centros dos pixels dos rasters...')
    
#     start_time = time.time()

#     gdf_centros_pixeis = processar_centros_pixeis(BDR_DGT, raster_path)
    
#     # Fim da execução do código
#     end_time = time.time()

#     # Calcula o tempo decorrido em segundos
#     execution_time_seconds = end_time - start_time

#     # Converte o tempo decorrido para minutos
#     execution_time_minutes = execution_time_seconds / 60

#     print("Processar centros dos pixels:", execution_time_minutes, "minutos")
    
#     dados_geoespaciais_metros = readPoints(gdf_centros_pixeis, N, random_state_value)

#     #recolhe nome dos tifs e respetivas datas
#     print('A recolher nome e data dos tifs...')
    
#     if var=='THEIA':
#         tif_names, tif_dates = read_tif_files_theia(S2_tile,tiles)
#     else:
#         tif_names, tif_dates = read_tif_files_gee(S2_tile,tiles)
        
#     #add full path to tif names
#     tif_names = [os.path.join(tiles,i) for i in tif_names]
#     #convert dates to ordinal
#     tif_dates_ord = [d.toordinal() for d in tif_dates]
    
#     print(f'Processando dados {var}... ({tiles})')
#     start_time = time.time()
#     #abre tifs com xarray e armazena informacao
#     print('A abrir tifs com xarray e carregar série temporal...')

#     dates = getTimeSeriesForPoints(tif_names, tif_dates_ord, bandas_desejadas, dados_geoespaciais_metros, output_file)

#     # Fim da execução do código
#     end_time = time.time()

#     # Calcula o tempo decorrido em segundos
#     execution_time_seconds = end_time - start_time

#     # Converte o tempo decorrido para minutos
#     execution_time_minutes = execution_time_seconds / 60

#     print(f"Ler dados {var} para um total de {N} pixels:", execution_time_minutes, "minutos")

    
#     dfs = []
#     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
#         tqdm_bar = tqdm(total=num_batches)
    
#         for batch_index, start_index in enumerate(range(0, num_pixels, batch_size)):
#             end_index = min(start_index + batch_size, num_pixels)
    
#             # Carregar apenas o bloco atual de 10 000 pontos
#             sel_values_block = np.load(output_file, mmap_mode='r')[:, :, start_index:end_index]
#             xs_slice = np.load(output_file + '_xs.npy', mmap_mode='r')[start_index:end_index]
#             ys_slice = np.load(output_file + '_ys.npy', mmap_mode='r')[start_index:end_index]
#             # xs_slice = xs[start_index:end_index]  # Fatiar xs
#             # ys_slice = ys[start_index:end_index]  # Fatiar ys
    
#             # Cria argumentos para cada lote de 10 000 pontos
#             arg_list = [(i, sel_values_block, dates, xs_slice, ys_slice, NODATA_VALUE, FOLDER_OUTPUTS, img_collection) for i in range(sel_values_block.shape[2])]
    
#             for result_df in executor.map(runDetectionForPoint, arg_list):
#                 dfs.append(result_df)
#                 tqdm_bar.update(1)
    
#         tqdm_bar.close()
        
#     # Concatenar os resultados de todos os lotes em um único DataFrame
#     if dfs:
#         result_df = pd.concat(dfs, ignore_index=True)
#         filename = fromParamsReturnName(img_collection, ccd_params, (S2_tile,tiles), N, random_state_value)
#         result_df.to_csv(FOLDER_OUTPUTS / 'tabular' / '{}.csv'.format(filename), index=False)
#%%

#%%
if __name__ == '__main__':
    main(batch_size)
    runValidation(filename, FOLDER_OUTPUTS, BDR_DGT, dt_ini, dt_end, bandFilter, theta)
    