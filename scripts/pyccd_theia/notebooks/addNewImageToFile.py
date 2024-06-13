import numpy as np
import xarray as xr
import rioxarray
import os
user_profile = os.environ['USERPROFILE']
import logging

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
from notebooks.avaliacao_exatidao_pyccd import filterDate, spatialJoin, preprocessCsvS2, valPol
from notebooks.read_files import read_tif_files_theia, read_tif_files_gee, get_most_recent_file, readPoints
from notebooks.processing import getTimeSeriesForPoints, runDetectionForPoint, processar_centros_pixeis
from notebooks.utils import fromParamsReturnName
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import math
#%%
def addNewImageToFile(tif_names, tif_dates_ord, bandas_desejadas, dados_geoespaciais_metros, output_file):
    # Carregar e concatenar o último GeoTIFF
    geotiff_da = rioxarray.open_rasterio(tif_names, chunks={'x': 10924, 'y': 10900}).sel(band=bandas_desejadas)
    geotiff_da = geotiff_da.expand_dims('time').assign_coords(time=[tif_dates_ord])

    # Coordenadas X e Y dos 10.000 pontos escolhidos
    points_x_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.x.values).astype('int'), dims=['location'])
    points_y_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.y.values).astype('int'), dims=['location'])

    selection = geotiff_da.sel(x=points_x_int, y=points_y_int, band=bandas_desejadas)
    sel_values = selection.values

    # Carregar o arquivo existente
    try:
        existing_data = np.load(output_file)
        print("Arquivo existente carregado com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{output_file}.npy' não foi encontrado.")

    # Adicionar os novos dados ao array existente
    updated_data = np.concatenate((existing_data, sel_values), axis=0)
    
    # Salvar o array atualizado de volta ao arquivo
    np.save(output_file, updated_data)
    print("Nova imagem adicionada e dados salvos com sucesso.")

    return updated_data
#%%
public_documents = Path('C:/Users/Public/Documents/')

samples = public_documents / 'inputs_pontos'
pontos_input = 'pontos_300_buffers_1_metros.gpkg'
caminho_arquivo = samples / pontos_input

FOLDER_THEIA = public_documents / 'imagens_Theia' # Caminho dados THEIA
FOLDER_GEE = public_documents / 'imagens_GEE' # Caminho dados GEE

FOLDER_BDR = public_documents / 'BDR_300_artigo' / 'BDR_CCDC_TNE_Adjusted.shp' # Caminho para a base de dados de validação

FOLDER_OUTPUTS = public_documents / 'output_BDR300'
S2_tile = 'T29TNE'
var = 'THEIA' # choose variable: THEIA or GEE

if var == 'THEIA':
    tiles = FOLDER_THEIA / S2_tile
else:
    tiles = FOLDER_GEE / S2_tile

img_collection = tiles.parts[-2]

# N=241941 # numero total de pontos presentes na BDR
N = 100000

random_state_value = 42

bandas_desejadas = [1, 2, 3, 7, 9, 10]

NODATA_VALUE= 65535

raster_path = tiles / 'Theia_T29TNE_20170813-112433.tif'
    
print('Processar centros dos pontos de cada geometria para corresponder aos centros dos pixels dos rasters...')

start_time = time.time()

gdf_centros_pixeis = processar_centros_pixeis(FOLDER_BDR, raster_path)

dados_geoespaciais_metros = readPoints(gdf_centros_pixeis, N, random_state_value)
#%%
#recolhe nome dos tifs e respetivas datas
print('A recolher nome e data dos tifs...')

if var=='THEIA':
    tif_names, tif_dates = read_tif_files_theia(S2_tile,tiles)
else:
    tif_names, tif_dates = read_tif_files_gee(S2_tile,tiles)
    
#add full path to tif names, read only the last file
tif_names = [os.path.join(tiles,i) for i in tif_names][-1]
#convert dates to ordinal, read only the last file
tif_dates_ord = [d.toordinal() for d in tif_dates][-1]

print(f'Processando dados {var}... ({tiles})')
start_time = time.time()
#abre tifs com xarray e armazena informacao
print('A abrir tifs com xarray e carregar série temporal...')

output_file = f"sel_values_{N}.npy"
numpy_file = addNewImageToFile(tif_names, tif_dates_ord, bandas_desejadas, dados_geoespaciais_metros, output_file)

