import os
#os.chdir("C:\\Users\\scaetano\\Desktop\\CCD")
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
import pandas as pd
import numpy as np
import rasterio
import glob
import re
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from pathlib import Path
# chamar python a partir da pasta 'CCD'
module_path= Path(__name__ ).parent.absolute() / 'S2CHANGE' / 'scripts' / 'pyccd_theia' #  / 'CCD' / 'S2CHANGE' / 'scripts' / 
base_path= Path(__name__ ).parent.absolute()  # dir do script; # dir referência (acima): 'DGT-S2CHANGE_2023'
if module_path not in sys.path:
    sys.path.append(str(module_path))
import ccd
from notebooks.avaliacao_exatidao_pyccd import filterDate, spatialJoin, preprocessCsvS2, valPol
from notebooks.read_files import read_tif_files, get_most_recent_file, readPoints
import utils
import processing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
#import haversine as hs # not used apparentely
#from haversine.haversine import Unit # not used apparentely
from datetime import timezone

import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import rioxarray
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
#%%
# Início da medição do tempo
start_time = time.time()
#%%
samples = base_path / 'inputs_pontos'
pontos_input = 'pontos_300_buffers_1_metros.gpkg' 
S2_tile = 'T29TNE'
img_collection = 'pyCCD_Theia'
tiles = base_path / img_collection / S2_tile
N=10000

random_state_value = 42

caminho_arquivo = samples / pontos_input

bandas_desejadas = [1, 2, 3, 7, 9, 10]

alpha = ccd.parameters.defaults['ALPHA'] #looks for alpha in the parameters.py file
ccd_params = ccd.parameters.defaults

NODATA_VALUE= 65535

#parametros da validacao
# datas do filtro das datas da análise (DGT 300)
########### Não alterar ################
dt_ini = '2018-09-12' # data inicial
dt_end = '2021-09-30' # data final
# Margem de tolerância entre a quebra do Modelo e do Analista
theta = 60 # +/- theta dias de diferenca
# bandar a filtrar com base na magnitude
bandFilter = None #não implementado ainda - não mexer
path_adjusted_bdr = base_path / 'BDR_300_artigo' / 'BDR_CCDC_TNE_Adjusted.shp'

#%%
def main():
    #abre geopackage com pontos
    print('A abrir o geopackage com pontos...')
    dados_geoespaciais_metros = readPoints(caminho_arquivo, N, random_state_value)

    #recolhe nome dos tifs e respetivas datas
    print('A recolher nome e data dos tifs...')
    tif_names, tif_dates = read_tif_files(S2_tile,tiles)
    #add full path to tif names
    tif_names = [os.path.join(tiles,i) for i in tif_names]
    #convert dates to ordinal
    tif_dates_ord = [d.toordinal() for d in tif_dates]

    #abre tifs com xarray e armazena informacao
    print('A abrir tifs com xarray e carregar série temporal...')
    sel_values, dates, xs, ys = processing.getTimeSeriesForPoints(tif_names, tif_dates_ord, bandas_desejadas, dados_geoespaciais_metros)

    # Fim da execução do código
    end_time = time.time()

    # Calcula o tempo decorrido em segundos
    execution_time_seconds = end_time - start_time

    # Converte o tempo decorrido para minutos
    execution_time_minutes = execution_time_seconds / 60

    print("Ler dados Theia:", execution_time_minutes, "minutos")

    #executa o ccd em paralelo por ponto
    print('A executar o ccd nos pontos...')
    dfs = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        tqdm_bar = tqdm(total=sel_values.shape[2])

        arg_list = [(i, sel_values, dates, xs, ys, NODATA_VALUE) for i in range(sel_values.shape[2])]

        for result_df in executor.map(processing.runDetectionForPoint, arg_list):
            dfs.append(result_df)
            tqdm_bar.update(1)
        tqdm_bar.close()
    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = utils.fromParamsReturnName(img_collection, ccd_params, (S2_tile,tiles), N, random_state_value)
        df_final.to_csv('{}.csv'.format(filename), index=False)#df_final.to_csv(f'C:\\Users\\scaetano\\Desktop\\CCD\\outputs\\csv\\Theia\\csv_{timestamp}_{random_state_value}.csv', index=False)
    

def runValidation():
    print('A correr validação dos resultados do ccd...')
    filename = utils.fromParamsReturnName(img_collection, ccd_params, (S2_tile,tiles), N, random_state_value)
    csv_s2 = pd.read_csv('{}.csv'.format(filename))
    #correr pre-processamento
    csv_s2 = preprocessCsvS2(csv_s2)
    csv_preprocessed_path = '{}_pre_proc.csv'.format(filename)
    csv_s2.to_csv(csv_preprocessed_path)

    """## Filtrar datas
    Limitar análise ao período considerado pelos analistas DGT
    """

    #correr filtro de datas
    ccdcFiltro = filterDate(csv_preprocessed_path, dt_ini, dt_end, bandFilter)

    """## Spatial join
    Faz join dos pontos do csv com a informação de referencia da DGT (300 buffers). É associada aos pontos a informação da validação - data de alteração, tipo, classes, etc.
    """
    gdfVal = gpd.read_file(path_adjusted_bdr)
    gdfVal.to_crs(crs = 'EPSG:3763', inplace = True)
    #executa o join
    ccdcVal, ccdcVal_T = spatialJoin(path_adjusted_bdr, ccdcFiltro)

    """## Validação
    Faz a validação da deteção - compara resultado do modelo (ccd) com dados de referência DGT
    """
    #faz a validação da deteção
    DF_FINAL, DF_FINAL_T = valPol(ccdcVal_T, theta) #funcoes.valPol

    """**Resultados da validação**"""

    #delimita análise apenas para pontos referentes a transições entre Pinheiro Bravo e Eucalipto para Superfície sem vegetação, herbáceas e matos
    #elimina também pontos da bordadura
    df_aux = DF_FINAL_T.copy()
    df_aux = df_aux.loc[(df_aux.altera=="Sem Alteracao")|((df_aux.altera=="Com Alteracao")&(df_aux.classeAnterior.isin(['Pinheiro bravo','Eucalipto']))&(df_aux.classeAtual.isin(['Superficie sem vegetacao escura','Superficie sem vegetacao clara','Vegetacao herbacea espontanea','Matos'])))]
    df_aux = df_aux.loc[df_aux.bordadura==0]

    #imprime f1-score, erro e omissão e erro de comissão
    cm = df_aux.FP.sum()/(df_aux.FP.sum()+df_aux.VP.sum())
    om = df_aux.FN.sum()/(df_aux.FN.sum()+df_aux.VP.sum())
    f1 = 2*(1-om)*(1-cm)/(2-om-cm)

    print(f'Alpha: {alpha}')
    print('F1-score = {}%'.format(round(100*f1,2)))
    print('Omission error = {}%'.format(round(100*om,2)))
    print('Commission error = {}%'.format(round(100*cm,2)))

    DF_FINAL_T.to_csv(base_path / 'outputs' / f'VAL_{filename}.csv', index=False)

if __name__ == '__main__':
    main()
    runValidation()
