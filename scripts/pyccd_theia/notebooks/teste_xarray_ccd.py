import os
os.chdir("C:\\Users\\scaetano\\Desktop\\CCD")
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
base_path= Path(__name__ ).parent.absolute()  
if module_path not in sys.path:
    sys.path.append(str(module_path))
import ccd
from avaliacao_exatidao_pyccd import filterDate, spatialJoin, preprocessCsvS2, valPol
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import haversine as hs # Novo
from haversine.haversine import Unit
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
def get_most_recent_file(directory, exclude_string=None):
    try:
        # Get a list of all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

        # If there are no files, return None
        if not files:
            return None

        # Filter files based on the exclude_string
        if exclude_string:
            files = [f for f in files if exclude_string not in f]

        # Get the full path for each file and its corresponding modification time
        file_times = [(os.path.join(directory, file), os.path.getmtime(os.path.join(directory, file))) for file in files]

        # Find the file with the maximum modification time
        most_recent_file = max(file_times, key=lambda x: x[1])

        return most_recent_file[0]

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
#%%
def convertPointToCrs(point, source_crs, target_crs):
    """
    Converts a point from a source crs to a target crs.

    Args:
        point: point (shapely.geometry.poin.Point) as extracted from a gdf.
        source_crs: original crs of the input point. Use int (e.g. 4326) or string (e.g. 'EPSG:4326')
        target_crs: new crs the the point should bear. Use int (e.g. 32629) or string (e.g. 'EPSG:32629')
    Returns:
        point with new crs
    """
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    #create a transformer for the conversion
    x, y = point

    # transform coordinates to new crs
    new_x, new_y = transformer.transform(x, y)

    return new_x, new_y
#%%
def read_tif_files(S2_tile,tiles):
    # DGT
    DGT=False
    # outro
    # Theia_T29TNE_20171007-112058

    list_files=[]
    for i in range(2017, 2022):
        if DGT: 
            if i == 2017:
                base_folder = fr"\\192.168.10.35\\Imag_sentinel2\\Theia_S2process\\" + S2_tile
            else:
                base_folder = fr"\\192.168.10.35\\Imag_sentinel2\\Theia_S2process_" + str(i + 1) + "\\" + S2_tile
            tiff_pattern = fr"{base_folder}\\S2*.tif"
        else:
            base_folder=tiles
            #print('base_folder',base_folder)
            tiff_pattern=re.compile('^Theia_T29TNE_' + re.escape(str(i)) + '.*tif$')

        tiff_files1=[]
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                if tiff_pattern.match(file):
                    tiff_files1.append(file)
        
        # Ordena os arquivos pela data
        tiff_files = sorted(tiff_files1)
        list_files.extend(tiff_files)


    if DGT:
        dates = []
        date_pattern = re.compile(r"S2A_L2A_(\d{8})-\d{6}_"+S2_tile+".tif")
        date_pattern2 = re.compile(r"S2B_L2A_(\d{8})-\d{6}_"+S2_tile+".tif")
        for tiff_file in tiff_files:
            match = date_pattern.search(tiff_file)
            match1 = date_pattern2.search(tiff_file)
            if match:
                date = match.group(1)
                dates.append(date)
            if match1:
                date = match1.group(1)
                dates.append(date)
    else:
        L=len('Theia_T29TNE_')
        dates= [x[L:(L+8)] for x in list_files]

    date_objects = [datetime.strptime(date, '%Y%m%d').date() for date in dates]
    return list_files, date_objects
#%%
samples = base_path / 'inputs_pontos'
pontos_input = 'pontos_300_buffers_1_metros.gpkg' 
S2_tile = 'T29TNE'
tiles = base_path / 'pyCCD_Theia' / S2_tile
N=10000

random_state_value = 42

caminho_arquivo = samples / pontos_input
dados_geoespaciais_metros = gpd.read_file(caminho_arquivo) # seria melhor ler csv; apenas coordenadas interessam
dados_geoespaciais_metros = dados_geoespaciais_metros.sample(N, random_state=random_state_value).copy()

tif_names, tif_dates = read_tif_files(S2_tile,tiles)
#add full path to tif names
tif_names = [os.path.join(tiles,i) for i in tif_names]
#convert dates to ordinal
tif_dates_ord = [d.toordinal() for d in tif_dates]
#%%
bandas_desejadas = [1, 2, 3, 7, 9, 10]

time_var = xr.Variable('time',tif_dates_ord)
# Load in and concatenate all individual GeoTIFFs
tifs_xr = [rioxarray.open_rasterio(i, chunks={'x':10924, 'y':10900}) for i in tif_names]
geotiffs_da = xr.concat(tifs_xr, dim=time_var).sel(band=bandas_desejadas)
#%% 
# COORDENADAS X E Y DOS 10 000 PONTOS ESCOLHIDOS
points_x_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.x.values).astype('int'), dims=['location'])
points_y_int = xr.DataArray(np.round(dados_geoespaciais_metros.geometry.y.values).astype('int'), dims=['location'])
#%%
selection = geotiffs_da.sel(x=points_x_int, y=points_y_int, band=bandas_desejadas)
sel_values = selection.values
#%%
# Fim da execução do código
end_time = time.time()

# Calcula o tempo decorrido em segundos
execution_time_seconds = end_time - start_time

# Converte o tempo decorrido para minutos
execution_time_minutes = execution_time_seconds / 60

print("Ler dados Theia:", execution_time_minutes, "minutos")
#%%
alpha = 30
NODATA_VALUE= 65535
df_result=[]
for i in tqdm(range(selection.shape[2])):
    ponto = sel_values[:,:,i]
    dates = selection.time
    
    ponto_desejado=selection.x[i],selection.y[i]
    
    ponto_with_dates = np.column_stack((dates, ponto[:, 0], ponto[:, 1:]))
    
    mask = (ponto_with_dates != NODATA_VALUE).all(axis=1)
    ponto_with_dates_filtered = ponto_with_dates[mask].transpose()
    
    dates, blues, greens, reds, nirs, swir1s, swir2s = ponto_with_dates_filtered
    
    ndvis = np.where((nirs + reds) > 0, 10000 * (nirs - reds) / (nirs + reds), NODATA_VALUE)
    
    ponto_with_dates_filtered[1]=ndvis
    
    ponto_with_dates_filtered1=ponto_with_dates_filtered.transpose()
    
    ponto_with_dates_filtered2 = ponto_with_dates_filtered1[~np.any(ponto_with_dates_filtered1 == NODATA_VALUE, axis=1)]
    
    ponto_with_dates_filtered3=ponto_with_dates_filtered2.transpose()
    
    dates, ndvis, greens, reds, nirs, swir1s, swir2s = ponto_with_dates_filtered3
    
    results = ccd.detect(dates, ndvis, greens, reds, nirs, swir1s, swir2s)
    
    
    predicted_values = []
    prediction_dates = []
    break_dates = []
    start_dates = []
    end_dates=[]
    # coeficientes=[]
    prob=[]
    
    for num, result in enumerate(results['change_models']):
        days = np.arange(result['start_day'], result['end_day'] + 1)
        prediction_dates.append(days)
        break_dates.append(result['break_day'])
        start_dates.append(result['start_day'])
        end_dates.append(result['end_day'])
        prob.append(result['change_probability'])
        
        intercept = result['blue']['intercept']
        coef = result['blue']['coefficients']
        
        predicted_values.append(intercept + coef[0] * days +
                                coef[1]*np.cos(days*1*2*np.pi/365.25) + coef[2]*np.sin(days*1*2*np.pi/365.25) +
                                coef[3]*np.cos(days*2*2*np.pi/365.25) + coef[4]*np.sin(days*2*2*np.pi/365.25) +
                                coef[5]*np.cos(days*3*2*np.pi/365.25) + coef[6]*np.sin(days*3*2*np.pi/365.25))
    
    ndvi_magnitudes = [predicted_values[num][-1] - predicted_values[num + 1][0] for num in range(len(predicted_values) - 1)]
    
    # Se não houver mais segmentos a seguir adiciona NODATA_VALUE se só existir um segmento adiciona 0
    ndvi_magnitudes.append(65535 if ndvi_magnitudes and any(ndvi_magnitudes) else 0)
    
    datas = [datetime.fromordinal(data) for data in break_dates]
    break_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in start_dates]
    start_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in end_dates]
    end_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    ponto_desejado_wgs = convertPointToCrs(ponto_desejado, 32629, 4326)
    
    ponto_desejado_wgs_x, ponto_desejado_wgs_y = ponto_desejado_wgs
    
    dados = [
        {'tBreak': break_dates_epoch,'tEnd': end_dates_epoch,'tStart':start_dates_epoch,'changeProb':prob, 'Lat': ponto_desejado_wgs_y,'Lon': ponto_desejado_wgs_x, 'ndvi_magnitude' : ndvi_magnitudes}
    ]
    
    df = pd.DataFrame(dados)
    
    # Reorganizar colunas
    ordem_colunas = ['tBreak', 'tEnd', 'tStart', 'changeProb', 'Lat', 'Lon', 'ndvi_magnitude']
    df=df[ordem_colunas]
    df_result.append(df)

df_final = pd.concat(df_result, ignore_index=True)
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
df_final.to_csv(f'C:\\Users\\scaetano\\Desktop\\CCD\\outputs\\csv\\Theia\\csv_{timestamp}_{random_state_value}.csv', index=False)

# datas do filtro das datas da análise (DGT 300)
########### Não alterar ################
dt_ini = '2018-09-12' # data inicial
dt_end = '2021-09-30' # data final
# Margem de tolerância entre a quebra do Modelo e do Analista
theta = 60 # +/- theta dias de diferenca
# bandar a filtrar com base na magnitude
bandFilter = None #não implementado ainda - não mexer
csv_file_ccd=get_most_recent_file(str('C:\\Users\\scaetano\\Desktop\\CCD\\outputs\\csv\\Theia\\'),exclude_string='pre_proc')
if csv_file_ccd is None: 
    raise ValueError('Pasta vazia')
print('csv_file_ccd',csv_file_ccd)
# Validação com BDR-300
#caminho para gravar o csv pre-processado
filename=Path(csv_file_ccd).with_suffix('')
csv_preprocessed_path = str(filename)+'_pre_proc.csv'
print('csv_preprocessed_path',csv_preprocessed_path)
#caminho para a base de dados de validação
path_adjusted_bdr = base_path / 'BDR_300_artigo' / 'BDR_CCDC_TNE_Adjusted.shp'

csv_s2 = pd.read_csv(csv_file_ccd)
#correr pre-processamento
csv_s2 = preprocessCsvS2(csv_s2)
csv_s2.to_csv(csv_preprocessed_path)

"""## Filtrar datas
Limitar análise ao período considerado pelos analistas DGT
"""

#correr filtro de datas
ccdcFiltro = filterDate(csv_preprocessed_path,dt_ini, dt_end, bandFilter)

"""## Spatial join
Faz join dos pontos do csv com a informação de referencia da DGT (300 buffers). É associada aos pontos a informação da validação - data de alteração, tipo, classes, etc.
"""

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

DF_FINAL_T.to_csv(base_path / 'outputs' / f'Theia_FP_FN_{random_state_value}_alpha_{alpha}.csv', index=False)