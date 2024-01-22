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
from datetime import datetime
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

import warnings
warnings.filterwarnings('ignore')

#%%
def extract_date(file_path):
    match = re.search(r'\d{8}', file_path)
    if match:
        return datetime.strptime(match.group(), '%Y%m%d')
    return None
#%%
#test
def main():
    THEIA=True # if False, use S2 and S2 cloudness
    NODATA_VALUE= 65535
    CRIAR_CSV= False
    VALIDAR_CSV=True
    CRIAR_PLOTS=False
    CRIAR_NDVI=True
    pontos_input = 'pontos_300_buffers_1_metros.gpkg' 
    S2_tile = 'T29TNE'
    if THEIA:
        tiles = base_path / 'pyCCD_Theia' / S2_tile
    else: 
        pass # necessário pré-procesar 
    samples = base_path / 'inputs_pontos'
    
    # datas do filtro das datas da análise (DGT 300)
    ########### Não alterar ################
    dt_ini = '2018-09-12' # data inicial
    dt_end = '2021-09-30' # data final
    # Margem de tolerância entre a quebra do Modelo e do Analista
    theta = 60 # +/- theta dias de diferenca
    # bandar a filtrar com base na magnitude
    bandFilter = None #não implementado ainda - não mexer
    
    # nome output
    csv_folder_ccd= base_path / 'outputs' / 'csv'

    if CRIAR_CSV: 
        #caminho_arquivo = os.path.join(module_path, tiles, 'BUFFER_300','pontos_300_buffers_1_metros.gpkg') #"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\pontos_300_buffers 1_metros.gpkg"
        caminho_arquivo = samples / pontos_input #"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\pontos_300_buffers 1_metros.gpkg"
        dados_geoespaciais_metros = gpd.read_file(caminho_arquivo) # seria melhor ler csv; apenas coordenadas interessam
        dados_geoespaciais_metros = dados_geoespaciais_metros.sample(1000, random_state=42).copy() #para quando se quer uma amostra dos pontos

        dfs = []
        points_per_csv = 1000
        csv_counter = 1

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            N=10 #len(dados_geoespaciais)

            tqdm_bar = tqdm(total=N)
            
            args_list = [(k, dados_geoespaciais_metros.iloc[k].geometry, S2_tile, tiles) for k in range(N)] #len(dados_geoespaciais))]
            
            for result_df in executor.map(processar_ponto, args_list):
                dfs.append(result_df)
                tqdm_bar.update(1)

                # Check if the length of dfs is a multiple of points_per_csv
                if False:#len(dfs) % points_per_csv == 0:
                    # Concatenate DataFrames and save to CSV
                    partial_df = pd.concat(dfs, ignore_index=True)
                    #partial_df.to_csv(f"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\CSV 300\\csv_{csv_counter}_{timestamp}.csv", index=False)
                    # tqdm_bar.set_postfix({"CSV Files": csv_counter})
                    csv_counter += 1
                    dfs = []  # Reset the list for the next batch

            tqdm_bar.close()

        # Save the remaining points
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        csv_file_ccd= base_path / 'outputs' / 'csv' / f"csv_{csv_counter}_{timestamp}.csv"
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.to_csv(csv_file_ccd, index=False)
            #final_df.to_csv(f"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\CSV 300\\csv_{csv_counter}_{timestamp}.csv", index=False)
    
    if VALIDAR_CSV:
        #caminho para o csv (leitura)
        csv_file_ccd=get_most_recent_file(str(csv_folder_ccd),exclude_string='pre_proc')
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
        print('F1-score = {}%'.format(round(100*f1,2)))
        print('Omission error = {}%'.format(round(100*om,2)))
        print('Commission error = {}%'.format(round(100*cm,2)))

def read_tif_files(S2_tile,tiles):
        # DGT
    DGT=False
    # outro
    # Theia_T29TNE_20171007-112058

    list_files=[]
    for i in range(2017, 2024):
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
        tiff_files = sorted(tiff_files1) #, key=extract_date)
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

def processar_ponto(args):
    k, ponto_desejado, S2_tile, tiles = args
    #print('module path', module_path)
    #print('S2_tile',S2_tile)
    base_folder=module_path / S2_tile
    bandas_desejadas = [1, 2, 3, 7, 9, 10]

    tiff_files,date_objects = read_tif_files(S2_tile, tiles)

    #print(tiff_files)

    query_bands = []
    for j, tiff_path in enumerate(tiff_files):
        numero_ordinal = date_objects[j].toordinal()

        #with rasterio.open(os.path.join(module_path,S2_tile,tiff_path)) as src:
        with rasterio.open(str(tiles / tiff_path)) as src:
            valores_ponto_desejado = [banda for banda in src.sample([(ponto_desejado.x, ponto_desejado.y)], indexes=bandas_desejadas)]

            linha_pixeis1 = np.concatenate(valores_ponto_desejado).tolist()
            
            linha_pixeis = [numero_ordinal] + linha_pixeis1

            query_bands.append(linha_pixeis)
    
    df1 = pd.DataFrame(query_bands)
    df2 = df1[~(df1 == NODATA_VALUE).any(axis=1)].reset_index(drop=True)
    df3 = df2.transpose()

    dates, blues, greens, reds, nirs, swir1s, swir2s = df3.values
    if CRIAR_NDVI:
        ndvis=[]
        for (nir,red) in zip(nirs,reds):
            if nir+red>0:
                ndvis.append(10000*(nir-red)/(nir+red))
            else:
                ndvis.append(0)
        #results = ccd.detect(dates, blues, greens, reds, nirs, swir1s, swir2s)
        results = ccd.detect(dates, ndvis, greens, reds, nirs, swir1s, swir2s)
    else:
        results = ccd.detect(dates, blues, greens, reds, nirs, swir1s, swir2s)
    
    # mask = np.array(results['processing_mask'], dtype='bool')

    # print('Start Date: {0}\nEnd Date: {1}\n'.format(datetime.fromordinal(int(dates[0])),
    #                                                 datetime.fromordinal(int(dates[-1]))))
    
    # predicted_values = []
    prediction_dates = []
    break_dates = []
    start_dates = []
    end_dates=[]
    # coeficientes=[]
    prob=[]
    for num, result in enumerate(results['change_models']):
        # print('Result: {}'.format(num))
        # print('Start Date: {}'.format(datetime.fromordinal(result['start_day'])))
        # print('End Date: {}'.format(datetime.fromordinal(result['end_day'])))
        # print('Break Date: {}'.format(datetime.fromordinal(result['break_day'])))
        # print('Norm: {}\n'.format(np.linalg.norm([result['green']['magnitude'],
        #                                         result['red']['magnitude'],
        #                                         result['nir']['magnitude'],
        #                                         result['swir1']['magnitude'],
        #                                         result['swir2']['magnitude']])))
        # print('Change prob: {}'.format(result['change_probability']))
        
        days = np.arange(result['start_day'], result['end_day'] + 1)
        prediction_dates.append(days)
        break_dates.append(result['break_day'])
        start_dates.append(result['start_day'])
        end_dates.append(result['end_day'])
        prob.append(result['change_probability'])
        
        # intercept = result['nir']['intercept']
        # coef = result['nir']['coefficients']
        # coeficientes.append(coef)
        
        # predicted_values.append(intercept + coef[0] * days +
        #                         coef[1]*np.cos(days*1*2*np.pi/365.25) + coef[2]*np.sin(days*1*2*np.pi/365.25) +
        #                         coef[3]*np.cos(days*2*2*np.pi/365.25) + coef[4]*np.sin(days*2*2*np.pi/365.25) +
        #                         coef[5]*np.cos(days*3*2*np.pi/365.25) + coef[6]*np.sin(days*3*2*np.pi/365.25))
     
    datas = [datetime.fromordinal(data) for data in break_dates]
    break_dates_epoch = [int(data.timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in start_dates]
    start_dates_epoch = [int(data.timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in end_dates]
    end_dates_epoch = [int(data.timestamp() * 1000) for data in datas]

    ponto_desejado_wgs = convertPointToCrs(ponto_desejado, 32629, 4326)

    dados = [
        {'tBreak': break_dates_epoch,'tEnd': end_dates_epoch,'tStart':start_dates_epoch,'changeProb':prob, 'Lat': ponto_desejado_wgs.y,'Lon': ponto_desejado_wgs.x}
    ]
    
    df = pd.DataFrame(dados)
    
    # Reorganizar colunas
    ordem_colunas = ['tBreak', 'tEnd', 'tStart', 'changeProb', 'Lat', 'Lon']
    df = df[ordem_colunas]
    
    return df
    
    # # Nome do arquivo CSV
    # nome_arquivo_csv = f'C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\CSV 300\\csv_ponto_{k}_v2.csv'
    
    # try:
    #     df_existente = pd.read_csv(nome_arquivo_csv)
    #     df = pd.concat([df_existente, df], ignore_index=True)
    # except FileNotFoundError:
    #     pass
    
    # # Escrever DataFrame para o arquivo CSV
    # df.to_csv(nome_arquivo_csv, index=False)

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
    #create a transformer for the conversion
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    #extract coordinates from the input point
    x, y = point.xy

    #transform coordinates to new crs
    new_x, new_y = transformer.transform(x[0], y[0])

    return Point(new_x, new_y)

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
    
if __name__ == "__main__":
    main()
#%%
# def main():
#     caminho_arquivo = "C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\pontos_300_buffers 1_metros.gpkg"
#     dados_geoespaciais = gpd.read_file(caminho_arquivo)

#     tiles = "T29TNE"
    
#     dfs = []

#     # with ThreadPoolExecutor() as executor:
#     #     tqdm_bar = tqdm(total=len(dados_geoespaciais))
        
#     #     # Crie argumentos para cada ponto
#     #     args_list = [(k, dados_geoespaciais.iloc[k].geometry, tiles) for k in range(len(dados_geoespaciais))]
        
#     #     # Execute as tarefas em paralelo
#     #     for _ in executor.map(processar_ponto, args_list):
#     #         tqdm_bar.update(1)
        
#     #     tqdm_bar.close()
#     num_threads = 12
#     with ProcessPoolExecutor(max_workers=num_threads) as executor:
#         tqdm_bar = tqdm(total=5)#len(dados_geoespaciais))
        
#         # Crie argumentos para cada ponto
#         args_list = [(k, dados_geoespaciais.iloc[k].geometry, tiles) for k in range(5)]#(len(dados_geoespaciais))]
        
#         # Execute tasks in parallel
#         for result_df in executor.map(processar_ponto, args_list):
#             dfs.append(result_df)  # Collect DataFrames
#             tqdm_bar.update(1)
        
#         tqdm_bar.close()

#     # Concatenate all DataFrames
#     final_df = pd.concat(dfs, ignore_index=True)
    
#     final_df.to_csv("C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\CSV 300\\csv_final.csv",index=False)
#     # You can use 'final_df' for further processing or analysis
#     return final_df

# if __name__ == "__main__":
#     final_df = main()
#     print(final_df)