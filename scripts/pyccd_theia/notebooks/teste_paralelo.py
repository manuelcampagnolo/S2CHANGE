import os
os.chdir("C:\\Users\\Utilizador\\Desktop\\CCD")
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
NODATA_VALUE= 65535
THEIA=True # if False, use S2 and S2 cloudness
CRIAR_CSV= True
VALIDAR_CSV=True
CRIAR_PLOTS=False
CRIAR_NDVI=True

def main():
    N=10000 #len(dados_geoespaciais)
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
    csv_folder_ccd= base_path / 'outputs' / 'csv' / 'Theia'

    if CRIAR_CSV: 
        #caminho_arquivo = os.path.join(module_path, tiles, 'BUFFER_300','pontos_300_buffers_1_metros.gpkg') #"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\pontos_300_buffers 1_metros.gpkg"
        caminho_arquivo = samples / pontos_input #"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\pontos_300_buffers 1_metros.gpkg"
        dados_geoespaciais_metros = gpd.read_file(caminho_arquivo) # seria melhor ler csv; apenas coordenadas interessam
        dados_geoespaciais_metros = dados_geoespaciais_metros.sample(N, random_state=42).copy() #para quando se quer uma amostra dos pontos

        dfs = []
        # escrever csv para cada múltiplo de:
        points_per_csv = 10000
        csv_counter = 1

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

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
        csv_file_ccd= base_path / 'outputs' / 'csv' / 'Theia' / f"csv_{csv_counter}_{timestamp}.csv"
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

        return DF_FINAL_T, N

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
    # print('module path', module_path)
    # print('S2_tile',S2_tile)
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
                ndvis.append(NODATA_VALUE)
    
        df3.iloc[1]=ndvis
        
        df3=df3.transpose()
        df3 = df3[~(df3 == NODATA_VALUE).any(axis=1)].reset_index(drop=True)
        df3=df3.transpose()
        dates, ndvis, greens, reds, nirs, swir1s, swir2s = df3.values
        
        results = ccd.detect(dates, ndvis, greens, reds, nirs, swir1s, swir2s)
    else:
        results = ccd.detect(dates, blues, greens, reds, nirs, swir1s, swir2s)

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
    
    ndvi_magnitudes=[]
    for num in range(len(predicted_values) - 1):
        diff = predicted_values[num][-1] - predicted_values[num + 1][0]
        ndvi_magnitudes.append(diff)
    
    # Verificar se há valores em diff
    if len(ndvi_magnitudes) > 0 and any(ndvi_magnitudes):
        # Se houver valores, adicionar np.nan como último elemento
        ndvi_magnitudes.append(65535)
    else:
        # Se não houver valores, adicionar zero como último elemento
        ndvi_magnitudes.append(0)
    
    # # Preencher com zeros para garantir que ndvi_magnitudes tenha o mesmo número de elementos que predicted_values
    # while len(ndvi_magnitudes) < len(predicted_values):
    #     ndvi_magnitudes.append(0)


    # variavel_grafico = ndvis
    
    # mask = np.array(results['processing_mask'], dtype='bool')
    # date_objects1 = [datetime.fromordinal(int(ordinal)) for ordinal in dates]
    
    # plt.style.use('ggplot')
    # fg = plt.figure(figsize=(14, 4), dpi=90)
    
    # a1 = fg.add_subplot(1, 1, 1, xlim=(min(date_objects1), max(date_objects1)))#, ylim=(0, 1500))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    
    # a1.xaxis.set_major_locator(mdates.YearLocator(1))
    # a1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    
    # colors = ['orange', 'purple', 'brown']
    
    # # Predicted curves
    # for idx, (_preddate, _predvalue) in enumerate(zip(prediction_dates, predicted_values)):
    #     # Converter números ordinais de volta para objetos de data
    #     _preddate = [datetime.fromordinal(int(ordinal)) for ordinal in _preddate]
    #     color = colors[idx % len(colors)]
    #     a1.plot(_preddate, _predvalue, color, linewidth=1, label=f'Predicted values {idx + 1}')
    
    # a1.plot(np.array(date_objects1)[mask], np.array(variavel_grafico)[mask], 'g+',label='Observed values')  # Observed values
    # a1.plot(np.array(date_objects1)[~mask], np.array(variavel_grafico)[~mask], 'g+')  # Observed values masked out

    # ticks = [min(date_objects1) + timedelta(days=i*365) for i in range(10) if min(date_objects1) + timedelta(days=i*365) <= datetime(2021, 12, 31)]
    # plt.xticks(ticks)
    
    # a1.plot([], [], color='r', linestyle='--', label='Start dates')
    # a1.plot([], [], color='brown', linestyle='--', label='End Dates')
    # a1.plot([], [], color='b', linestyle='--', label='Break dates')
    # a1.plot([], [], color='black', linestyle='--', label='DGT Dates')
    
    # for b in break_dates:
    #     b_date = datetime.fromordinal(b)
    #     a1.axvline(b_date, color='b', linestyle='--')
    #     a1.text(mdates.date2num(b_date)+1, a1.get_ylim()[1], b_date.strftime('%d-%m-%Y'), rotation=90, ha='right',weight='bold', va='top', color='b',size=8)
    
    # # Linhas verticais para datas de início (color='r')
    # for s in start_dates:
    #     s_date = datetime.fromordinal(s)
    #     a1.axvline(s_date, color='r', linestyle='--')
    #     a1.text(mdates.date2num(s_date) + 1, a1.get_ylim()[0], s_date.strftime('%d-%m-%Y'), rotation=90, ha='right',weight='bold', va='bottom', color='r',size=8)
    
    # for e in end_dates:
    #     e_date = datetime.fromordinal(e)
    #     a1.axvline(e_date, color='brown', linestyle='--')
    #     a1.text(mdates.date2num(e_date) + 1, a1.get_ylim()[0], e_date.strftime('%d-%m-%Y'), rotation=90, ha='right',weight='bold', va='bottom', color='brown',size=8,alpha=0.6)

    # reference_start_date = datetime.strptime('2018-09-12', '%Y-%m-%d')
    # reference_end_date = datetime.strptime('2021-09-30', '%Y-%m-%d')
    # a1.axvspan(reference_start_date, reference_end_date, facecolor='pink', alpha=0.3,label='Período de Referência')
    # # plt.text(0.5, 0.9, 'Período de Referência', transform=plt.gca().transAxes, color='blue', size=10, ha='center', bbox=dict(facecolor='yellow', alpha=0.3))
    
    # plt.ylabel('NDVI')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.7, -0.1), fancybox=True, shadow=True, ncol=3)
    # plt.tight_layout()
    # plt.savefig('C:\\Users\\Utilizador\\Desktop\\CCD\outputs\\teste.png')
    # plt.close()

    
    datas = [datetime.fromordinal(data) for data in break_dates]
    break_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in start_dates]
    start_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in end_dates]
    end_dates_epoch = [int(data.replace(tzinfo=timezone.utc).timestamp() * 1000) for data in datas]

    ponto_desejado_wgs = convertPointToCrs(ponto_desejado, 32629, 4326)

    dados = [
        {'tBreak': break_dates_epoch,'tEnd': end_dates_epoch,'tStart':start_dates_epoch,'changeProb':prob, 'Lat': ponto_desejado_wgs.y,'Lon': ponto_desejado_wgs.x, 'ndvi_magnitude' : ndvi_magnitudes}
    ]
    
    df = pd.DataFrame(dados)
    
    # Reorganizar colunas
    ordem_colunas = ['tBreak', 'tEnd', 'tStart', 'changeProb', 'Lat', 'Lon', 'ndvi_magnitude']
    df = df[ordem_colunas]
    print(df)
    
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
#%%
if __name__ == "__main__":
    DF,N=main()
    DF.to_csv(base_path / 'outputs' / f'Theia_FP_VN_{N}_sem_zeros.csv', index=False)