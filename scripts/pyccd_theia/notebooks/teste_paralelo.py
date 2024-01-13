import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import glob
import re
import os
import sys
from datetime import datetime
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
import ccd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
#%%
def extract_date(file_path):
    match = re.search(r'\d{8}', file_path)
    if match:
        return datetime.strptime(match.group(), '%Y%m%d')
    return None
#%%

def main():
    tiles = "T29TNE"
    caminho_arquivo = os.path.join(module_path, tiles, 'BUFFER_300','pontos_300_buffers_1_metros.gpkg') #"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\pontos_300_buffers 1_metros.gpkg"
    dados_geoespaciais = gpd.read_file(caminho_arquivo) # seria melhor ler csv; apenas coordenadas interessam
    
    dfs = []
    points_per_csv = 20000
    csv_counter = 1
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        N=10 #len(dados_geoespaciais)

        tqdm_bar = tqdm(total=N)
        
        args_list = [(k, dados_geoespaciais.iloc[k].geometry, tiles) for k in range(N)] #len(dados_geoespaciais))]
        
        for result_df in executor.map(processar_ponto, args_list):
            dfs.append(result_df)
            tqdm_bar.update(1)

            # Check if the length of dfs is a multiple of points_per_csv
            if len(dfs) % points_per_csv == 0:
                # Concatenate DataFrames and save to CSV
                partial_df = pd.concat(dfs, ignore_index=True)
                #partial_df.to_csv(f"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\CSV 300\\csv_{csv_counter}_{timestamp}.csv", index=False)
                # tqdm_bar.set_postfix({"CSV Files": csv_counter})
                csv_counter += 1
                dfs = []  # Reset the list for the next batch

        tqdm_bar.close()

    # Save the remaining points
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        #final_df.to_csv(f"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\CSV 300\\csv_{csv_counter}_{timestamp}.csv", index=False)

    return final_df

def read_tif_files(S2_tile):
        # DGT
    DGT=False
    # outro
    # Theia_T29TNE_20171007-112058

    list_files=[]
    for i in range(2017, 2023):
        if DGT: 
            if i == 2017:
                base_folder = fr"\\192.168.10.35\\Imag_sentinel2\\Theia_S2process\\" + S2_tile
            else:
                base_folder = fr"\\192.168.10.35\\Imag_sentinel2\\Theia_S2process_" + str(i + 1) + "\\" + S2_tile
            tiff_pattern = fr"{base_folder}\\S2*.tif"
        else:
            base_folder=os.path.join(module_path,S2_tile)
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
    k, ponto_desejado, S2_tile = args
    print('module path', module_path)
    print('S2_tile',S2_tile)
    base_folder=os.path.join(module_path,S2_tile)
    bandas_desejadas = [1, 2, 3, 7, 9, 10]

    tiff_files,date_objects = read_tif_files(S2_tile)

    #print(tiff_files)

    query_bands = []
    for j, tiff_path in enumerate(tiff_files):
        numero_ordinal = date_objects[j].toordinal()

        with rasterio.open(os.path.join(module_path,S2_tile,tiff_path)) as src:
            valores_ponto_desejado = [banda for banda in src.sample([(ponto_desejado.x, ponto_desejado.y)], indexes=bandas_desejadas)]

            linha_pixeis1 = np.concatenate(valores_ponto_desejado).tolist()
            
            linha_pixeis = [numero_ordinal] + linha_pixeis1

            query_bands.append(linha_pixeis)
    
    df1 = pd.DataFrame(query_bands)
    df2 = df1[~(df1 == 65535).any(axis=1)].reset_index(drop=True)
    df3 = df2.transpose()

    dates, blues, greens, reds, nirs, swir1s, swir2s = df3.values
    ndvis=[]
    for (nir,red) in zip(nirs,reds):
        if nir+red>0:
            ndvis.append((nir-red)/(nir+red))
        else:
            ndvis.append(0)
    #results = ccd.detect(dates, blues, greens, reds, nirs, swir1s, swir2s)
    results = ccd.detect(dates, ndvis, greens, reds, nirs, swir1s, swir2s)
    
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
    
    caminho_arquivo1 = os.path.join(base_folder,'BUFFER_300','pontos_300_buffers_1.gpkg') #"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\pontos_300_buffers 1.gpkg"
    dados_geoespaciais1 = gpd.read_file(caminho_arquivo1)

    ponto_desejado = dados_geoespaciais1.iloc[k].geometry

    dados = [
        {'tBreak': break_dates_epoch,'tEnd': end_dates_epoch,'tStart':start_dates_epoch,'changeProb':prob, 'Lat': ponto_desejado.y,'Lon': ponto_desejado.x}
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

if __name__ == "__main__":
    final_df = main()
    print(final_df)
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