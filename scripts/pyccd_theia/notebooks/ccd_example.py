import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
import ccd
import numpy as np
def read_data(path):
    return np.genfromtxt(path, delimiter=',', dtype=int).T
#from test.shared import read_data
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import geopandas as gpd
import pandas as pd
from pyproj import Proj, Transformer



caminho_arquivo = os.path.join(module_path, 'Buffers', 'pontos_teste_buffer_v2.gpkg') #"C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\pontos_teste_buffer_v2.gpkg"
dados_geoespaciais = gpd.read_file(caminho_arquivo)

#%%

for i in range(0,5):#len(dados_geoespaciais)):
    print('iteracao', i)
    ponto_desejado = dados_geoespaciais.iloc[i].geometry
    coordenadas = (ponto_desejado.x, ponto_desejado.y)
    
    file_path=os.path.join(module_path,'pontos buffer v2',"Ponto"+str(i)+".csv")
    
    # # Carregue o CSV para um DataFrame do Pandas
    # df = pd.read_csv(file_path)
    
    # # Remova a primeira coluna
    # df = df.iloc[:, 1:]
    # df.iloc[:, 1:] *= 10000
    # df = df.dropna()
    
    # # # Salve o DataFrame de volta como um CSV sem cabeçalho
    # df.to_csv("C:\\Users\\scaetano\\Desktop\\Ponto5.csv", index=False,header=False)
    
    data = np.loadtxt(file_path, delimiter=',')
    rows_to_remove = np.all(data[:, 1:] == 65535, axis=1)
    data = data[~rows_to_remove]
    np.savetxt(file_path, data, delimiter=',', fmt='%d')
    #%%
    data=read_data(file_path)
    #%%
    dates, blues, greens, reds, nirs, swir1s, swir2s = data
    results = ccd.detect(dates, blues, greens, reds, nirs, swir1s, swir2s)
    #%%
    mask = np.array(results['processing_mask'], dtype='bool')

    print('Start Date: {0}\nEnd Date: {1}\n'.format(datetime.fromordinal(int(dates[0])),
                                                    datetime.fromordinal(int(dates[-1]))))
    
    predicted_values = []
    prediction_dates = []
    break_dates = []
    start_dates = []
    end_dates=[]
    coeficientes=[]
    prob=[]
    for num, result in enumerate(results['change_models']):
        print('Result: {}'.format(num))
        print('Start Date: {}'.format(datetime.fromordinal(result['start_day'])))
        print('End Date: {}'.format(datetime.fromordinal(result['end_day'])))
        print('Break Date: {}'.format(datetime.fromordinal(result['break_day'])))
        print('Norm: {}\n'.format(np.linalg.norm([result['green']['magnitude'],
                                                result['red']['magnitude'],
                                                result['nir']['magnitude'],
                                                result['swir1']['magnitude'],
                                                result['swir2']['magnitude']])))
        print('Change prob: {}'.format(result['change_probability']))
        
        days = np.arange(result['start_day'], result['end_day'] + 1)
        prediction_dates.append(days)
        break_dates.append(result['break_day'])
        start_dates.append(result['start_day'])
        end_dates.append(result['end_day'])
        prob.append(result['change_probability'])
        
        intercept = result['nir']['intercept']
        coef = result['nir']['coefficients']
        coeficientes.append(coef)
        
        predicted_values.append(intercept + coef[0] * days +
                                coef[1]*np.cos(days*1*2*np.pi/365.25) + coef[2]*np.sin(days*1*2*np.pi/365.25) +
                                coef[3]*np.cos(days*2*2*np.pi/365.25) + coef[4]*np.sin(days*2*2*np.pi/365.25) +
                                coef[5]*np.cos(days*3*2*np.pi/365.25) + coef[6]*np.sin(days*3*2*np.pi/365.25))
    #%%
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    from datetime import timedelta
    date_objects = [datetime.fromordinal(int(ordinal)) for ordinal in dates]
    
    plt.style.use('ggplot')
    fg = plt.figure(figsize=(14, 3), dpi=90)
    
    a1 = fg.add_subplot(1, 1, 1, xlim=(min(date_objects), max(date_objects)))#, ylim=(0, 1500))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    
    a1.xaxis.set_major_locator(mdates.YearLocator(1))
    a1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    
    colors = ['orange', 'purple', 'brown']
    
    # Predicted curves
    for idx, (_preddate, _predvalue) in enumerate(zip(prediction_dates, predicted_values)):
        # Converter números ordinais de volta para objetos de data
        _preddate = [datetime.fromordinal(int(ordinal)) for ordinal in _preddate]
        color = colors[idx % len(colors)]
        a1.plot(_preddate, _predvalue, color, linewidth=1, label=f'Predicted values {idx + 1}')
    
    a1.plot(np.array(date_objects)[mask], np.array(nirs)[mask], 'g+',label='Observed values')  # Observed values
    a1.plot(np.array(date_objects)[~mask], np.array(nirs)[~mask], 'g+')  # Observed values masked out
    
    
    ticks = [min(date_objects) + timedelta(days=i*365) for i in range(10) if min(date_objects) + timedelta(days=i*365) <= datetime(2023, 10, 3)]
    plt.xticks(ticks)
    plt.title('Continuous Change Detection')
    
    # Linhas verticais para datas de quebra
    for b in break_dates:
        b_date = datetime.fromordinal(b)
        a1.axvline(b_date, color='b', linestyle='--')
        a1.text(mdates.date2num(b_date)+1, a1.get_ylim()[0], b_date.strftime('%d-%m-%Y'), rotation=90, ha='right', va='bottom', color='b',size=8)
    
    # Linhas verticais para datas de início (color='r')
    for s in start_dates:
        s_date = datetime.fromordinal(s)
        a1.axvline(s_date, color='r', linestyle='--')
        a1.text(mdates.date2num(s_date) + 1, a1.get_ylim()[0], s_date.strftime('%d-%m-%Y'), rotation=90, ha='right', va='bottom', color='r',size=8)
    plt.ylabel('BAND NIR')
    plt.legend(fontsize=7)
    plt.tight_layout()
    #plt.show()
    #plt.savefig('C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\graficos buffer v2\\ccdc_ponto_'+str(i)+'.png')
    # plt.close()
    #%%
    datas = [datetime.fromordinal(data) for data in break_dates]
    break_dates_epoch = [int(data.timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in start_dates]
    start_dates_epoch = [int(data.timestamp() * 1000) for data in datas]
    
    datas = [datetime.fromordinal(data) for data in end_dates]
    end_dates_epoch = [int(data.timestamp() * 1000) for data in datas]
    #%%
    import csv
    
    # Dados com tamanhos diferentes
    dados = [
        {'tBreak': break_dates_epoch,'tEnd': end_dates_epoch,'tStart':start_dates_epoch,'changeProb':prob, 'Lat': ponto_desejado.y,'Lon': ponto_desejado.x}
    ]
    
    # Criar DataFrame
    df = pd.DataFrame(dados)
    
    # Reorganizar colunas
    ordem_colunas = ['tBreak', 'tEnd', 'tStart', 'changeProb', 'Lat', 'Lon']
    df = df[ordem_colunas]
    
    # Nome do arquivo CSV
    nome_arquivo_csv = 'C:\\Users\\scaetano\\Desktop\\PPT realizados\\Buffer\\csv.csv'
    
    if False:
        try:
            df_existente = pd.read_csv(nome_arquivo_csv)
            # Concatene o DataFrame existente com o novo
            df = pd.concat([df_existente, df], ignore_index=True)
        except FileNotFoundError:
            # Se o arquivo não existir, crie um novo
            pass
        # Escrever DataFrame para o arquivo CSV
        df.to_csv(nome_arquivo_csv, index=False)