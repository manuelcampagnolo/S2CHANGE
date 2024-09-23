import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
#%%
def plotFromCSV(csv_file, row_index=0, save_dir=None):
    """
    Plota os dados de um arquivo CSV contendo informações do CCD.
    
    Args:
    - csv_file (str): Caminho para o arquivo CSV contendo os dados.
    - row_index (int): Índice da linha do CSV a ser utilizada para plotar.
    - save_dir (str): Diretório onde o gráfico será salvo. Se None, o gráfico não será salvo.
    
    Returns:
    - Gráfico para um pixel com as informações do CCD.
    """
    # Ler o cabeçalho
    header = pd.read_csv(csv_file, nrows=0).columns

    # Ler apenas a linha especificada no CSV
    df = pd.read_csv(csv_file, skiprows=lambda x: x != row_index + 1, nrows=1, header=None)

    # Aplicar o cabeçalho ao DataFrame
    df.columns = header
    
    # Garantir que as colunas sejam convertidas corretamente
    df['ndvis'] = df['ndvis'].apply(eval)
    df['dates'] = df['dates'].apply(eval)
    df['prediction_dates'] = df['prediction_dates'].apply(eval)
    df['predicted_values'] = df['predicted_values'].apply(eval)
    df['coeficientes'] = df['coeficientes'].apply(eval)
    df['mask'] = df['mask'].apply(eval)
        
    # Extrair a linha especificada pelo índice do CSV
    row = df.iloc[0]
    
    ndvis = np.array(row['ndvis'])
    dates = np.array(row['dates'])
    prediction_dates = [np.array(d) for d in row['prediction_dates']]
    predicted_values = [np.array(v) for v in row['predicted_values']]
    coeficientes = row['coeficientes']
    mask = np.array(row['mask'])

    ponto_desejado_wgs_x = row['Lon']
    ponto_desejado_wgs_y = row['Lat']
    break_dates = [datetime.fromtimestamp(b / 1000).toordinal() for b in eval(row['tBreak'])]
    start_dates = [datetime.fromtimestamp(s / 1000).toordinal() for s in eval(row['tStart'])]
    end_dates = [datetime.fromtimestamp(e / 1000).toordinal() for e in eval(row['tEnd'])]

    # Plotagem
    plt.style.use('ggplot')
    fg = plt.figure(figsize=(14, 4), dpi=90)
    
    limite_inicial = datetime.strptime('2018-01-01', '%Y-%m-%d')
    limite_final = datetime.strptime('2023-12-31', '%Y-%m-%d')
    
    a1 = fg.add_subplot(1, 1, 1, xlim=(limite_inicial, limite_final))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    
    a1.xaxis.set_major_locator(mdates.YearLocator(1))
    a1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    
    colors = ['orange', 'purple', 'brown']
    
    # Predicted curves
    for idx, (_preddate, _predvalue, _coef) in enumerate(zip(prediction_dates, predicted_values, coeficientes)):
        _preddate = [datetime.fromordinal(int(ordinal)) for ordinal in _preddate]
        color = colors[idx % len(colors)]
        coef_str = f"({', '.join([f'{c:.2f}' for c in _coef])})"
        label = f'Predicted values {idx + 1} (Coefs: {coef_str})'
        a1.plot(_preddate, _predvalue, color, linewidth=1, label=label)
    
    date_objects1 = [datetime.fromordinal(int(ordinal)) for ordinal in dates]
    a1.plot(np.array(date_objects1)[mask], np.array(ndvis)[mask], 'g+', label='Observed values')
    a1.plot(np.array(date_objects1)[~mask], np.array(ndvis)[~mask], 'g+')
    
    ticks = [min(date_objects1) + timedelta(days=i*365) for i in range(10) if min(date_objects1) + timedelta(days=i*365) <= datetime(2023, 12, 31)]
    plt.xticks(ticks)
    plt.title('Lat:' + str(round(ponto_desejado_wgs_x, 5)) + ' Lon:' + str(round(ponto_desejado_wgs_y, 5)))
    
    a1.plot([], [], color='r', linestyle='--', label='Start dates')
    a1.plot([], [], color='brown', linestyle='--', label='End Dates')
    a1.plot([], [], color='b', linestyle='--', label='Break dates')
    
    for b in break_dates:
        b_date = datetime.fromordinal(b)
        a1.axvline(b_date, color='b', linestyle='--')
        a1.text(mdates.date2num(b_date) + 1, a1.get_ylim()[1], b_date.strftime('%d-%m-%Y'), rotation=90, ha='right', weight='bold', va='top', color='b', size=8)
    
    for s in start_dates:
        s_date = datetime.fromordinal(s)
        a1.axvline(s_date, color='r', linestyle='--')
        a1.text(mdates.date2num(s_date) + 1, a1.get_ylim()[0], s_date.strftime('%d-%m-%Y'), rotation=90, ha='right', weight='bold', va='bottom', color='r', size=8)
    
    for e in end_dates:
        e_date = datetime.fromordinal(e)
        a1.axvline(e_date, color='brown', linestyle='--')
        a1.text(mdates.date2num(e_date) + 1, a1.get_ylim()[0], e_date.strftime('%d-%m-%Y'), rotation=90, ha='right', weight='bold', va='bottom', color='brown', size=8, alpha=0.6)

    # reference_start_date = datetime.strptime('2018-09-12', '%Y-%m-%d')
    # reference_end_date = datetime.strptime('2021-09-30', '%Y-%m-%d')
    # a1.axvspan(reference_start_date, reference_end_date, facecolor='pink', alpha=0.3, label='Período de Referência')

    plt.ylabel('NDVI')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}")
    plt.show()