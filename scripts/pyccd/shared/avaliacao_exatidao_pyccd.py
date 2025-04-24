import sys, os
from datetime import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
import csv

import warnings
warnings.filterwarnings('ignore')


def inferDelimiter(pathDF):
  with open(pathDF, 'r') as csvfile:
    dialect = csv.Sniffer().sniff(csvfile.readline())
    return dialect.delimiter

def convertDate(data):
  """Retorna ano, mês e dia a partir de data no formato YYYY-MM-DD"""
  data = data.split('-')
  y = int(data[0])
  m = int(data[1])
  d = int(data[2])
  return y,m,d

def filterDate(pathDF, dataI, dataF,bandFilter, mag = None):
    """
    Reduz o número de linhas do data frame de entrada, removendo as linhas fora do período de análise e
    para o limite estabelecido de magnitude máxima.
    Entrada:
        pathDF: caminho do Data Frame do CCDC
        dataI: String com a data inicial na forma = 'AAAA-MM-DD' (e.g. a data inicial dos analistas nos pontos DGT 300)
        dataF: String com a data final na forma = 'AAAA-MM-DD' (e.g. a data final dos analistas nos pontos DGT 300)
        bandFilter: String com a banda para a qual se deseja filtrar os dados. A esta banda é aplicado o criterio do mag.
        mag: Número com o limite da magnitude, e.g 0 só serão utilizadas as linhas com magnitudo menor ou igual a zero
    Saída:
        Data Frame filtrado
    """
    # Data Frame CCDC
    if pathDF.endswith('.csv'):
        delimiter = inferDelimiter(pathDF)
        df = pd.read_csv(pathDF, delimiter = delimiter)
    if pathDF.endswith('.pkl'):
        df = pd.read_pickle(pathDF)

    for dtCol in df.columns:
        if 'tBreak' in dtCol or 'tEnd' in dtCol or 'tStart' in dtCol:
            mask = df.loc[:, dtCol] == 0
            df[dtCol] = pd.to_datetime(df[dtCol], unit = 'ms')
            df.loc[mask, dtCol] = np.nan
        elif 'End_S' in dtCol:
            df[dtCol] = pd.to_datetime(df[dtCol]) # Esta coluna inicialmente esta em formato texto
    df.rename(columns={ 'Unnamed: 0':'IDCCDC'}, inplace=True)

    if mag != None:
        # caso haja magnitude limite, colocar tudo como NAT que seja acima deste limite
        df.loc[df[bandFilter] > mag, 'tBreak'] = pd.to_datetime(np.nan)
        df = df.copy()
    else:
        df = df.copy()

    # filtro das datas
    yi, mi, diai = convertDate(dataI)
    fltInicial = datetime(yi, mi, diai)
    yf, mf, diaf = convertDate(dataF)
    fltFinal = datetime(yf, mf, diaf)

    # 1 Adiciona a coluna com a menor data de start do fit
    df['startMin'] = df.groupby(['coord_ccdc'])['tStart'].transform('min')

    # 2 Adiciona o número de breaks existentes num grupo de IDCCDC, independente de fltInicial e fltFinal
    df['numBreak'] = np.ceil(df.groupby(['coord_ccdc'])['changeProb'].transform('sum'))

    # Colocar Nat nas probabilidades fracionadas
    df.loc[((df.changeProb > 0) & (df.changeProb < 1)), 'tBreak'] = pd.to_datetime(np.nan)

    # 3 Verifica se se os breaks estão dentro do período de análise e transforma em NaT todos os que não estão
    df['breaks_in_tmask'] = (~df.tBreak.isnull()).astype(int)
    df.loc[(df['tBreak'] <= fltInicial) | (df['tBreak'] >= fltFinal), 'breaks_in_tmask'] = 0
    df.loc[(df['tBreak'] <= fltInicial) | (df['tBreak'] >= fltFinal), 'tBreak'] = np.nan

    # Mascaras necessárias
    # a) Verifica os breaks NaT para as linhas com mais de 1 break
    mask = pd.Series(np.zeros(len(df),dtype=bool),index = df.index)
    mask.loc[(df.tBreak.isnull()) & (df.numBreak > 1)]= True #cond3

    # b) Verifica nas linhas de 1 break e sejam nulos qual é aquele que tem o início da série,
    #pois caso esteja fora da data de análise deve ser eliminado
    nmask = pd.Series(np.zeros(len(df),dtype=bool),index = df.index)
    nmask.loc[(df.tBreak.isnull()) & (df.numBreak == 1) & (df.breaks_in_tmask == 0) & (df.tStart == df.startMin)]= True

    # Aplica as mascaras acima e gera um novo DF
    subset_Filtro = df[((mask == False) & (nmask == False))].copy()

    # c) Calcula quantos linhas há por IDCCDC e caso ainda existam 2 significa que o break está dentro do período de análise e o fit final, sem break
    # deve ser eliminado
    smask = pd.Series(np.zeros(len(subset_Filtro),dtype=bool),index = subset_Filtro.index)
    smask.loc[(subset_Filtro.groupby(['coord_ccdc'])['IDCCDC'].transform('count') == 2) & (subset_Filtro.changeProb == 0) & (df.numBreak == 1)] = True
    subset_Filtro = subset_Filtro[(smask == False)].copy()

    # d) Para os IDCCDC que apresentam linhas com probabilidade fracionada, mantem esta linha, no caso de todas estarem fora do período de análise
    pmask = pd.Series(np.zeros(len(df),dtype=bool),index = df.index)
    pmask.loc[~((df.changeProb > 0) & (df.changeProb < 1) & (df.tBreak.isnull()) & (df.groupby(['coord_ccdc'])['tBreak'].transform('count') == 0))]=True
    subset_Filtro = pd.concat([subset_Filtro,df[pmask == False]])#subset_Filtro.append(df[pmask == False])

    # e) Para os IDCCDC que tem mais de um break e todos estao fora do periodo e devemos manter o fit final
    fmask = pd.Series(np.zeros(len(df),dtype=bool),index = df.index)
    fmask.loc[((df.changeProb == 0) & (df.numBreak > 1) & (df.tBreak.isnull()) & (df.groupby(['coord_ccdc'])['tBreak'].transform('count') == 0))]=True
    subset_Filtro = pd.concat([subset_Filtro,df[fmask]])#subset_Filtro.append(df[fmask])


    return subset_Filtro

def spatialJoin(pathPoligonosDGT, dfCCDC):
  """
  Realizar o spatial join entre o dataframe do CCDC e os poligonos com alteracoes identificadas pela DGT
  Entrada:
   - pathPoligonosDGT: String com o caminho completo dos poligonos desenhados pela DGT
   - pathDataFrameCCDC: Data Frame filtrado do CCDC
   Saida:
   """
  # 1) ABRIR OS ARQUIVOS
  ## Poligonos DGT
  gdfVal = gpd.read_file(pathPoligonosDGT)
  gdfVal.to_crs(crs = 'EPSG:3763', inplace = True) # Originalmente eles estao em WGS84 29N converte para ETRS
  ## Pontos ISA

  # 2) CONVERTER O DF PARA GEO DF
  gdfCCDC = gpd.GeoDataFrame(dfCCDC, geometry = gpd.points_from_xy(dfCCDC.longitude, dfCCDC.latitude), crs=32629) # old csvs - crs=4326
  gdfCCDC.to_crs(crs=4326, inplace=True)

  ## criar a bordadura
  ###idBord = identity.copy() # cria uma copia do identity gerado acima
  idBord = gdfVal.copy()
  idBord['geometry'] = idBord.geometry.buffer(-10) # reduz a geometria em 10 metros
  idBord.drop(list(idBord.columns)[:-1], axis = 1, inplace = True) #remove todas as colunas menos a da geometria
  idBord['bordadura'] = 1 # cria uma nova coluna para poder identificar a borda dura
  ## novo identity para termos a area da borda dura

  ###identity = gpd.overlay(identity, idBord, how='identity')
  identity = gpd.overlay(gdfVal, idBord, how = 'identity')

  # Como o poligono inicial nao tinha a coluna de bordadura, há feições onde
  # temos 1 e Nulos, com a linha abaixo invertemos o campo onde era Nullo passa a True
  # e onde era 1 passa para False, ou 1 e 0
  identity.bordadura = identity.bordadura.isnull()
  # Convertemos o resultado para WGS84
  identity.to_crs(crs = 'EPSG: 4326', inplace = True)

  
  ## As datas da DGT estao no formato (20200103) e precisam ser convertidas
  for dataCol in ['data_0', 'data_1', 'data_2', 'data_3']:
      # primeiro converter para datetime
      maskZero = pd.Series(np.zeros(len(identity),dtype=bool))
      erro = identity[dataCol].isnull()
      identity.loc[erro, dataCol] = 0
      # converter tudo para inteiros e onde for 0 indicar 1970
      identity[dataCol] = identity[dataCol].astype(int)
      maskZero = identity.loc[:, dataCol] == 0
      identity.loc[maskZero, dataCol] = 19700101
      # converter para datetime
      identity[dataCol] = pd.to_datetime(identity[dataCol], format = '%Y%m%d')
      identity.loc[maskZero, dataCol] = np.nan


  # 4) SPATIAL JOIN ENTRE OS CENTROIDES DO CCDC COM OS BUFFERS DE 200 METROS
  subset = gpd.sjoin(gdfCCDC, identity, how='inner')
  subset.reset_index(inplace = True)
  subset['buffer_ID'] = subset.buffer_ID.astype('int')

  
  #Descobrir quais linhas precisam ser duplicadas.
  #Pressupondo que não é possível ter informação da 'data_3' sem existir a 'data_1'
  #é possível filtrar e verificar a negação de quais dados são nulos e depois somar
  #o reultado.
  #0 = False False: não há data_1 e nem data_3
  #1 = True  False: existe data_1 e não data_3
  #2 = True  True: existem data_1 e Data_3
  
  cond = ~subset.filter(items=['data_1', 'data_3']).isnull()
  subset['analistas'] = cond.sum(axis=1)
  subset.loc[subset['analistas'] == 0, 'exists_event'] = False # Analista nao identificou nada
  subset.loc[subset['analistas'] > 0, 'exists_event'] = True # Analista identificou alteracao

  
  #CRIA UM DF TEMPORARIO PARA COPIAR AS LINHAS ONDE EXISTEM A 'DATA_3' E INSERE ESTA DATA NO CAMPO 'DATA1_Z'
  #DEPOIS ADICIONA ISTO AO DATA FRAME ORIGINAL
  
  subset['data1_z'] = ''
  # criar coluna para as datas anteriores
  # subset['data0_z'] = ''
  subset['nome'] = '' # teste para nomear os analistas
  subset['tipo'] = ''
  subset['classeAnterior'] = ''
  subset['classeAtual'] = ''
  dfTemp = pd.DataFrame(columns = subset.columns)
  for row in subset.itertuples():
    # verifica se há duas datas e duplica a linha
    if row.analistas == 2:
        dfTemp = pd.concat([dfTemp, subset[subset.index==row.Index]],ignore_index=False)#dfTemp.append(subset[subset.index == row.Index], ignore_index=False)
  dfTemp.data1_z = dfTemp.data_3
  # capturar o valor da data_2
  # defTemp.data0_z = dfTemp.data_2
  dfTemp.nome = 'B' # teste para nomear os analistas
  dfTemp.tipo = dfTemp.tipo_2
  dfTemp.classeAtual = dfTemp.classe_3
  dfTemp.classeAnterior = dfTemp.classe_2

  subset.data1_z = subset.data_1
  # capturar o valor da data_0
  # subset.data0_z = subset.data_0
  subset.nome = 'A' # teste para nomear os analistas
  subset.tipo = subset.tipo_1
  subset.classeAtual = subset.classe_1
  subset.classeAnterior = subset.classe_0

  subset = pd.concat([subset, dfTemp],ignore_index=False)#subset.append(dfTemp, ignore_index=False)

  # Contagem do numero de breaks
  subset['Valid_breaks'] = np.ceil(subset.groupby(['coord_ccdc', 'nome'])['changeProb'].transform('sum'))

  # COLUNA DO DELTA MIN
  subset['delta_min'] = (subset.data1_z - subset.tBreak).dt.days
  subset.drop(['data_1', 'data_3', 'tipo_1', 'tipo_2','classe_0', 'classe_1','classe_2', 'classe_3'], axis = 1, inplace = True)

  # verificar quais colunas tem magnitude de indices
  mags = [ t for t in subset.columns if 'magnitude' in t and not 'B' in t]
  ordem = [ 'coord_ccdc','buffer_ID', 'IDCCDC', 'altera', 'changeProb'] + mags + ['tBreak', 'data1_z',
         'bordadura', 'classe2018', 'classe2019', 'classe2020','classe2021', 'classeAnterior','tipo',
         'classeAtual', 'analistas', 'nome', 'exists_event', 'Valid_breaks' , 'delta_min', 'geometry']


  return subset[ordem], subset


def preprocessCsvS2(csv_s2, end_of_series):
  """
  Does a pre-processing of the csv containing detection results to ensure it has the necessary columns
  and coherent values for the validation procedure.
  Args:
    csv_s2: a pandas dataframe obtained after reading the csv file containing ccd detection results;
    end_of_series: date of the last image in the series - a string in the form YYYY-mm-dd.
  Returns:
    Pre-processed dataframe.
  """
  
  csv_s2 = csv_s2.copy()
  from ast import literal_eval
  #do some processing on the csv
  # Selecionar as colunas a explodir e as dos coeficientes
  tabExplode = []
  tabCoefs = []
  for c in csv_s2.columns:
    if 'coefs' in c or 'magnitude' in c or 'rmse' in c:
      tabExplode.append(c)
    if 'coefs' in c:
      tabCoefs.append(c)
  tabExplode = tabExplode + ['changeProb', 'tBreak', 'tEnd', 'tStart']

  #convert from string of list to list
  for col in tabExplode:
    try:
      csv_s2[col] = csv_s2[col].apply(literal_eval)
    except: #sometimes CCDC returns 'Infinity' or 'NaN' as a rmse value, which results in literal_eval not working
      #csv_s2[col] = csv_s2[col].apply(lambda x: x.replace('Infinity','9999999'))
      #csv_s2[col] = csv_s2[col].apply(lambda x: x.replace('NaN','-9999999'))
      csv_s2[col] = csv_s2[col].apply(literal_eval)
  #convert lat long separated by comma to separated by point
  #csv_s2['Lat'] = csv_s2['Lat'].apply(lambda x: x.replace(",","."))
  #csv_s2['Lon'] = csv_s2['Lon'].apply(lambda x: x.replace(",","."))

  #explode
  csv_s2 = csv_s2.explode(tabExplode)

  csv_s2['End_S'] = end_of_series
  csv_s2['coord_ccdc'] = list(zip(csv_s2.Lat, csv_s2.Lon))
  csv_s2['Dist_Point'] = -1#''
  csv_s2['Point_Val'] = -1#''

  #convert date columns from float to int
  for col in ['tBreak', 'tEnd', 'tStart']:
    csv_s2[col] = csv_s2[col].astype('int64')

  csv_s2.rename(columns={'Lat':'latitude','Lon':'longitude'}, inplace=True)

  return csv_s2


def preprocessParquetS2(parquet_directory, end_of_series):
  """
  Does a pre-processing of a directory containing parquet files with detection results to ensure it has the necessary columns
  and coherent values for the validation procedure.
  Args:
    parquet_directory: path to a directory containing parquet files with ccd detection results;
    end_of_series: date of the last image in the series - a string in the form YYYY-mm-dd.
  Returns:
    Pre-processed dataframe.
  """
  
  column_names = ['tBreak', 'tEnd', 'tStart', 'changeProb', 'x_coord', 'y_coord', 'coeficientes']
  main_df = pd.DataFrame(columns=column_names)
  
  for file in os.listdir(parquet_directory):
    if file.endswith('.parquet'):
      file_path = os.path.join(parquet_directory, file)
      temp_df = pd.read_parquet(file_path)
      temp_df = temp_df[column_names].copy()
      main_df = pd.concat([main_df, temp_df], ignore_index=True)
  
  main_df.reset_index(drop=True, inplace=True)

  main_df.rename(columns={'x_coord':'longitude','y_coord':'latitude'}, inplace=True)
  main_df['End_S'] = end_of_series
  main_df['coord_ccdc'] = list(zip(main_df.latitude, main_df.longitude))
  main_df['Dist_Point'] = -1
  main_df['Point_Val'] = -1

  main_df = fix_changeProb(main_df)

  return main_df

def fix_changeProb(df):

    #gets original column names
    cols = df.columns

    #puts changeProb in the 0-1 range
    df['changeProb'] = df['changeProb']/100

    #creates a column to store the number of breaks in a group
    df['count_breaks'] = df.groupby('coord_ccdc')['tStart'].transform('count')
    #creates a column to store the maximum tbreak in a group
    df['max_tbreak_group'] = df.groupby('coord_ccdc')['tBreak'].transform('max')
    #sets the changeProb to 1 in all segments, except the last one (the one with largest tBreak)
    df.loc[(df['count_breaks'] > 1) & (df['max_tbreak_group']!=df['tBreak']), 'changeProb'] = 1

    return df[cols].copy()

# função de validação do data frame
def valPol(df, theta):
  """
  Esta função recebe o geodataframe gerado no spatialJoin() e contabiliza as métricas de positivos e negativos.
  A Saída é a matriz com os cálculos e um dicinário com as métricas contabilizadas.

  """

  # transforma a coluna de delta min para valor absoluto e cria uma nova coluna com o mínimo delta min por ponto
  df.reset_index(inplace = True)
  original_delta_min = df['delta_min'].copy()
  df['delta_min'] = abs(df['delta_min'].fillna(99999)) # substitui os nullos para evitar que sejam os minimos
  df['Min_delta_min'] = df.groupby(['coord_ccdc', 'nome'])['delta_min'].transform('min') # calcula o valor minimo por ponto
  df['delta_min'] = abs(original_delta_min) # retorna o valor absoluto da coluna original
  df['Min_delta_min'] = df['Min_delta_min'].replace(99999,np.nan) # substitui os 99999 por nullos

  bf = df.copy()

  bf['Valid_breaks'] = bf.groupby(['coord_ccdc', 'nome']).transform('count')[['tBreak']] # verifica os breaks validos por pontos
  # SE O TBREAK FOR OBJETO ELE JAMAIS SERA NULO, CONVERTER PARA DATA.
  bf.tBreak = pd.to_datetime(bf.tBreak)
  bf.tStart = pd.to_datetime(bf.tStart)
  bf.tEnd = pd.to_datetime(bf.tEnd)
  bf.analistas = bf.analistas.astype(int)
  bf.exists_event = bf.exists_event.astype(int)
  bf.buffer_ID = bf.buffer_ID.astype(int)
  bf.IDCCDC = bf.IDCCDC.astype(int)

  ## ALGUMAS MASCARAS INICIAIS NECESSARIAS
  # mascara dos breaks a mais que analistas ainda em reformulacao

  # PARA O CASO DE TER SOMENTE UM BREAK FP E DOIS ANALISTAS PARA NAO TER DUPLICACAO
  mask = pd.Series(np.zeros(len(bf),dtype=bool), index= bf.index)
  mask.loc[(bf.analistas == 2) & (bf.Valid_breaks < bf.analistas) ] = True #& (bf.delta_min > theta)

  bf.loc[mask, 'Min_delta_min'] = bf.loc[mask].groupby(['coord_ccdc'])['delta_min'].transform('min')

  # Contabilizar
  # colocar todos os VP (delta_min <=31)
  #VP
  bf.loc[( (bf.delta_min <= theta) & (~bf.tBreak.isnull()) & (bf.analistas > 0) ), 'VP'] = 1
  # #FP
  # # sem a condição da magnitude ou (changeProb ==1) serao selecionados os que devem ser negativos
  # bf.loc[( (bf.analistas == 0) & (bf.ndvi_magnitude != 0) & (~bf.tBreak.isnull())), 'FP' ] = 1 #FP puro
  # bf.loc[( (bf.delta_min > theta) & (bf.ndvi_magnitude != 0) & ( (bf.delta_min == bf.Min_delta_min) & (~bf.Min_delta_min.isnull()) ) )  , 'FP' ] = 1
  # bf.loc[( (bf.delta_min > theta) & (bf.ndvi_magnitude != 0) & (bf.analistas == 1)  ) & (~bf.tBreak.isnull()), 'FP' ] = 1
  #FP
  # sem a condição da magnitude ou (changeProb ==1) serao selecionados os que devem ser negativos
  bf.loc[( (bf.analistas == 0) & (~bf.tBreak.isnull())), 'FP' ] = 1 #FP puro
  bf.loc[( (bf.delta_min > theta) & ( (bf.delta_min == bf.Min_delta_min) & (~bf.Min_delta_min.isnull()) ) )  , 'FP' ] = 1
  bf.loc[( (bf.delta_min > theta) & (bf.analistas == 1)  ) & (~bf.tBreak.isnull()), 'FP' ] = 1
  #FN
  bf.loc[( (bf.analistas > 0)  & (bf.tBreak.isnull()) ), 'FN' ] = 1 # FN puro
  # falsos negativos que precisam ser contabilizado para os FPs
  bf.loc[(bf.analistas == 1) & (bf.Valid_breaks == 1) & (bf.FP == 1), 'FN'] = 1 # parece funcionar
  bf.loc[(bf.analistas == 2) & (bf.Valid_breaks == 3) & (bf.FP == 1) , 'FN'] = 1

  #VN
  bf.loc[( (bf.analistas == 0) & (bf.tBreak.isnull()) ), 'VN' ] = 1

  # converter os NaN para 0
  bf[['VP', 'FP', 'FN', 'VN']] = bf[['VP', 'FP', 'FN', 'VN']].fillna(0)

  # verificar os breaks que nao foram classificados
  # para isso gero uma coluna total onde somo todas as metricas, as linhas onde ha 0 nao foram classificadas
  bf['total'] = bf.VP + bf.FP +bf.FN + bf.VN
  mask = pd.Series(np.zeros(len(bf),dtype=bool), index= bf.index) #mascara
  # agrupar por coordenada e t break, assim as somente os breaks que nao foram validados para nenhum analista terao valor 0
  mask.loc[(bf.groupby(['coord_ccdc','tBreak'])['total'].transform('sum')==0) & (bf.analistas == 2) & (bf.Valid_breaks > bf.analistas)] = True
  # neste grupo selecionado devo procurar aquele que tem menor distancia para um analista e classificar como FP
  mask2 = bf[mask].groupby(['coord_ccdc'])['delta_min'].transform('min') == bf.delta_min[mask]
  # agora classificar os candidatos que atendem as duas mascaras
  bf.loc[(mask & mask2), ['FP']] = 1

  # Ajuste FN
  # se for na célula anterior isso contará para o total e a mascara anterior não será feita em alguns pontos onde deve ser feita
  bf.loc[((bf.FP ==1) & (bf.analistas == 1) & (bf.delta_min == bf.Min_delta_min) & (bf.Valid_breaks == 2))   , 'FN' ] = 1
  bf.loc[((bf.FP ==1) & (bf.analistas == 1) & (bf.delta_min == bf.Min_delta_min) & (bf.Valid_breaks == 3))   , 'FN' ] = 1
  bf.loc[(bf.analistas == 2) & (bf.Valid_breaks == 1) & (bf.VP == 0), 'FN'] = 1
  bf.loc[(bf.analistas == 2) & (bf.Valid_breaks == 2) & (bf.FP == 1), 'FN'] = 1
  #return bf
  # Bloco para corrigir o problema de quando as duas datas DGT estão mais próximas do mesmo break
  # listar as coordenadas que tem o problema com mesmo break classificado
  listCoord = list(bf.coord_ccdc[(bf.groupby(['coord_ccdc','tBreak'])['total'].transform('sum') == 0) & (bf.analistas == 2) & (bf.Valid_breaks == 2)])
  #return listCoord
  # dividir o data frame em dois para poder limpar as linhas com problema
  bf_filter = bf.loc[~bf.coord_ccdc.isin(listCoord)].copy()
  # limpeza
  bf_remove_lines = bf.loc[bf.coord_ccdc.isin(listCoord)].copy()
  # zerar todas as métricas para poder recalcular
  bf_remove_lines.loc[:, ['VP','VN','FP', 'FN']] = 0
  #return bf_remove_lines
  bf_removed = bf_remove_lines.groupby(['buffer_ID','IDCCDC']).apply(testeRemove).copy() # função de remoção
  #return bf_removed
  try:
    bf_removed = bf_removed.drop(columns=['buffer_ID','IDCCDC']).reset_index() # evitar problema de indece dup.
  except:
    pass
  # Agora teremos somente duas linhas por ponto que são obrigatóriamente FP ou VP
  #VP
  bf_removed.loc[( (bf_removed.delta_min <= theta) ), 'VP'] = 1
  #FP, FN
  bf_removed.loc[( (bf_removed.delta_min > theta) ), ['FP', 'FN']] = 1
  # unir os dois dfs novamente
  bf_final = pd.concat([bf_filter, bf_removed])#bf_filter.append(bf_removed)

  # remover aqueles que nao possuem metrica
  bf_final = bf_final[(bf_final.VP > 0) | (bf_final.FP > 0) | (bf_final.FN > 0) | (bf_final.VN > 0) ].copy()
  # remover aqueles que apresentam as classes especificas
  bf_final = bf_final[~(bf_final.tipo.isin(['Agricultura','Agua']))].copy()


  # verificar quais colunas tem magnitude de indices
  mags = [ t for t in bf_final.columns if 'magnitude' in t and not 'B' in t]
  # colunas para retornar um DF mais limpo
  c = ['buffer_ID', 'IDCCDC', 'coord_ccdc', 'changeProb'] + mags + ['tBreak',
       'data1_z', 'analistas', 'nome', 'exists_event', 'Valid_breaks',
       'delta_min', 'Min_delta_min', 'VP', 'FP', 'FN', 'VN'] #geometry
  # também poderá retornar o DF todo classificado, em processo.
  return bf_final[c], bf_final


# função para realizar a limpeza de linhas indesejadas
def testeRemove(groupedby):
  min_delta_min = groupedby['Min_delta_min'].min()
  #remove rows only if there is more than 1 row per point, the number of analyst dates is not zero and min_delta_min is greater than zero.
  if len(groupedby) > 1 and groupedby.analistas.min() > 0 and min_delta_min >= 0:
    # Updated section with check on matching rows
    # Add a check to see if there are any rows matching the condition
    matching_rows = groupedby.loc[groupedby['delta_min']==min_delta_min][['tBreak','data1_z']]
      
    if len(matching_rows) > 0:  # Only proceed if matching rows
      Bj, Ai = matching_rows.values[0]
      mask = ((groupedby['tBreak'] == Bj) | (groupedby['data1_z'] == Ai)) & (groupedby['delta_min']!=min_delta_min)
      groupedby = groupedby[~mask]

    # original
    # Bj, Ai = groupedby.loc[groupedby['delta_min']==min_delta_min][['tBreak','data1_z']].values[0]
    # #remove rows that contain Ai or Bj (other than the row with the min_delta_min)
    # mask = ((groupedby['tBreak'] == Bj) | (groupedby['data1_z'] == Ai)) & (groupedby['delta_min']!=min_delta_min)
    # groupedby = groupedby[~mask]

  return groupedby


# def runValidation(filename, FOLDER_CSV, BDR_DGT, dt_ini, dt_end, bandFilter, theta):
#     """
#     Corre a validação dos resultados da deteção realizando spatial join
#     com a base de dados de referência.
#     Imprime as métricas de validação e gera ficheiro csv VAL.

#     Args:
#         filename: nome do ficheiro csv guardado anteriormente com resultados
#                   da deteção. Nome do ficheiro é definido em função dos parâmetros
#                   de execução utilizados;
#         FOLDER_CSV: pasta para buscar o ficheiro csv e guardar o resultado da
#                         validação;
#         BDR_DGT: caminho para o ficheiro da base de dados de validação;
#         dt_ini: data inicial do período de referência dos analistas (str YYYY-mm-dd);
#         df_end: data final do período de referência dos analistas (str YYYY-mm-dd);
#         bandFIlter: não implementado ainda;
#         theta: margem de tolerância da validação, em dias (int);
    
#     Returns: 
#         None (imprime métricas e gera ficheiro csv)

#     """
#     print('A correr validação dos resultados do ccd...')
#     #pegar data do fim da serie temporal (ultima imagem)
#     reference_index = filename.find('END')
#     end_of_series = filename[reference_index + 3 : reference_index + 11]
#     year, month, day = [end_of_series[:4], end_of_series[4:6], end_of_series[6:]]
#     end_of_series = f"{year}-{month}-{day}"


#     csv_s2 = pd.read_csv(FOLDER_CSV / '{}.csv'.format(filename))
#     #correr pre-processamento
#     csv_s2 = preprocessCsvS2(csv_s2, end_of_series)
#     csv_preprocessed_path = '{}_pre_proc.csv'.format(filename)
#     csv_s2.to_csv(csv_preprocessed_path)

#     """## Filtrar datas
#     Limitar análise ao período considerado pelos analistas DGT
#     """
#     #correr filtro de datas
#     ccdcFiltro = filterDate(csv_preprocessed_path, dt_ini, dt_end, bandFilter)
#     """## Spatial join
#     Faz join dos pontos do csv com a informação de referencia da DGT (300 buffers). É associada aos pontos a informação da validação - data de alteração, tipo, classes, etc.
#     """
#     #gdfVal = gpd.read_file(BDR_DGT)
#     #gdfVal.to_crs(crs = 'EPSG:3763', inplace = True)
#     #executa o join
#     ccdcVal, ccdcVal_T = spatialJoin(BDR_DGT, ccdcFiltro)
#     """## Validação
#     Faz a validação da deteção - compara resultado do modelo (ccd) com dados de referência DGT
#     """ 
#     #faz a validação da deteção
#     DF_FINAL, DF_FINAL_T = valPol(ccdcVal_T, theta) #funcoes.valPol
#     """**Resultados da validação**"""
#     #delimita análise apenas para pontos referentes a transições entre Pinheiro Bravo e Eucalipto para Superfície sem vegetação, herbáceas e matos
#     #elimina também pontos da bordadura
#     df_aux = DF_FINAL_T.copy()
#     df_aux = df_aux.loc[(df_aux.altera=="Sem Alteracao")|((df_aux.altera=="Com Alteracao")&(df_aux.classeAnterior.isin(['Pinheiro bravo','Eucalipto']))&(df_aux.classeAtual.isin(['Superficie sem vegetacao escura','Superficie sem vegetacao clara','Vegetacao herbacea espontanea','Matos'])))]
#     df_aux = df_aux.loc[df_aux.bordadura==0]
#     #imprime f1-score, erro e omissão e erro de comissão
#     cm = df_aux.FP.sum()/(df_aux.FP.sum()+df_aux.VP.sum())
#     om = df_aux.FN.sum()/(df_aux.FN.sum()+df_aux.VP.sum())
#     f1 = 2*(1-om)*(1-cm)/(2-om-cm)
#     print("Métricas de validação para ficheiro:")
#     print(filename)
#     print('F1-score = {}%'.format(round(100*f1,2)))
#     print('Omission error = {}%'.format(round(100*om,2)))
#     print('Commission error = {}%'.format(round(100*cm,2)))

#     DF_FINAL_T.to_csv(FOLDER_CSV / f'VAL_{filename}.csv', index=False)


def runValidation(FOLDER_PARQUET, BDR_DGT, dt_ini, dt_end, bandFilter, theta):
    """
    Corre a validação dos resultados da deteção realizando spatial join
    com a base de dados de referência.
    Imprime as métricas de validação e gera ficheiro csv VAL.

    Args:
        FOLDER_PARQUET: path to folder containing CCDC results in parquet files, will save results;
        BDR_DGT: caminho para o ficheiro da base de dados de validação;
        dt_ini: data inicial do período de referência dos analistas (str YYYY-mm-dd);
        df_end: data final do período de referência dos analistas (str YYYY-mm-dd);
        bandFIlter: não implementado ainda;
        theta: margem de tolerância da validação, em dias (int);
    
    Returns: 
        None (imprime métricas e gera ficheiro csv)

    """

    print('A correr validação dos resultados do ccd...')
    #pegar data do fim da serie temporal (ultima imagem)

    single_file = os.listdir(FOLDER_PARQUET)[0]
    reference_index = single_file.find('END')
    end_of_series = single_file[reference_index + 3 : reference_index + 11]
    year, month, day = [end_of_series[:4], end_of_series[4:6], end_of_series[6:]]
    end_of_series = f"{year}-{month}-{day}"

    results_path = os.path.join(FOLDER_PARQUET, "accuracy_assessment")
    if not os.path.exists(results_path):
       os.makedirs(results_path)
    
    #correr pre-processamento
    csv_s2 = preprocessParquetS2(FOLDER_PARQUET, end_of_series)
    csv_preprocessed_path = os.path.join(results_path, 'pre_proc.csv')
    csv_s2.to_csv(csv_preprocessed_path)

    """## Filtrar datas
    Limitar análise ao período considerado pelos analistas DGT
    """
    #correr filtro de datas
    ccdcFiltro = filterDate(csv_preprocessed_path, dt_ini, dt_end, bandFilter)
    """## Spatial join
    Faz join dos pontos do csv com a informação de referencia da DGT (300 buffers). É associada aos pontos a informação da validação - data de alteração, tipo, classes, etc.
    """
    #gdfVal = gpd.read_file(BDR_DGT)
    #gdfVal.to_crs(crs = 'EPSG:3763', inplace = True)
    #executa o join
    ccdcVal, ccdcVal_T = spatialJoin(BDR_DGT, ccdcFiltro)
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
    print("Métricas de validação para ficheiro:")

    group_name = single_file.split("_rank_")[0]

    print(group_name)
    print('F1-score = {}%'.format(round(100*f1,2)))
    print('Omission error = {}%'.format(round(100*om,2)))
    print('Commission error = {}%'.format(round(100*cm,2)))

    DF_FINAL_T.to_csv(os.path.join(results_path, f'VAL_{group_name}.csv'), index=False)
#%%
# ---------------------------------
#      PARAMETROS DA VALIDACAO
# ---------------------------------
# datas do filtro das datas da analise (DGT 300)
dt_ini = '2018-09-12' # data inicial
dt_end = '2021-09-30' # data final
# Margem de tolerancia entre a quebra do Modelo e do Analista
theta = 60 # +/- theta dias de diferenca
# banda a filtrar com base na magnitude
bandFilter = None #nao implementado ainda - nao mexer

FOLDER_PARQUET = r'C:\Users\scaetano\Downloads\T29TNE'
BDR_DGT = r'C:\Users\Public\Documents\BDR_300_artigo\BDR_CCDC_TNE_Adjusted.shp'
runValidation(FOLDER_PARQUET, BDR_DGT, dt_ini, dt_end, bandFilter, theta)
