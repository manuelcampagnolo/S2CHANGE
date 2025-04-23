import geopandas as gpd
import pandas as pd
import numpy as np

# Carrega o shapefile
gdf = gpd.read_file(r"C:\Users\scaetano\Downloads\BDR_MIX_TNE\BDR_MIX_TNE.shp")
#%%
# Função para normalizar datas
def formatar_data(data):
    if pd.isnull(data):
        return data
    try:
        data_formatada = pd.to_datetime(str(data), dayfirst=True, errors='coerce')
        return data_formatada.strftime('%d%m%Y')
    except Exception as e:
        return data

# Aplica nas colunas
gdf['data_0'] = gdf['data_0'].apply(formatar_data)
gdf['data_1'] = gdf['data_1'].apply(formatar_data)

gdf['buffer_ID'] = range(1, len(gdf) + 1)
gdf.loc[gdf['buffer_ID'] == 789, 'data_1'] = '20181022'

gdf['classe2018'] = np.nan
gdf['classe2019'] = np.nan
gdf['classe2020'] = np.nan
gdf['classe2021'] = np.nan
gdf['altera'] = np.where(gdf['data_0'].isna(), 'Sem Alteracao', 'Com Alteracao')
cols = ['buffer_ID'] + [col for col in gdf.columns if col != 'buffer_ID' and col != gdf.geometry.name] + [gdf.geometry.name]
gdf = gdf[cols]
#%%
gdf.to_file(r"C:\Users\scaetano\Downloads\BDR_MIX_TNE\BDR_MIX_TNE_new.shp")