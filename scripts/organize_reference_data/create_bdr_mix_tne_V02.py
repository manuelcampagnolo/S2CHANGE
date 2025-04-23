import geopandas as gpd
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# years outside this range will be converted to None
start_year, end_year = 2017, 2024

# adapt
fn_bdr300=r"C:/Users/mlc/OneDrive - Universidade de Lisboa/Documents/investigacao-projectos-reviews-alunos-juris/projetos/DGT-S2CHANGE_2023/BDR_TNE_300/BDR_CCDC_TNE_Adjusted.shp"
fn_tiles=r"C:/Users/mlc/OneDrive - Universidade de Lisboa/Documents/investigacao-projectos-reviews-alunos-juris/projetos/DGT-S2CHANGE_2023/S2_tile_locations/sentinel2_tiles_PT_terra_tm06.shp"
fn_icnf_2023=r"C:/Users/mlc/OneDrive - Universidade de Lisboa/Documents/investigacao-projectos-reviews-alunos-juris/projetos/DGT-S2CHANGE_2023/ICNF/ardida_2023/ardida_2023.shp"
fn_nvg_poly=r"C:/Users/mlc/OneDrive - Universidade de Lisboa/Documents/investigacao-projectos-reviews-alunos-juris/projetos/DGT-S2CHANGE_2023/dados_ref_nvg/BDR_NVG_S2_V02_polygons/dissolved_buffered_-0.01_meters.shp"

# CRS
crs_bdr300="32629"
crs_nvg_poly="32629"
crs_tiles="3763"
crs_icnf_2023="3763"

# Function to convert DD-MM-YYYY to DDMMYYYY, and nullify dates before start_year and later than end_year
def formatar_data(date_str):
    date_str=str(date_str)
    #sys.exit(date_str)
    # Try parsing with both formats
    for fmt in ('%Y-%m-%d', '%Y%m%d'):
        try:
            dt = datetime.strptime(date_str, fmt)
            # Check year range
            if start_year <= dt.year <= end_year:
                return dt.strftime('%Y%m%d')
            else:
                print('dt.year',dt.year)
                return None
        except ValueError:
            continue
    # If both formats fail
    return None

# reproject
gdf = gpd.read_file(fn_bdr300)
gdf = gdf.set_crs(epsg=crs_bdr300)
gdf_bdr300 = gdf.to_crs(epsg=crs_bdr300)
# nvg
gdf = gpd.read_file(fn_nvg_poly)
gdf = gdf.set_crs(epsg=crs_nvg_poly)
gdf_nvg_poly = gdf.to_crs(epsg=crs_bdr300)
# icnf
gdf = gpd.read_file(fn_icnf_2023)
gdf = gdf.set_crs(epsg=crs_icnf_2023)
gdf_icnf_2023 = gdf.to_crs(epsg=crs_bdr300)

# tiles
gdf = gpd.read_file(fn_tiles)
gdf = gdf.set_crs(epsg=crs_tiles)
gdf_tiles = gdf.to_crs(epsg=crs_bdr300)

selected_columns=['label','data_0', 'data_1', 'data_2', 'data_3', 'classe_0','classe_1', 'classe_2', 'classe_3', 'tipo_1', 'tipo_2','geometry']

# reformat ICNF
gdf_icnf_2023['label']='incf_2023_tne'
gdf_icnf_2023['data_0']= gdf_icnf_2023['DH_Inicio']
gdf_icnf_2023['data_1']= gdf_icnf_2023['DH_Fim']
for col in ['data_2', 'data_3', 'classe_0','classe_1', 'classe_2', 'classe_3', 'tipo_1', 'tipo_2']:
    gdf_icnf_2023[col]=np.nan
gdf_icnf_2023=gdf_icnf_2023[selected_columns]

# select 'tipo_1'='Corte' in BDR300
gdf = gdf_bdr300.loc[gdf_bdr300["tipo_1"] == "Corte"]
gdf['label']='bdr_300_tne_corte'
gdf_bdr300=gdf[selected_columns]
#gdf.to_file('BDR300_corte.shp')

# label and select ~0.5 of features based on the oddity ofthe sum of digits in id_gleba
gdf_nvg_poly['label']='nvg_V02_poly_tne'
gdf=gdf_nvg_poly.copy()
gdf['id_gleba'] = gdf['id_gleba'].astype(str)
# Select rows where the sum of the first 5 digits is even
filtered_gdf = gdf[
    gdf['id_gleba']
    .str[:5]  # Extract first 5 characters
    .apply(lambda x: sum(int(d) for d in x)) % 2 == 0  # Sum digits and check if even
]
gdf_nvg_poly=filtered_gdf[selected_columns]

# select tile TNE
tne = gdf_tiles.loc[gdf_tiles["Name"] == "T29TNE"]
tne = tne[['geometry']] # drop all attributes

# select by location with sjoin
# NVG
# Perform spatial join with "within" predicate
within_features = gpd.sjoin(gdf_nvg_poly, tne, predicate="within", how="inner")
# Drop duplicate columns added by the join (e.g., 'index_right')
gdf_nvg_poly = within_features.drop(columns='index_right')
# ICNF
# Perform spatial join with "within" predicate
within_features = gpd.sjoin(gdf_icnf_2023, tne, predicate="within", how="inner")
# Drop duplicate columns added by the join (e.g., 'index_right')
gdf_icnf_2023 = within_features.drop(columns='index_right')

# concatenate
gdfs_list = [gdf_bdr300, gdf_icnf_2023, gdf_nvg_poly]
gdf = gpd.GeoDataFrame(pd.concat(gdfs_list, ignore_index=True)) 

# select features by area
gdf['area_m2'] = gdf.geometry.area
gdf = gdf.loc[(gdf['area_m2'] <= 1000000) & (gdf['area_m2'] >= 2500)]

# outras adaptações (from Sara's oberservations)
gdf['classe2018'] = np.nan
gdf['classe2019'] = np.nan
gdf['classe2020'] = np.nan
gdf['classe2021'] = np.nan
gdf['buffer_ID'] = range(1, len(gdf) + 1)
gdf['altera'] = np.where(gdf['data_0'].isna(), 'Sem Alteracao', 'Com Alteracao')
gdf['data_0']= gdf['data_0'].apply(formatar_data)
gdf['data_1']= gdf['data_1'].apply(formatar_data)

# display total area per label
print(gdf.groupby('label')['area_m2'].sum().reset_index())

gdf.to_file('BDR_MIX_TNE_V02.shp')
