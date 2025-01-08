import os
import sys
import csv
from osgeo import ogr, gdal, osr 
from pathlib import Path 
from console.console import _console 
from qgis.core import QgsProject, QgsExpression, QgsExpressionContext, QgsExpressionContextUtils, QgsVectorLayer, QgsField, QgsFields
from PyQt5.QtCore import QVariant
import collections
from PyQt5.QtWidgets import QAction
import processing
from shapely.geometry import Point, Polygon, MultiPolygon
from datetime import datetime, timedelta
import numpy as np
import geopandas as gpd
import pandas as pd
#import ee
import geemap
from dateutil.relativedelta import relativedelta
import glob
import matplotlib.pyplot as plt
from typing import Tuple, Optional



#project and data folders
project_name='database_navigator'
input_folder= 'input'
output_folder='output'
ndvi_folder = 'NDVI'
ccd_folder = 'ccd'

# Working directory:
# |----myfolder
#    |---- main.py
#    |---- my_functions_main.py
#    |---- input_folder
#         |---- NVG_proprios_2015_2023_clean.gpkg
#    |---- output_folder
#    |----ccd
#         |---- tiles
#              |---- df_ccd_tile29SNB.shp
#              |---- df_ccd_tile29SNB2.shp
#              |---- S2_T29SNB
#              |---- S2_T29SNB2

# Determine path to working directory ("my_folder")
# Find path to the directory where the script is 
script_path = Path(_console.console.tabEditorWidget.currentWidget().path)
my_folder=script_path.parent
# load my_functions.py
exec(Path(my_folder/ "my_functions_main.py").read_text())
exec(Path(my_folder/ "my_functions_aux.py").read_text())

# Layers name:
ln_gpkg='NVG_proprios_2015_2023_clean.gpkg'
ln_nvg = 'NVG_2015-2023_Proprios_clean'
ln_exploracao='Exploracao_NVG_2015-2023_Proprios_clean' ##nome mudado!! 
ln_silvicultura='Silvicultura_NVG_2015-2023_Proprios_clean' 

#paths to geopackage and layers
fn_gpkg = str(my_folder/input_folder/ln_gpkg)
fn_nvg = fn_gpkg + "|layername=" + ln_nvg #path to vector layer
fn_exploracao = fn_gpkg + "|layername=" + ln_exploracao #path to table exploracao
fn_silvicultura = fn_gpkg + "|layername=" + ln_silvicultura #path to table silvicultura

#read as gdf
gdf_nvg = gpd.read_file(fn_gpkg)
gdf_exp = gpd.read_file(fn_gpkg, layer=ln_exploracao)
df_exp = gdf_exp.drop(columns=gdf_exp.geometry.name)
gdf_silv = gpd.read_file(fn_gpkg, layer=ln_silvicultura)
df_silv = gdf_silv.drop(columns=gdf_silv.geometry.name)
df_exp.columns

# NORMALIZATION
#rename columns 
## table exploracao
df_exp = df_exp.rename(columns={'Id Gleba': 'id_gleba','Id Projeto': 'cod_un', 'Talhão': 'cod_talhao', 'Data Real': 'dt_real'}) #rename column
#table silvicultura
df_silv = df_silv.rename(columns={'Id Projeto': 'cod_un', 'Talhão': 'cod_talhao', 'Data Operação': 'dt_operacao', 'Desc. Atividade': 'desc_atividade'}) #rename column

#create normalized tables
## table nvg
# Split the 'ocupacao' column into two columns based on the '-' sign
# the output is two columns one with "ocupacao" and the other with "forma de plantacao"
gdf_nvg[['ocupacao', 'forma_plantacao']] = gdf_nvg['ocupacao'].str.split(' - ', n=1, expand=True)
fn_nvg_norm = str(my_folder / output_folder / 'nvg_norm.shp')
gdf_nvg.to_file(fn_nvg_norm, encoding='utf-8')
#gdf_nvg.to_csv(fn_nvg_norm, encoding='utf-8')
#print(gdf_nvg.columns)

## create table management_units
management_units = pd.DataFrame(gdf_nvg['cod_ug'].unique(), columns=['cod_ug'])
# Save the GeoDataFrame as a shapefile or another format
fn_management_units = str(my_folder / output_folder / 'management_units.csv')
management_units.to_csv(fn_management_units)

##add primary key to tables exploracao 
df_exp.insert(0, 'exploracao_ID', range(1, 1 + len(df_exp)))
exp = pd.DataFrame(df_exp)
exp.columns
# Save the GeoDataFrame as a shapefile or another format
fn_exp_norm = str(my_folder / output_folder / 'exploracao_norm.csv')
exp.to_csv(fn_exp_norm)

## add primary key to table silvicultura
df_silv.insert(0, 'silvicultura_ID', range(1, 1 + len(df_silv)))
# Save the GeoDataFrame to a new shapefile
fn_silv_norm = str(my_folder / output_folder / 'silvicultura_norm.csv')
df_silv.to_csv(fn_silv_norm)


######## EXPLORATORY DATA ANALYSIS


# Determine path to working directory ("my_folder")
script_path = Path(_console.console.tabEditorWidget.currentWidget().path)
my_folder=script_path.parent
# load my_functions.py
exec(Path(my_folder/ "my_functions_visualization.py").read_text())

# Layers name:
ln_nvg = 'nvg_norm'
#paths to geopackage and layers
fn_nvg = str(my_folder/output_folder/(ln_nvg + '.shp'))
gdf=gpd.read_file(fn_nvg, encoding='utf-8')

### Renaming 'ocupacao' values based on conditions (vegetation classes)
gdf = replace_ocupacao(gdf)
#print(gdf['ocupacao'])

#calculate nr of management units
nr_un_gestao = len(gdf['cod_ug'].unique())
# Calculate the number of features in the database
nr_talhoes = len(gdf)
#calculate nr of talhoes per ocupacao
nr_talhoes_ocupacao = gdf.groupby('ocupacao').size()
#calculate percentages
percentage_per_ocupacao = (nr_talhoes_ocupacao/nr_talhoes)*100
print("Number of Unidade de Gestao:", nr_un_gestao)
print("Number of Talhoes:", nr_talhoes)
print("Number of Talhoes por ocupacao:", nr_talhoes_ocupacao)
print("Percentage of Talhoes per ocupacao:", percentage_per_ocupacao)



# Nr of talhoes per ocupacao when area > 0.5 hectares
area_data = gdf[gdf['area_ha']> 0.5]
ocupacao_data = area_data.groupby('ocupacao').size()
print("Number of features per ocupacao when area_ha > 0.5:")
print(ocupacao_data)

total_area_per_class = area_data.groupby('ocupacao')['area_ha'].sum()
total_area = area_data['area_ha'].sum()

print("Total area of each class:")
print(total_area_per_class)
print(total_area)

#count nr of multipart and singlepart polygons
multipart_count, singlepart_count = count_multi_and_single_part(fn_gpkg, ln_nvg)

### count nr of multipart and singlepart polygons per type of ocupacao
# Apply the function to count parts for each feature
gdf['num_parts'] = gdf['geometry'].apply(count_parts)
# Group by 'ocupacao' and calculate statistics for each group
ocupacao_stats = gdf.groupby('ocupacao')['num_parts'].agg(['mean', 'min', 'max','sum'])
# Print the statistics
print("Statistics per ocupacao:")
print(ocupacao_stats)

###Stats for multipart polygons by ocupacao 
# Group by 'ocupacao' and calculate statistics for each group excluding features with only one part
ocupacao_stats_excluding_one_part = gdf.groupby('ocupacao').apply(calculate_min_nr_parts)
# Print the statistics
print("Statistics per ocupacao (excluding features with only one part):")
print(ocupacao_stats_excluding_one_part)

## calculate the number of singlepart features
# Group by 'ocupacao' and calculate the number of features with only one part for each group
features_with_one_part_by_ocupacao = gdf.groupby('ocupacao').apply(count_features_with_one_part)
# Print the result
print("Number of features with only one part by ocupacao:")
print(features_with_one_part_by_ocupacao)

## check id_gleba for parcel with max number of sub-parcels
id_gleba_max_parts = id_gleba_for_max_parts(gdf)
# Print the result
print("id_gleba for the parcel with the highest number of parts:", id_gleba_max_parts)

### STATS AREAS
#Calculate te total area with geopandas
## use the sum of the attribute column "area_ha"
total_area = gdf['area_ha'].sum()
total_area_rounded = round(total_area, 2)
print(f"Total area: {total_area_rounded} hectares")

# Area distribution by ocupacao
area_distribution = gdf.groupby('ocupacao')['area_ha'].sum()
print(area_distribution)
## percentagens
percentage_distribution = (area_distribution/total_area)*100
percentage_distribution_rounded = round(percentage_distribution, 2)
# Print the percentage distribution
print("Percentage distribution:")
print(percentage_distribution_rounded)


ocupacao_stats = gdf.groupby('ocupacao')['area_ha'].agg(['mean', 'min', 'max'])
# Print the statistics
print("Statistics per ocupacao:")
print(ocupacao_stats)





# BASE DE DADOS CRONOLOGICA
## read as geodf
gdf_nvg = gpd.read_file(fn_nvg_norm)
gdf_exp = gpd.read_file(fn_exp_norm)
gdf_silv = gpd.read_file(fn_silv_norm)


# create table NVG chronologically sorted
nvg_df = create_nvg_table(gdf_nvg, 'id_gleba', 'dt_referen', 'dt_plant', 'REF', 'PLANT')
#print(result_df['date_1'])
#print(nvg_df)
# Save the GeoDataFrame as a shapefile or another format
fn_nvg = str(my_folder / output_folder / 'nvg.csv')
nvg_df.to_csv(fn_nvg)

# create table exploraço chronologically sorted
pivot_table_sorted = create_pivot_table(gdf_exp, 'dt_real', 'Atividade', 'id_gleba')
#print(pivot_table_sorted)
# Save the GeoDataFrame as a shapefile or another format
fn_exp = str(my_folder / output_folder / 'pivot_table_exp.csv')
pivot_table_sorted.to_csv(fn_exp)

# create table silvicultura chronologically sorted
pivot_table_sorted = create_pivot_table(gdf_silv, 'dt_operacao', 'desc_atividade', 'id_gleba')
#print(pivot_table_sorted)
# Save the GeoDataFrame as a shapefile or another format
fn_silv = str(my_folder / output_folder / 'pivot_table_silv.csv')
pivot_table_sorted.to_csv(fn_silv)

# MERGE THE 3 TABLES
## read as geodf
df_nvg = pd.read_csv(fn_nvg)
df_exp = pd.read_csv(fn_exp)
df_silv = pd.read_csv(fn_silv)

result_df = merge_and_transform_dfs(df_nvg, df_exp, df_silv, 'id_gleba', 'inner')
#print(result_df)
# Save the GeoDataFrame as a shapefile or another format
fn_merged_df = str(my_folder / output_folder / 'merged_df.csv')
result_df.to_csv(fn_merged_df)


# CREATE THE FINAL DF
## read final_Df as geodf
df_all = pd.read_csv(fn_merged_df)
# create a lists with dates and activities
result_df_list = process_dataframe(df_all, 'id_gleba', 'data', 'actividade')
#print(result_df_list.columns) # should have 3 columns - id_gleba, datas, atividades
# Filter the DataFrame
filtered_df = result_df_list[result_df_list['id_gleba'] == '51001-T025_EG']

# fn_list = str(my_folder / output_folder / 'df_list.csv')
# result_df_list.to_csv(fn_list)

# sort pairs of data/atividades 
final_df = create_final_dataframe(result_df_list, 'id_gleba')
#print(final_df.columns)
fn_final = str(my_folder / output_folder / 'df_final.csv')
final_df.to_csv(fn_final)

# sort columns of final_df
df_final = pd.read_csv(fn_final)
sorted_df = sort_df(df_final)
# Save the GeoDataFrame as a shapefile or another format
fn_df_sorted = str(my_folder / output_folder / 'final_df_sorted.csv')
sorted_df.to_csv(fn_df_sorted)
# clean activity columns from white spaces and special characters
sorted_df = clean_atividade_columns(sorted_df)
# Save the GeoDataFrame as a shapefile or another format
fn_df_sorted = str(my_folder / output_folder / 'final_df_sorted_no_spaces.csv')
sorted_df.to_csv(fn_df_sorted)




### TILE 29TNE  - entregavel


#NEW FOLDER
new_folder_name = 'tile29tne'
new_folder_path = my_folder / new_folder_name
new_folder_path.mkdir(parents=True, exist_ok=True)
tile29_folder = 'tile29tne'

# csv com mediana do NDVI criado no VSCode
ln_csv_ndvi = 'Median_NDVI_Per_Polygon_All_Glebas.csv'
fn_csv_ndvi = str(my_folder/ndvi_folder/ln_csv_ndvi)
#shapefile com o Tile 29TNE
ln_tile = 'tile_29TNE.shp'
fn_tile = str(my_folder/tile29_folder/ln_tile)
#final_df_sorted
ln_df_sorted = 'final_df_sorted_no_spaces.csv'
fn_df_sorted = str(my_folder/output_folder/ln_df_sorted)
df_sorted = pd.read_csv(fn_df_sorted)

df_ndvi = pd.read_csv(fn_csv_ndvi)

#filter out id_glebas where all the median values are null 
filtered_df_ndvi = df_ndvi.groupby('id_gleba').filter(lambda x: not x['median'].isna().all())
fn_fdf = str(my_folder / ndvi_folder / 'Median_NDVI_filtered.csv')
filtered_df_ndvi.to_csv(fn_fdf)

id_gleba_list_f = extract_unique_id_gleba_from_nvg(fn_fdf, 'id')
id_gleba_list = extract_unique_id_gleba_from_nvg(fn_csv_ndvi, 'id')
# print(len(id_gleba_list_f))
# print(len(id_gleba_list))


# CRIAR UM CSV COM ID_GLEBA, START/END DATE, NR FEATURES, NR CORTES, AREA E AREA/NR CORTES para todas as parcelas
gdf_nvg = gpd.read_file(fn_gpkg)
id_gleba = []
num_features = []

for index, row in gdf_nvg.iterrows():
    id_value = row['id_gleba']
    geometry = row.geometry
    if isinstance(geometry, MultiPolygon):
        count = len(geometry.geoms)
    elif isinstance(geometry, Polygon):
        count = 1
    else:
        count = 0
    
    id_gleba.append(id_value)
    num_features.append(count)

new_data = {'id_gleba': id_gleba, 'num_features': num_features}
new_gdf = gpd.GeoDataFrame(new_data)
new_gdf.head()

# Add new columns to the gdf
new_gdf['first_start_date'] = pd.NaT
new_gdf['first_end_date'] = pd.NaT

for index, row in new_gdf.iterrows():
    id_gleba = row['id_gleba']
    date_pairs = find_date_pairs(df_sorted, id_gleba)
    
    if date_pairs:
        first_start_date, first_end_date = date_pairs[0]
        new_gdf.at[index, 'first_start_date'] = first_start_date
        new_gdf.at[index, 'first_end_date'] = first_end_date

new_gdf['start_date'] = pd.NaT
new_gdf['end_date'] = pd.NaT

for index, row in new_gdf.iterrows():
    id_gleba = row['id_gleba']
    date_pairs = find_date_pairs(df_sorted, id_gleba)
    
    if date_pairs:
        first_start_date, first_end_date = date_pairs[0]
        new_start_date, new_end_date = start_and_end_dates_two_months(first_start_date, first_end_date)
        new_gdf.at[index, 'start_date'] = new_start_date
        new_gdf.at[index, 'end_date'] = new_end_date

# convert to datetime
new_gdf['first_start_date'] = pd.to_datetime(new_gdf['first_start_date'], errors='coerce')
new_gdf['first_end_date'] = pd.to_datetime(new_gdf['first_end_date'], errors='coerce')
# calculate the difference in another column
new_gdf['date_difference_days'] = (new_gdf['first_end_date'] - new_gdf['first_start_date']).dt.days
#save 
# fn_with_dates = str(my_folder / output_folder / 'nvg_with_dates.csv')
# new_gdf.to_csv(fn_with_dates)


# count the number of clear cuts per id_gleba
df_sorted['nr_clear_cuts'] = df_sorted.apply(count_corte_activities, axis=1)

#new_gdf.columns
# merge the df with gdf based on 'id_gleba'
nvg_dates_cuts = pd.merge(df_sorted[['id_gleba', 'nr_clear_cuts']], new_gdf[['id_gleba', 'num_features','first_start_date','first_end_date','start_date', 'end_date', 'date_difference_days']], on='id_gleba', how='left')

# save
fn_nvg_dates_and_cuts = str(my_folder / output_folder / 'nvg_with_dates_and_cuts.csv')
nvg_dates_cuts.to_csv(fn_nvg_dates_and_cuts)

#join total area of id_gleba
result = join_attribute_to_layer(fn_nvg_dates_and_cuts, 'id_gleba', fn_nvg_norm, 'id_gleba', 'area_ha')
#save
ln_nvg_dates_cuts_and_area = 'nvg_with_dates_cuts_and_area.csv'
fn_nvg_dates_cuts_and_area = str(my_folder/output_folder/ln_nvg_dates_cuts_and_area)
result.selectAll()
result = processing.run("native:saveselectedfeatures", {'INPUT':result, 'OUTPUT':fn_nvg_dates_cuts_and_area})

# calcular a area de cada talhao por numero de cortes
df_dates_areas = pd.read_csv(fn_nvg_dates_cuts_and_area)
df_dates_areas['area_per_cuts'] = df_dates_areas['area_ha'] / df_dates_areas['nr_clear_cuts']
fn_csv_dates = str(my_folder / output_folder / 'nvg_with_dates_cuts_and_area.csv')
df_dates_areas.to_csv(fn_csv_dates)


# select rows where 'first_start_date' is not null
filtered_df = df_dates_areas[df_dates_areas['start_date'].notnull()]
#save
ln_nvg_dates_filtered = 'nvg_with_dates_filtered.csv'
fn_nvg_dates_filtered = str(my_folder/output_folder/ln_nvg_dates_filtered)
filtered_df.to_csv(fn_nvg_dates_filtered)



## TILE 29TNE

fn_nvg_singlepart_st = str(my_folder/output_folder/'nvg_singlepart.shp')

#extract by location all sub-parcels within tile 29TNE (id_gleba and id)
result = extract_by_location(fn_nvg_singlepart_st,fn_tile)
QgsProject().instance().addMapLayer(result)

# select all features of layer to export
ln_glebas_tile = 'id_glebas_tile.shp'
fn_glebas_tile = str(my_folder/tile29_folder/ln_glebas_tile)
result.selectAll()
# export layer
result = processing.run("native:saveselectedfeatures", {'INPUT':result, 'OUTPUT':fn_glebas_tile})

# create list of id_gleba of tile 29tne
id_gleba_list = extract_unique_id_gleba_from_nvg(fn_glebas_tile, 'id_gleba')
id_glebas_df_to_save = pd.DataFrame(id_gleba_list, columns=['id_gleba'])
fn = str(my_folder/tile29_folder/'id_gleba_list_tile29.csv')
id_glebas_df_to_save.to_csv(fn)
#create list of id of tile 29tne
id_list = extract_unique_id_gleba_from_nvg(fn_glebas_tile, 'id')
id_df_to_save = pd.DataFrame(id_list, columns=['id_gleba'])
fn = str(my_folder/tile29_folder/'id_list_tile29.csv')
id_df_to_save.to_csv(fn)
# find id_glebas that are not present in df_median_ndvi
# list of id_gleba from df_median_ndvi
id_gleba_ndvi_set = set(df_median_ndvi['id_gleba'].unique())
missing_id_glebas = [id_gleba for id_gleba in id_gleba_list if id_gleba not in id_gleba_ndvi_set]
missing_id_glebas_df_to_save = pd.DataFrame(missing_id_glebas, columns=['id_gleba'])
#save
fn = str(my_folder/tile29_folder/'missing_id_glebas_list_tile29.csv')
missing_id_glebas_df_to_save.to_csv(fn)


### VISUALIZAR CADA TALHAO COM AS DATAS DE CORTE ESTIMADAS
#CREATED PIVOT TABLES FOR TILE 29 IN VSCODE
###
id_gleba = '50445-T001_EG'
ln_talhao_singlepart = 'nvg_singlepart_' + id_gleba + '.shp'
fn_talhao_singlepart = str(my_folder / output_folder / ln_talhao_singlepart)
vscode_folder = 's2change'
fn_pivot_table = str(my_folder / vscode_folder / 'tile_29' /'nvg_dataset'/'all_pivot_tables'/(f'pivot_table_{id_gleba}.csv'))

## Label the parcel
# join 'estimated_date' to shapefile and add layer to the project
result = join_attribute_to_layer(fn_talhao_singlepart, 'id', fn_pivot_table, 'id', 'estimated_date')
result.setName('talhao_'+ id_gleba)
QgsProject().instance().addMapLayer(result)
 

# set labels according to estimated clear-cut dates 
layer = iface.activeLayer()
field_name = 'estimated_date'
field_index = layer.fields().indexFromName(field_name)
unique_values = sorted(layer.uniqueValues(field_index))

category_list = []
for value in unique_values:
    symbol = QgsSymbol.defaultSymbol(layer.geometryType())
    category = QgsRendererCategory(value, symbol, str(value))
    category_list.append(category)

# color ramp
ramp_name = 'Greens'
default_style = QgsStyle().defaultStyle()
color_ramp = default_style.colorRamp(ramp_name)
renderer = QgsCategorizedSymbolRenderer(field_name, category_list)
renderer.updateColorRamp(color_ramp)
layer.setRenderer(renderer)
layer.triggerRepaint()



### LIMITACOES

# JOIN ALL PIVOT TABLES 
fn_folder = str(my_folder / vscode_folder / 'tile_29' /'nvg_dataset'/'all_pivot_tables')
fn_nvg_pt = join_pivot_tables(fn_folder)



### CALCULAR DATA MEDIA DE CORTE POR ID GLEBA 
##com base na tabela exploracao

df_exp = pd.read_csv(fn_exp_norm)
# create a df with all clear cut dates per id_gleba
clear_cut_dates = {}
grouped = df_exp.groupby('id_gleba')

# Iterate over each group
for gleba_id, group in grouped:
    # Filter the rows where 'Atividade' starts with 'CORTE'
    corte_rows = group[group['Atividade'].str.startswith('CORTE', na=False)]
    
    # Extract the 'Data Real' column for these rows
    dates = corte_rows['dt_real'].tolist()
    
    # Store the dates in the dictionary
    clear_cut_dates[gleba_id] = dates

df_to_save = pd.DataFrame.from_dict(clear_cut_dates, orient='index')
df_to_save.reset_index(inplace=True)
df_to_save.columns = ['id_gleba'] + [f'date_{i}' for i in range(1, len(max(clear_cut_dates.values(), key=len)) + 1)]

fn_cut_dates = str(my_folder / output_folder / 'id_glebas_clear_cut_dates.csv')
df_to_save.to_csv(fn_cut_dates, index=False)

# calculate the mean month of all clear cuts per id_gleba
df=pd.read_csv(fn_cut_dates)

date_columns = [col for col in df.columns if col.startswith('date')]
date_columns_dt = pd.DataFrame()
for col in date_columns:
    date_columns_dt[col] = pd.to_datetime(df[col], dayfirst=True)

date_columns_dt['mean_date'] = date_columns_dt.mean(axis=1)
# month from the mean_date
date_columns_dt['mean_month'] = date_columns_dt['mean_date'].dt.month

# Concatenate the original DataFrame df['id_gleba'] and date_columns_dt along the columns axis
merged_df = pd.concat([df['id_gleba'], date_columns_dt], axis=1)

# Save the merged DataFrame to a CSV file
fn = str(my_folder/output_folder/'id_glebas_clear_cut_dates.csv')
merged_df.to_csv(fn, index=False)



# calcular a media da data de corte para cada id_gleba 
# criar graph com media da data de corte / nr de imagens S2
df_mean = pd.read_csv(fn)
df_nvg = pd.read_csv(fn_nvg_pt)

df_mean_extracted = df_mean[['id_gleba', 'mean_month']]
df_nvg_selected = df_nvg[['id_gleba', 'nr_s2_dates', 'nr_clear_cuts']]

join_df = pd.merge(df_mean_extracted, df_nvg_selected, on='id_gleba')
join_df.drop_duplicates(inplace=True)
# fn_s2_mean = str(my_folder/output_folder/'id_gleba_s2_mean_month.csv')
# join_df.to_csv(fn_s2_mean, index=False)

# Create a scatterplot Nr S2 dates vs Nr clear cuts
create_scatterplot(
    data=join_df, 
    x_col='nr_clear_cuts', 
    y_col='nr_s2_dates', 
    color_col='mean_month', 
    x_label='Nr Clear Cuts', 
    y_label='Nr S2 Dates', 
    title='Scatterplot of Nr S2 Dates vs. Nr Clear Cuts', 
    x_lim=(0, 20), 
    y_lim=(0, 60), 
    colorbar_label='Mean Month', 
    cmap='hsv', 
    alpha=0.7
)



### CALCULAR AREA DOS SUB-TALHOES
ln_nvg_singlepart = 'nvg_singlepart.shp'
fn_nvg_singlepart = str(my_folder / output_folder / ln_nvg_singlepart)
fn_output_layer = str(my_folder / output_folder / 'nvg_singlepart_area.shp')

# Call the function to process the shapefile
output_file = add_area_field_to_layer(
    input_shapefile=fn_nvg_singlepart,
    output_shapefile=fn_output_layer,
    field_name='area_ha_subtalhao',
    field_type=1,
    field_length=5,
    field_precision=2
)


# ln_nvg_singlepart = 'nvg_singlepart.shp'
# fn_nvg_singlepart = str(my_folder/output_folder/ln_nvg_singlepart)

# layer = QgsVectorLayer(fn_nvg_singlepart, 'My Layer', 'ogr')

# area_singlepart = processing.run("native:addfieldtoattributestable", 
#  {'INPUT':layer,
#  'FIELD_NAME':'area_ha_subtalhao',
#  'FIELD_TYPE':1,
#  'FIELD_LENGTH':5,
#  'FIELD_PRECISION':2,
#  'FIELD_ALIAS':'',
#  'FIELD_COMMENT':'',
#  'OUTPUT':'TEMPORARY_OUTPUT'})

# output_layer = area_singlepart['OUTPUT']

# # Ensure the output layer is added to the QGIS project
# if output_layer not in QgsProject.instance().mapLayers().values():
#     QgsProject.instance().addMapLayer(output_layer)

# # Calculate the area of the polygons and update the new field
# if output_layer.isEditable() or output_layer.startEditing():
#     for feature in output_layer.getFeatures():
#         geom = feature.geometry()
#         area = geom.area() / 10000 
#         feature['area_ha_subtalhao'] = area
#         output_layer.updateFeature(feature)
    
#     # Commit the changes
#     output_layer.commitChanges()

# fn_output_layer = str(my_folder/output_folder/'nvg_singlepart_area.shp')
# save = QgsVectorFileWriter.writeAsVectorFormat(output_layer, fn_output_layer, "UTF-8", output_layer.crs(), "ESRI Shapefile")


### CRIAR TABELAS DE ATRIBUTO AO NIVEL DO TALHAO E SUB-TALHAO

#NEW FOLDER
new_folder_name = 'entregavel2_2'
new_folder_path = my_folder / new_folder_name
new_folder_path.mkdir(parents=True, exist_ok=True)
entregavel_folder = 'entregavel2_2'

# AO NIVEL DO TALHAO
df_nvg_pt = pd.read_csv(fn_nvg_pt)
df_nvg_select = df_nvg_pt[['id_gleba', 'nr_s2_dates',
       'nr_clear_cuts', 'num_features',
       'first_start_date', 'first_end_date',
       'date_difference_days', 'area_ha', 'area_per_cuts']]
df_nvg_select_unique = df_nvg_select.drop_duplicates()
#add a column
df_nvg_select_unique['nr_s2_per_cuts'] = (df_nvg_select_unique['nr_s2_dates'] / df_nvg_select_unique['nr_clear_cuts']).astype(int)

fn_talhao = str(my_folder/entregavel_folder/'nvg_talhao.csv')
df_nvg_select_unique.to_csv(fn_talhao, index=False)



id_gleba_list_ndvi = extract_unique_id_gleba_from_nvg(fn_csv_ndvi, 'id_gleba')
print(len(id_gleba_list))
id_gleba_list = extract_unique_id_gleba_from_nvg(fn_nvg_pt, 'id_gleba')
print(len(id_gleba_list))

# Find the missing id_gleba
missing_id_gleba = list(set(id_gleba_list_ndvi) - set(id_gleba_list))
print("Missing id_gleba:", missing_id_gleba)
print("Number of missing id_gleba:", len(missing_id_gleba))

#### nr features vs nr clear cuts
df_talhao = pd.read_csv(fn_talhao)
df_talhao.columns
# Nr total
num_parcels = df_talhao.shape[0]
#nr features maiores do que nr cortes
nr_parcel_w_more_sub_parcels = df_talhao[df_talhao['num_features'] > df_talhao['nr_clear_cuts']]
#nr de features igual ao nr de cortes
nr_parcel_w_equal_feature_subt = df_talhao[df_talhao['num_features'] == df_talhao['nr_clear_cuts']]
# nr de features menor do que nr cortes
nr_parcel_w_less_sub_parcels = df_talhao[df_talhao['num_features'] < df_talhao['nr_clear_cuts']]

print("Number of parcels:", num_parcels)
print("Nr of features > Nr Clear Cuts:",len(nr_parcel_w_more_sub_parcels))
print("Nr of features = Nr Clear Cuts:",len(nr_parcel_w_equal_feature_subt))
print("Nr of features < Nr Clear Cuts:",len(nr_parcel_w_less_sub_parcels))


# AO NIVEL DO SUB TALHAO
df_nvg_select_id = df_nvg_pt[['id_gleba', 'id',
       'first_start_date', 'first_end_date', 'date_of_biggest_drop','estimated_date', 'nr_empty_cells',
       'nr_non_empty_cells', 'nr_s2_dates']]

df_nvg_select_id = df_nvg_select_id.rename(columns={'estimated_date': 'first_estimated_date'})

fn_sub_talhao = str(my_folder/entregavel_folder/'nvg_sub_talhao.csv')
df_nvg_select_id.to_csv(fn_sub_talhao, index=False)


# Load the output layer DataFrame
df_output_layer = gpd.read_file(fn_output_layer)
df_output_layer.columns
# Select only the 'id' and 'area_ha_sub' columns from the output layer
df_output_layer_select = df_output_layer[['id', 'area_ha_su']]
# Perform the join operation on the 'id' field
result = pd.merge(df_nvg_select_id, df_output_layer_select, on='id', how='inner')
# save the result
result.to_csv(fn_sub_talhao)





### QUANDO NAO HA DATA DE CORTE ESTIMADA ao nivel do sub talhao
df_subtalhao = pd.read_csv(fn_sub_talhao)
df_subtalhao.columns

# nr sub talhoes com 1 data de corte e sem data estimada
one_cut = df_subtalhao[
    df_subtalhao['date_of_biggest_drop'].notna() &
    (df_subtalhao['first_start_date'] == df_subtalhao['first_end_date']) &
    df_subtalhao['first_estimated_date'].isna()
]
print(len(one_cut))
# total area
total_area_one_cut = one_cut['area_ha_su'].sum()

# apply function to update estimated date
df_subtalhao = df_subtalhao.apply(update_first_estimated_date, axis=1)
df_subtalhao.to_csv(fn_sub_talhao, index=False)


### criar um shp com subtalhoes e info de datas, cortes e S2 images
ln_nvg_singlepart = 'nvg_singlepart.shp'
fn_nvg_singlepart = str(my_folder/output_folder/ln_nvg_singlepart)

result = join_attribute_to_layer(fn_nvg_singlepart, 'id', fn_sub_talhao, 'id', ['first_start_date','first_end_date','date_of_biggest_drop','first_estimated_date','nr_empty_cells','nr_non_empty_cells','nr_s2_dates','area_ha_su'])
result.setName('singlepart com sub-talhoes')
QgsProject().instance().addMapLayer(result)

result_w_deleted_fields = processing.run("native:deletecolumn", {'INPUT':result,
 'COLUMN':['cod_ug','cod_talhao','ciclo','rotacao','dt_referen','dt_plant','idade_ref','idade_plan','area_ha','forma_plan'],
 'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
result_w_deleted_fields.setName('singlepart com sub-talhoes_deleted_fields')
QgsProject().instance().addMapLayer(result_w_deleted_fields)
result_w_deleted_fields.selectAll()
# export layer
ln_nvg_singlepart_st = 'nvg_singlepart_sub_talhao.gpkg'
fn_nvg_singlepart_st = str(my_folder/output_folder/ln_nvg_singlepart_st)

result__shp = processing.run("native:saveselectedfeatures", {'INPUT':result_w_deleted_fields, 'OUTPUT':fn_nvg_singlepart_st})
result_w_deleted_fields.removeSelection()



## CREATE SHP TILE 29 AND SUB TALHOES

#extract by location all sub-parcels within tile 29TNE (id_gleba and id)
result = extract_by_location(fn_nvg_singlepart, fn_tile)
QgsProject().instance().addMapLayer(result)

result_w_deleted_fields_tile29 = processing.run("native:deletecolumn", {'INPUT':result,
 'COLUMN':['cod_ug','cod_talhao','ciclo','rotacao','dt_referen','dt_plant','idade_ref','idade_plan','area_ha','forma_plan','ocupacao'],
 'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
QgsProject().instance().addMapLayer(result_w_deleted_fields_tile29)

# export layer
ln_nvg_singlepart_st_tile = 'nvg_singlepart_tile29tne.shp'
fn_nvg_singlepart_st_tile = str(my_folder/output_folder/ln_nvg_singlepart_st_tile)
result_w_deleted_fields_tile29.selectAll()
result__csv = processing.run("native:saveselectedfeatures", {'INPUT':result_w_deleted_fields_tile29, 'OUTPUT':fn_nvg_singlepart_st_tile})
result_w_deleted_fields_tile29.removeSelection()

## this was exported as a shapefile and manually added to the geopackage

#### ANALISE DE AREAS E PERCENTAGENS DE SUB TALHOES
# Number of sub-parcels
num_subparcels = df_subtalhao.shape[0]
df_subtalhao.columns
print("Number of sub-parcels:", num_subparcels)

# id_glebas
unique_counts = df_subtalhao.groupby('id_gleba').nunique()
print(unique_counts)

# Total area
total_area = df_subtalhao['area_ha_su'].sum()
print("Total area (ha):", total_area)

# Number and percentage of sub-parcels without an estimated date
empty_estimated_dates = df_subtalhao['first_estimated_date'].isna().sum()
empty_estimated_dates_percentage = (empty_estimated_dates * 100) / num_subparcels
empty_estimated_dates_area = df_subtalhao[df_subtalhao['first_estimated_date'].isna()]['area_ha_su'].sum()
empty_estimated_dates_area_percentage = (empty_estimated_dates_area * 100) / total_area
print("Number of empty estimated dates:", empty_estimated_dates)
print("Percentage of empty estimated dates:", empty_estimated_dates_percentage)
print("Area of empty estimated dates (ha):", empty_estimated_dates_area)
print("Percentage area of empty estimated dates:", empty_estimated_dates_area_percentage)

# Number and percentage of sub-parcels with an estimated date
estimated_dates = df_subtalhao['first_estimated_date'].notna().sum()
estimated_dates_percentage = (estimated_dates * 100) / num_subparcels
estimated_dates_area = df_subtalhao[df_subtalhao['first_estimated_date'].notna()]['area_ha_su'].sum()
estimated_dates_area_percentage = (estimated_dates_area * 100) / total_area
print("Number of sub-parcels with estimated dates:", estimated_dates)
print("Percentage of sub-parcels with estimated dates:", estimated_dates_percentage)
print("Area of sub-parcels with estimated dates (ha):", estimated_dates_area)
print("Percentage area of sub-parcels with estimated dates:", estimated_dates_area_percentage)

# Number and percentage of sub-parcels without the date of biggest drop
empty_drop_dates = df_subtalhao['date_of_biggest_drop'].isna().sum()
empty_drop_dates_percentage = (empty_drop_dates * 100) / empty_estimated_dates
empty_drop_dates_area = df_subtalhao[df_subtalhao['date_of_biggest_drop'].isna()]['area_ha_su'].sum()
empty_drop_dates_area_percentage = (empty_drop_dates_area * 100) / empty_estimated_dates_area
print("Number of empty NDVI drop dates:", empty_drop_dates)
print("Percentage of empty NDVI drop dates from total estimated dates:", empty_drop_dates_percentage)
print("Area of empty NDVI drop dates (ha):", empty_drop_dates_area)
print("Percentage area of empty NDVI drop dates from total estimated dates area:", empty_drop_dates_area_percentage)

# Number and percentage of sub-parcels with the date of biggest drop
drop_dates = empty_estimated_dates - empty_drop_dates
drop_dates_percentage = (drop_dates * 100) / empty_estimated_dates
drop_dates_area = empty_estimated_dates_area - empty_drop_dates_area
drop_dates_area_percentage = (drop_dates_area * 100) / empty_estimated_dates_area
print("Number of non-empty NDVI drop dates:", drop_dates)
print("Percentage of non-empty NDVI drop dates from total estimated dates:", drop_dates_percentage)
print("Area of non-empty NDVI drop dates (ha):", drop_dates_area)
print("Percentage area of non-empty NDVI drop dates from total estimated dates area:", drop_dates_area_percentage)

# Number of sub-parcels with the date of biggest drop before the first start date
df_subtalhao['date_of_biggest_drop'] = pd.to_datetime(df_subtalhao['date_of_biggest_drop'])
df_subtalhao['first_start_date'] = pd.to_datetime(df_subtalhao['first_start_date'])
drop_before_first_cut = df_subtalhao[df_subtalhao['date_of_biggest_drop'] <= df_subtalhao['first_start_date']]
drop_before_first_cut_area = drop_before_first_cut['area_ha_su'].sum()
print("Number of sub-parcels with NDVI drop date before first start date:", len(drop_before_first_cut))
print("Area of sub-parcels with NDVI drop date before first start date (ha):", drop_before_first_cut_area)

# Number and percentage of sub-parcels without the date of biggest drop but no empty cells
empty_drop_dates_no_empty_cells = df_subtalhao[df_subtalhao['date_of_biggest_drop'].isna() & (df_subtalhao['nr_empty_cells'] == 0)]
empty_drop_dates_no_empty_cells_area = empty_drop_dates_no_empty_cells['area_ha_su'].sum()
percent_no_empty_cells = (len(empty_drop_dates_no_empty_cells) * 100) / empty_drop_dates
percent_no_empty_cells_area = (empty_drop_dates_no_empty_cells_area * 100) / empty_drop_dates_area
print("Number of sub-parcels with empty NDVI drop dates and no empty cells:", len(empty_drop_dates_no_empty_cells))
print("Area of sub-parcels with empty NDVI drop dates and no empty cells (ha):", empty_drop_dates_no_empty_cells_area)
print("Percentage of sub-parcels with empty NDVI drop dates and no empty cells from total empty drop dates:", percent_no_empty_cells)
print("Percentage area of sub-parcels with empty NDVI drop dates and no empty cells from total empty drop dates area:", percent_no_empty_cells_area)

# Number and percentage of sub-parcels without the date of biggest drop due to clouds
even_condition = (df_subtalhao['nr_s2_dates'] % 2 == 0) & (df_subtalhao['nr_empty_cells'] >= df_subtalhao['nr_non_empty_cells']) #check if the nr of S2 is even and if the nr of empty cells is greater than the nr of non empty
odd_condition = (df_subtalhao['nr_s2_dates'] % 2 != 0) & (df_subtalhao['nr_empty_cells'] >= 0.4 * df_subtalhao['nr_s2_dates'])
empty_drop_dates_clouds = df_subtalhao[df_subtalhao['date_of_biggest_drop'].isna() & (even_condition | odd_condition)]
empty_drop_dates_clouds_area = empty_drop_dates_clouds['area_ha_su'].sum()
percentage_clouds = (len(empty_drop_dates_clouds) * 100) / empty_drop_dates
percentage_clouds_area = (empty_drop_dates_clouds_area * 100) / empty_drop_dates_area
print("Number of sub-parcels without NDVI drop dates due to clouds:", len(empty_drop_dates_clouds))
print("Area of sub-parcels without NDVI drop dates due to clouds (ha):", empty_drop_dates_clouds_area)
print("Percentage of sub-parcels without NDVI drop dates due to clouds from total empty drop dates:", percentage_clouds)
print("Percentage area of sub-parcels without NDVI drop dates due to clouds from total empty drop dates area:", percentage_clouds_area)

# nr sub talhoes que nao tem nuvens nas imagens S2
nr_empty_cells = df_subtalhao[df_subtalhao['nr_empty_cells'] == 0]
print(len(nr_empty_cells))

### S2 COM NDVI E SEM NDVI
#talhao
nr_total_s2 = df_talhao['nr_s2_dates'].sum()

#subtalhao
nr_total_s2 = df_subtalhao['nr_s2_dates'].sum()
nr_total_empty_cells = df_subtalhao['nr_empty_cells'].sum()
perc_empty_cells = (nr_total_empty_cells*100)/nr_total_s2
nr_total_non_empty_cells = df_subtalhao['nr_non_empty_cells'].sum()
perc_non_empty_cells = (nr_total_non_empty_cells*100)/nr_total_s2
perc_subparcel_no_empty_cells = (len(nr_empty_cells)*100)/nr_total_s2

print("Total number of S2 images:", nr_total_s2)
print("Total number of S2 images with no NDVI values:", nr_total_empty_cells)
print("Percentage of S2 images with no NDVI values:", perc_empty_cells)
print("Total number of S2 images with NDVI values:", nr_total_non_empty_cells)
print("Percentage of S2 images with NDVI values:", perc_non_empty_cells)
print("Total number of sub-parcels without empty NDVI values:", len(nr_empty_cells))
print("Percentage of S2 images with NDVI values:", perc_subparcel_no_empty_cells)


# mean_month
df_subtalhao2 = df_nvg_select_id[['id_gleba','id','nr_empty_cells','nr_non_empty_cells','nr_s2_dates']]
df_sub_talhao_month = join_df[['id_gleba', 'mean_month']]

df_st_month = pd.merge(df_subtalhao2, df_sub_talhao_month, on='id_gleba')
df_st_month.drop_duplicates(inplace=True)

fn_st_month = str(my_folder/output_folder/'id_gleba_st_month.csv')
df_st_month.to_csv(fn_st_month, index=False)
df_st_month.columns
df_st_month = pd.read_csv(fn_st_month)

# month groups of column 'mean_month'
group_1 = [1, 2, 11, 12]
group_2 = [3, 4, 9, 10]
group_3 = [5, 6, 7, 8]

# Apply the classification to the dataframe
df_st_month['month_group'] = df_st_month['mean_month'].apply(classify_month)

# Calculate the number and percentage of rows in each group
total_rows = len(df_st_month)
group_counts = df_st_month['month_group'].value_counts()
group_percentages = (group_counts / total_rows) * 100

# Create a summary dataframe
summary_df = pd.DataFrame({
    'Group': group_counts.index,
    'Count': group_counts.values,
    'Percentage': group_percentages.values
})

print(summary_df)

grouped = df_st_month.groupby('month_group').sum()
# Percentage calculations
grouped['perc_empty_cells'] = (grouped['nr_empty_cells'] / grouped['total_rows']) * 100
grouped['perc_non_empty_cells'] = (grouped['nr_non_empty_cells'] / grouped['total_rows']) * 100



# Apply the classification to the dataframe
df_st_month['month_group'] = df_st_month['mean_month'].apply(classify_month)

# Group the data by 'month_group' and calculate the sums
grouped = df_st_month.groupby('month_group').agg({
    'nr_empty_cells': 'sum',
    'nr_non_empty_cells': 'sum'
})

# Calculate the total number of cells for each group
grouped['total_cells'] = grouped['nr_empty_cells'] + grouped['nr_non_empty_cells']

# Calculate the percentage of empty and non-empty cells for each group
grouped['perc_empty_cells'] = (grouped['nr_empty_cells'] / grouped['total_cells']) * 100
grouped['perc_non_empty_cells'] = (grouped['nr_non_empty_cells'] / grouped['total_cells']) * 100

# Print the results
for group in grouped.index:
    print(f"Month group: {group}")
    print(f"  Total number of cells: {grouped.loc[group, 'total_cells']}")
    print(f"  Total number of empty cells: {grouped.loc[group, 'nr_empty_cells']}")
    print(f"  Percentage of empty cells: {grouped.loc[group, 'perc_empty_cells']:.2f}%")
    print(f"  Total number of non-empty cells: {grouped.loc[group, 'nr_non_empty_cells']}")
    print(f"  Percentage of non-empty cells: {grouped.loc[group, 'perc_non_empty_cells']:.2f}%")


import matplotlib.pyplot as plt
import numpy as np

# Set up the bar chart
groups = grouped.index
perc_empty_cells = grouped['perc_empty_cells']
perc_non_empty_cells = grouped['perc_non_empty_cells']

x = np.arange(len(groups))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, perc_empty_cells, width, label='Empty Cells')
rects2 = ax.bar(x + width/2, perc_non_empty_cells, width, label='Non-Empty Cells')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Month Group')
ax.set_ylabel('Percentage')
ax.set_title('Percentage of Empty and Non-Empty Cells by Month Group')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()

# Autolabel function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()


### SAME THING ABOVE BUT FOR EACH MONTH

# Group the data by 'mean_month' and calculate the sums
grouped = df_st_month.groupby('mean_month').agg({
    'nr_empty_cells': 'sum',
    'nr_non_empty_cells': 'sum'
})

# Calculate the total number of cells for each month
grouped['total_cells'] = grouped['nr_empty_cells'] + grouped['nr_non_empty_cells']

# Calculate the percentage of empty and non-empty cells for each month
grouped['perc_empty_cells'] = (grouped['nr_empty_cells'] / grouped['total_cells']) * 100
grouped['perc_non_empty_cells'] = (grouped['nr_non_empty_cells'] / grouped['total_cells']) * 100

# Print the results
for month in grouped.index:
    print(f"Month: {month}")
    print(f"  Total number of cells: {grouped.loc[month, 'total_cells']}")
    print(f"  Total number of empty cells: {grouped.loc[month, 'nr_empty_cells']}")
    print(f"  Percentage of empty cells: {grouped.loc[month, 'perc_empty_cells']:.2f}%")
    print(f"  Total number of non-empty cells: {grouped.loc[month, 'nr_non_empty_cells']}")
    print(f"  Percentage of non-empty cells: {grouped.loc[month, 'perc_non_empty_cells']:.2f}%")

# Plotting the histogram

# Set up the bar chart
months = grouped.index
perc_empty_cells = grouped['perc_empty_cells']
perc_non_empty_cells = grouped['perc_non_empty_cells']

x = np.arange(len(months))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, perc_empty_cells, width, label='Empty Cells')
rects2 = ax.bar(x + width/2, perc_non_empty_cells, width, label='Non-Empty Cells')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Month')
ax.set_ylabel('Percentage')
ax.set_title('Percentage of Empty and Non-Empty Cells by Month')
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.legend()


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()



### CALCULATE TEMP DISTRIBUTION FROM FIRST AND LAST DATE
df_subtalhao.columns
# convert to datetime
df_subtalhao['first_start_date'] = pd.to_datetime(df_subtalhao['first_start_date'], errors='coerce')
df_subtalhao['first_end_date'] = pd.to_datetime(df_subtalhao['first_end_date'], errors='coerce')
df_subtalhao['date_of_biggest_drop'] = pd.to_datetime(df_subtalhao['date_of_biggest_drop'], errors='coerce')
df_subtalhao['first_estimated_date'] = pd.to_datetime(df_subtalhao['first_estimated_date'], errors='coerce')

# calculate the difference in another column
df_subtalhao['diff_days_from_first_date'] = (df_subtalhao['first_end_date'] - df_subtalhao['first_start_date']).dt.days

df_subtalhao['diff_days_from_first_date'] = df_subtalhao.apply(
    lambda row: (
        0 if row['date_of_biggest_drop'] == row['first_start_date']
        else (row['first_start_date'] - row['date_of_biggest_drop']).days
    ) if pd.isnull(row['first_estimated_date']) and row['date_of_biggest_drop'] <= row['first_start_date']
    else None,
    axis=1
)

df_subtalhao['diff_days_from_last_date'] = df_subtalhao.apply(
    lambda row: 0 if row['date_of_biggest_drop'] == row['first_end_date']
    else (row['date_of_biggest_drop'] - row['first_end_date']).days if row['first_end_date'] < row['date_of_biggest_drop']
    else None,
    axis=1
)


#save
fn_temp_dist = str(my_folder/output_folder/'nvg_sub_talhao_temp_dist.csv')
df_subtalhao.to_csv(fn_temp_dist, index=False)



df_temp_dist = pd.read_csv(fn_temp_dist)
df_temp_dist.columns

data = df_temp_dist['diff_days_from_last_date'].dropna()

# Determine the range of the data
min_days = data.min()
max_days = data.max()

# Create bins with 15-day intervals
bins = np.arange(min_days, max_days + 5, 5)

# Plot the histogram
plt.figure(figsize=(10, 6))
hist = plt.hist(data, bins=bins, edgecolor='black')
plt.title('Histogram of diff_days_from_last_date')
plt.xlabel('Difference in Days from Last Date')
plt.ylabel('Frequency')
plt.grid(False)

# Add count number above each bar
for i in range(len(hist[0])):
    plt.text(hist[1][i] + (hist[1][1] - hist[1][0])/2, hist[0][i], int(hist[0][i]), 
             ha='center', va='bottom')

plt.show()

# Count the total number of rows
total_rows = len(data)
print(f"Total number of rows: {total_rows}")

#count id_gleba
# Drop NaN values from 'diff_days_from_last_date' column
data = df_temp_dist['diff_days_from_last_date'].dropna()

# Filter the DataFrame to keep only rows where 'diff_days_from_last_date' is not NaN
filtered_df = df_temp_dist.dropna(subset=['diff_days_from_last_date'])

# Group by 'id_gleba' column
grouped_data = filtered_df.groupby('id_gleba')
num_unique_id_glebas = grouped_data.ngroups


# Count the number of rows within the first 30 days
rows_within_30_days = data[data <= 30].count()
print(f"Number of rows within the first 30 days: {rows_within_30_days}")






# histogram for 'diff_days_from_first_date'
plt.figure(figsize=(10, 6))
plt.hist(df_temp_dist['diff_days_from_first_date'], bins=30, edgecolor='black')
plt.title('Histogram of diff_days_from_first_date')
plt.xlabel('Difference in Days from First Date')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()

# histogram for 'diff_days_from_first_date'
plt.figure(figsize=(10, 6))
plt.hist(df_temp_dist['diff_days_from_last_date'], bins=30, edgecolor='black')
plt.title('Histogram of diff_days_from_last_date')
plt.xlabel('Difference in Days from Last Date')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()





#####VISUALIZATION

### NR CORTES VS NR SUB TALHOES
import seaborn as sns

df_talhao = pd.read_csv(fn_talhao)
df_talhao.columns
# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_talhao['nr_clear_cuts'], df_talhao['num_features'], alpha=0.5)
# Add x = y line
max_value = max(max(df_talhao['nr_clear_cuts']), max(df_talhao['num_features']))
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label='x = y')

plt.xlabel('Number of Clear Cuts')
plt.ylabel('Number of Sub-parcels')
plt.ylim(0, 50)
plt.xlim(0, 50)
plt.title('Number of Sub-parcels vs Number of Clear Cuts')
# plt.grid(True)
plt.show()


# Check if the required columns exist
if 'nr_clear_cuts' not in df_talhao.columns or 'num_features' not in df_talhao.columns:
    raise ValueError("The DataFrame does not contain required columns 'nr_clear_cuts' or 'num_features'.")

# Initialize counts
same_count = 0
fewer_count = 0
more_count = 0

# Iterate through the DataFrame to count the occurrences
for _, row in df_talhao.iterrows():
    if row['nr_clear_cuts'] == row['num_features']:
        same_count += 1
    elif row['nr_clear_cuts'] < row['num_features']:
        fewer_count += 1
    else:
        more_count += 1

print(f"Parcels with the same number of clear cuts and sub-parcels: {same_count}")
print(f"Parcels with fewer clear cuts than sub-parcels: {fewer_count}")
print(f"Parcels with more clear cuts than sub-parcels: {more_count}")


### NR CORTES VS AREA

df_talhao = pd.read_csv(fn_talhao)
df_talhao.columns
# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_talhao['nr_clear_cuts'], df_talhao['area_ha'], alpha=0.5)
# Add x = y line
max_value = max(max(df_talhao['nr_clear_cuts']), max(df_talhao['area_ha']))
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label='x = y')

plt.xlabel('Number of Clear Cuts')
plt.ylabel('Area (ha)')
plt.ylim(0, 50)
plt.xlim(0, 50)
plt.title('Area (ha) vs Number of Clear Cuts')
# plt.grid(True)
plt.show()


### NR DIAS DIFFERENCA VS NR S2 IMAGENS

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_talhao['date_difference_days'], df_talhao['nr_s2_dates'], alpha=0.5)
# Add x = y line
max_value = max(max(df_talhao['date_difference_days']), max(df_talhao['nr_s2_dates']))
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label='x = y')

plt.xlabel('Number of Days')
plt.ylabel('Number of S2 images')
plt.ylim(0, 75)
plt.xlim(0, 100)
plt.title('Scatter Plot of Number of Days Between first and last Clear cut vs Number of S2 Images')
plt.grid(True)
plt.show()

## NR DIAS DIFFERENCA VS NR CORTES
plt.figure(figsize=(10, 6))
plt.scatter(df_talhao['date_difference_days'], df_talhao['nr_clear_cuts'], alpha=0.5)
# Add x = y line
max_value = max(max(df_talhao['date_difference_days']), max(df_talhao['nr_clear_cuts']))
plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label='x = y')

plt.xlabel('Number of Days')
plt.ylabel('Number of Clear Cuts')
plt.ylim(0, 20)
plt.xlim(0, 800)
plt.title('Scatter Plot of Number of Days Between first and last Clear cut vs Number of Clear cuts')
plt.grid(True)
plt.show()


## same but colored by area
x_col = 'date_difference_days'
y_col = 'nr_clear_cuts'

# Define date_difference_days categories and corresponding colors
categories = ['less than 0,5ha', '0,5 - 5ha', '5 - 10ha', '10 - 20ha', '20 - 30ha', '30 - 40ha', 'more than 40ha']

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

# Define bins for date_difference_days
bins = [-np.inf, 0.5, 5, 10, 20, 30, 40, np.inf]

# Add a new column 'category' to df_talhao based on date_difference_days
df_talhao['category'] = pd.cut(df_talhao['area_ha'], bins=bins, labels=categories)

# Plot the scatter plot
plt.figure(figsize=(10, 6))
for category, color in zip(categories, colors):
    plt.scatter(df_talhao[df_talhao['category'] == category][x_col],
                df_talhao[df_talhao['category'] == category][y_col],
                alpha=0.5,
                color=color,
                label=category)

plt.xlabel('Number of Days')
plt.ylabel('Number of Clear cuts')
plt.title('Scatter Plot of Number of Days vs Number of clear cuts')
plt.legend()
plt.xlim(0, 400)
plt.ylim(0, 20)
plt.show()





### NR CORTES VS NR S2 IMAGENS
##### colored by area
x_col = 'nr_clear_cuts'
y_col = 'nr_s2_dates'

# Define date_difference_days categories and corresponding colors
categories = ['less than 0,5ha', '0,5 - 5ha', '5 - 10ha', '10 - 20ha', '20 - 30ha', '30 - 40ha', 'more than 40ha']

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']

# Define bins for date_difference_days
bins = [-np.inf, 0.5, 5, 10, 20, 30, 40, np.inf]

# Add a new column 'category' to df_talhao based on date_difference_days
df_talhao['category'] = pd.cut(df_talhao['area_ha'], bins=bins, labels=categories)

# Plot the scatter plot
plt.figure(figsize=(10, 6))
for category, color in zip(categories, colors):
    plt.scatter(df_talhao[df_talhao['category'] == category][x_col],
                df_talhao[df_talhao['category'] == category][y_col],
                alpha=0.5,
                color=color,
                label=category)

plt.xlabel('Number of Clear Cuts')
plt.ylabel('Number of S2 images')
plt.title('Scatter Plot of Number of Clear Cuts vs Number of S2 Images')
plt.legend()
plt.xlim(0, 50)
plt.ylim(0, 125)
plt.show()


### NR S2 DATES PER NR CUTS PER AREA
###colored by average clear cut month

df_s2 = df_nvg_select_unique[['id_gleba','nr_s2_dates','nr_clear_cuts', 'nr_s2_per_cuts']]
df_month = join_df[['id_gleba', 'mean_month']]

df_s2_month = pd.merge(df_s2, df_month, on='id_gleba')
df_s2_month.drop_duplicates(inplace=True)

fn_s2_month = str(my_folder/output_folder/'id_gleba_s2_month.csv')
df_s2_month.to_csv(fn_s2_month, index=False)
df_s2_month.columns

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_s2_month['nr_clear_cuts'], df_s2_month['nr_s2_dates'], c=df_s2_month['mean_month'], cmap='bwr', alpha=0.5)

# Add labels and title
plt.ylabel('Nr of S2 images')
plt.xlabel('Nr of clear cuts')
plt.title('Scatter Plot of Nr S2 images vs Nr of Clear Cuts')
plt.colorbar(label='Mean Month')
# plt.xlim(0, 10)
# plt.ylim(0, 20)
# Show plot
plt.grid(True)
plt.show()


# //
##### colored by mean month
x_col = 'nr_clear_cuts'
y_col = 'nr_s2_dates'

# Define date_difference_days categories and corresponding colors
categories = ['1,2,11,12', '3,4,9,10', '5,6,7,8']
colors = ['blue', 'green', 'red']

# Define bins for date_difference_days
bins = [-np.inf, 2, 4, np.inf]

# Add a new column 'category' to df_talhao based on date_difference_days
df_s2_month['category'] = pd.cut(df_s2_month['mean_month'], bins=bins, labels=categories)

# Plot the scatter plot
plt.figure(figsize=(10, 6))
for category, color in zip(categories, colors):
    plt.scatter(df_s2_month[df_s2_month['category'] == category][x_col],
                df_s2_month[df_s2_month['category'] == category][y_col],
                alpha=0.5,
                color=color,
                label=category)

plt.xlabel('Number of Clear Cuts')
plt.ylabel('Number of S2 images')
plt.title('Scatter Plot of Number of Clear Cuts vs Number of S2 Images')
plt.legend()
# plt.xlim(0, 50)
# plt.ylim(0, 125)
plt.xlim(0, 20)
plt.ylim(0, 110)
plt.show()



# Apply the classification to the dataframe
df_s2_month['month_group'] = df_s2_month['mean_month'].apply(classify_month)

# Calculate the number and percentage of rows in each group
total_rows = len(df_s2_month)
group_counts = df_s2_month['month_group'].value_counts()
group_percentages = (group_counts / total_rows) * 100

# Create a summary dataframe
summary_df = pd.DataFrame({
    'Group': group_counts.index,
    'Count': group_counts.values,
    'Percentage': group_percentages.values
})

# Display the summary
print(summary_df)

###
# Define custom color mapping
def month_to_color(month):
    if month in [1, 2, 11, 12]:
        return 'blue'
    elif month in [3, 4, 9, 10]:
        return 'green'
    elif month in [5, 6, 7, 8]:
        return 'red'
    else:
        return 'black'  # In case of unexpected values

# Apply color mapping
df_s2_month['color'] = df_s2_month['mean_month'].apply(classify_month)

# Plotting histograms for each color group
plt.figure(figsize=(10, 6))
# Blue months
plt.hist(df_s2_month[df_s2_month['mean_month'].isin([1, 2, 11, 12])]['nr_s2_dates'], bins=20, color='blue', alpha=0.5, label='Months 1, 2, 11, 12')
# Green months
plt.hist(df_s2_month[df_s2_month['mean_month'].isin([3, 4, 9, 10])]['nr_s2_dates'], bins=20, color='green', alpha=0.5, label='Months 3, 4, 9, 10')
# Red months
plt.hist(df_s2_month[df_s2_month['mean_month'].isin([5, 6, 7, 8])]['nr_s2_dates'], bins=20, color='red', alpha=0.5, label='Months 5, 6, 7, 8')

# Add labels and title
plt.xlabel('Nr S2 images')
plt.ylabel('Frequency')
plt.title('Histogram of Nr S2 images by Month Groups')
plt.legend()

# Show plot
plt.show()





#### EMPTY CELLS ao nivel do sub-talhao

df_subtalhao2 = df_nvg_select_id[['id_gleba','id','nr_empty_cells','nr_non_empty_cells','nr_s2_dates']]
df_sub_talhao_month = join_df[['id_gleba', 'mean_month']]

df_st_month = pd.merge(df_subtalhao2, df_sub_talhao_month, on='id_gleba')
df_st_month.drop_duplicates(inplace=True)

fn_st_month = str(my_folder/output_folder/'id_gleba_st_month.csv')
df_st_month.to_csv(fn_st_month, index=False)
df_st_month.columns

x_col = 'nr_empty_cells'
y_col = 'nr_non_empty_cells'

# Define date_difference_days categories and corresponding colors
# categories = ['1,2,3,4,11,12', '5,6,7,8,9,10']
# colors = ['blue', 'red']

categories = ['1,2,11,12', '3,4,9,10','5,6,7,8']
colors = ['blue', 'green','red']
# Define bins for date_difference_days
bins = [-np.inf, 2,4, np.inf]

# Add a new column 'category' to df_talhao based on date_difference_days
df_st_month['category'] = pd.cut(df_st_month['mean_month'], bins=bins, labels=categories)

# Plot the scatter plot
plt.figure(figsize=(10, 6))
for category, color in zip(categories, colors):
    plt.scatter(df_st_month[df_st_month['category'] == category][x_col],
                df_st_month[df_st_month['category'] == category][y_col],
                alpha=0.5,
                color=color,
                label=category)

plt.xlabel('Number of Empty S2 images')
plt.ylabel('Number of Total S2 images')
plt.title('Scatter Plot of Number of Empty S2 images vs Number of Non Empty S2 Images')
plt.legend()
plt.xlim(0, 20)
plt.ylim(0, 110)
plt.show()




## TIMELINE OF CLEAR CUT DATES AND NDVI DROPS
# Convert dates to datetime
df_subtalhao['first_start_date'] = pd.to_datetime(df_subtalhao['first_start_date'])
df_subtalhao['first_end_date'] = pd.to_datetime(df_subtalhao['first_end_date'])
df_subtalhao['date_of_biggest_drop'] = pd.to_datetime(df_subtalhao['date_of_biggest_drop'])

# Timeline plot
plt.figure(figsize=(14, 8))
for i, row in df_subtalhao.iterrows():
    plt.plot([row['first_start_date'], row['first_end_date'], row['date_of_biggest_drop']],
             [i, i, i], marker='o')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('ID')
plt.title('Timeline of Clear Cut Dates and NDVI Drops')
plt.grid(True)
plt.show()


# NUMBER OF CLEAR CUTS PER DAY OF THE MONTH
gdf_exp = gpd.read_file(fn_gpkg, layer=ln_exploracao)

gdf_exp.columns

df_exp_norm = pd.read_csv(fn_exp_norm)

df_exp_norm['dt_real'] = pd.to_datetime(df_exp_norm['dt_real'], errors='coerce')
# Filter rows where the 'activity' column starts with 'CORTE'
df_corte_activities = df_exp_norm[df_exp_norm['Atividade'].str.startswith('CORTE')]
df_corte_activities.columns
df_corte_activities['day'] = df_corte_activities['dt_real'].dt.day

count_cortes = df_corte_activities.groupby('day')['Atividade'].count()

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.bar(id_gleba_counts.index, count_cortes.values, color='green')
plt.xlabel('Day of the Month')
plt.ylabel('Number of Clear cuts')
plt.title('Number of Clear Cuts by Day of the Month')
plt.xticks(range(1, 32))  # Assuming days 1 to 31
plt.grid(axis='y')

plt.show()





#### CRIAR DF PARA IDENTIFICAR OS TALHOES QUE JA FORAM ANALIZADOS VISUALMENTE
####

# Create an empty DataFrame with the desired columns
df_limitations = pd.DataFrame(columns=['id_gleba', 'status', 'limitations'])

# Add a new row to the DataFrame
df_limitations = add_row_to_df(df_limitations,'53010-T001_EG', 'checked', 'corte antes da primeira data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50438-T001_EG', 'checked', 'corte antes da primeira data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'70348-T002_EG', 'checked', 'corte antes da primeira data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50497-T002_EG', 'checked', 'corte antes da primeira data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'53013-T005_EG', 'checked', 'corte antes da primeira data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'53017-T002_EG', 'checked', 'corte antes da primeira data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'51281-T001_EG', 'checked', 'corte antes da primeira data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50002-T007_EG', 'checked', 'corte antes da primeira data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50002-T021_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'50023-T014_EG', 'checked', 'nenhuma das subparcelas foi cortada')
df_limitations = add_row_to_df(df_limitations,'50547-T001_EG', 'checked', 'corte meses depois da primeira data de corte nvg') # um mes depois
df_limitations = add_row_to_df(df_limitations,'50548-T001_EG', 'checked', 'corte antes da primeira data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'53039-T003_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'53099-T015_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'50002-T002_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50446-T001_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'50146-T001_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'50020-T001_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50002-T032_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'54008-T002_EG', 'checked', 'corte depois da ultima data de corte nvg') ## duvida - parece que esta dentro do intervalo de tempo mas h uma parcela que deixa duvidas
df_limitations = add_row_to_df(df_limitations,'50002-T034_EG', 'checked', 'cortes dentro do intervalo de corte nvg') ## parece que o primeiro corte e a volta do talhao
df_limitations = add_row_to_df(df_limitations,'61701-T002_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50230-T001_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'51001-T019_EG', 'checked', 'corte depois da ultima data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50296-T001_EG', 'checked', 'poucas imagens S2; cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50260-T002_EG', 'checked', 'corte antes da primeira data de corte nvg; parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'50011-T002_EG', 'checked', 'images S2 nao sao claras')
df_limitations = add_row_to_df(df_limitations,'50023-T004_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'50007-T004_EG', 'checked', 'corte antes da primeira data de corte nvg; poucas imagens S2')
df_limitations = add_row_to_df(df_limitations,'56001-T003_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'53006-T001_EG', 'checked', 'parte das subparcelas nao foi cortada; corte meses depois da primeira data de corte nvg') # tres meses
df_limitations = add_row_to_df(df_limitations,'55016-T001_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'50588-T001_EG', 'checked', 'parte das subparcelas nao foi cortada; corte depois da ultima data de corte nvg') #duvida?
df_limitations = add_row_to_df(df_limitations,'50445-T001_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50445-T002_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'70136-T002_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'70136-T001_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'55043-T028_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'51351-T001_EG', 'checked', 'corte antes da primeira data de corte nvg;corte depois da ultima data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'55040-T006_EG', 'checked', 'parte das subparcelas nao foi cortada;corte depois da ultima data de corte nvg; poucas imagens S2')
df_limitations = add_row_to_df(df_limitations,'53002-T001_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'53001-T010_EG', 'checked', 'parte das subparcelas nao foi cortada;corte depois da ultima data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'53001-T006_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'50341-T004_EG', 'checked', 'corte antes da primeira data de corte nvg; corte depois da ultima data de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50341-T002_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'51063-T001_EG', 'checked', 'cortes dentro do intervalo de corte nvg')
df_limitations = add_row_to_df(df_limitations,'50283-T003_EG', 'checked', 'parte das subparcelas nao foi cortada')
df_limitations = add_row_to_df(df_limitations,'51001-T015_EG', 'checked', 'corte antes da primeira data de corte nvg; ') # talhao nao estava totalmente plantado??


#save df_limitations
fn_df_limit = str(my_folder/output_folder/'parcels_limitations.csv')
df_limitations.to_csv(fn_df_limit, index=False)

result = join_attribute_to_layer(fn_df_limit, 'id_gleba', fn_nvg_dates_cuts_and_area, 'id_gleba', 'area_ha')
result.setName('limit talhoes com area')
QgsProject().instance().addMapLayer(result)
result.selectAll()
# export layer
result_csv = processing.run("native:saveselectedfeatures", {'INPUT':result, 'OUTPUT':fn_df_limit})
result.removeSelection()

df_limitations = pd.read_csv(fn_df_limit)

# Total area of parcels with clear cuts and NDVI values
total_area = df_talhao['area_ha'].sum()
total_id_glebas = df_talhao['id_gleba'].nunique()
print(f"Total id_glebas: {total_id_glebas}")
print(f"Total area (ha): {total_area}")

# Number of analyzed sub-parcels and total area
num_parcels = df_limitations.shape[0]
area_parcels = df_limitations['area_ha'].sum()
print(f"Number of analyzed id_glebas: {num_parcels}")
print(f"Area of analyzed id_glebas: {area_parcels}")

# Percentages
perc_parcels_analyzed = (num_parcels / total_id_glebas) * 100
perc_area_analyzed = (area_parcels / total_area) * 100
print(f"Percentage of analyzed id_glebas: {perc_parcels_analyzed:.2f}%")
print(f"Percentage of analyzed area: {perc_area_analyzed:.2f}%")

# Filter and calculate total area and percentage for each case
cases = {
    "corte antes da primeira data de corte nvg": df_limitations['limitations'].str.contains('corte antes da primeira data de corte nvg'),
    "corte depois da ultima data de corte nvg": df_limitations['limitations'].str.contains('corte depois da ultima data de corte nvg'),
    "cortes dentro do intervalo de corte nvg": df_limitations['limitations'].str.contains('cortes dentro do intervalo de corte nvg'),
    "parte das subparcelas nao foi cortada": df_limitations['limitations'].str.contains('parte das subparcelas nao foi cortada'),
    "poucas imagens S2": df_limitations['limitations'].str.contains('poucas imagens S2'),
    "nenhuma das subparcelas foi cortada": df_limitations['limitations'].str.contains('nenhuma das subparcelas foi cortada'),
    "corte meses depois da primeira data de corte nvg": df_limitations['limitations'].str.contains('corte meses depois da primeira data de corte nvg')
}

for case, condition in cases.items():
    area = df_limitations[condition]['area_ha'].sum()
    percentage_area = (area / area_parcels) * 100
    num_glebas = df_limitations[condition]['id_gleba'].nunique()
    percentage_glebas = (num_glebas / num_parcels) * 100
    print(f"Total area for '{case}': {area} ha ({percentage_area:.2f}%)")
    print(f"Percentagem de id_glebas com '{case}': {percentage_glebas:.2f}%")


### PARA USAR O CCD
# ccd directory
ccd_folder = 'ccd'

#retrieve  a list of all the id_glebas on df_limitations
id_glebas_list = df_limitations['id_gleba'].tolist()

gdf_nvg_ccd = gpd.read_file(fn_nvg_norm)
gdf_nvg_ccd.columns
# filter
gdf_selected = gdf_nvg_ccd[gdf_nvg_ccd['id_gleba'].isin(id_glebas_list)]
# Drop specific columns
columns_to_drop = [
    'cod_talhao', 'cod_ug', 'ciclo', 'rotacao', 'dt_referen', 'dt_plant', 
    'ocupacao', 'idade_ref', 'idade_plan', 'area_ha', 'forma_plan'
]
gdf_selected = gdf_selected.drop(columns=columns_to_drop)

# Merge with the second GeoDataFrame to add attributes
attributes_to_copy = ['start_date', 'end_date']
gdf_merged = gdf_selected.merge(df_dates_areas[['id_gleba'] + attributes_to_copy], on='id_gleba', how='left')

#select the columns with the first clear cut date
gdf_merged['start_date'] = pd.to_datetime(gdf_merged['start_date'], format='%Y-%m-%d')
# filter rows where year is 2018 or later
gdf_2018 = gdf_merged[gdf_merged['start_date'].dt.year >= 2018]
#convert back to str
gdf_2018['start_date'] = gdf_2018['start_date'].dt.strftime('%Y-%m-%d')

# Save the selected parcels to a new shapefile
fn_limitations_to_ccd = str(my_folder/ccd_folder/'selected_parcels_ccd.gpkg')
gdf_2018.to_file(fn_limitations_to_ccd)



## DISTRIBUIÇAO DA FREQUENCIA DAS DATAS


# Define your folder paths

# List of shapefiles to process
file_names = [
    '53001-T010_53013-T005_53001-T006_53010-T001_53002-T001_53017-T002.shp',
    '53039-T003_50020-T001.shp',
    '50002-T021_50002-T007_50002-T032_50002-T002_50002-T034.shp',
    '50445-T001_50445-T002.shp', '50588-T001_50548-T001.shp',
    '70348-T002_50438-T001.shp',
    '51063-T001_50547-T001.shp', '50023-T004.shp', '50146-T001.shp', '50230-T001.shp', 
    '50260-T002.shp', '53006-T001.shp', '53099-T015.shp', '54008-T002.shp', '55016-T001.shp',
    '55043-T028.shp', '61701-T002.shp', '50446-T001.shp'
]

for filename in file_names:
    # extract id_glebas
    id_glebas = extract_id_glebas(filename)
    #print(id_glebas)
    for id_gleba in id_glebas:
        # get singlepart shp
        singlepart_file = 'nvg_singlepart_' + id_gleba + '.shp'
        fn_sp_file = str(my_folder/output_folder/singlepart_file)
        #extract by location
        fn_file_ccd = str(my_folder / ccd_folder / f'ccd_{id_gleba}.shp')
        fn_file = str(my_folder/ccd_folder/ filename)
        extract_by_location_permanent(fn_file, fn_sp_file, fn_file_ccd)
        print(f"Processed {filename}, output saved as {fn_file_ccd}")
        # read df per id_gleba
        df_talhao_ccd = gpd.read_file(fn_file_ccd)
        # Extract the relevant values for the specified id_gleba
        row = df_dates_areas[df_dates_areas['id_gleba'] == id_gleba]
        start_date = pd.to_datetime(row['start_date'].values[0])
        end_date = pd.to_datetime(row['end_date'].values[0])

        #add columns to df_talhao_ccd
        # Add the new columns to df_talhao_ccd
        df_talhao_ccd['start_date'] = start_date
        df_talhao_ccd['end_date'] = end_date

        # Ensure the tBreak_ddm column contains lists of strings
        df_talhao_ccd['tBreak_ddm'] = df_talhao_ccd['tBreak_ddm'].apply(eval)

        # Apply the function to create the 'drop_date' column
        df_talhao_ccd['drop_date'] = df_talhao_ccd['tBreak_ddm'].apply(lambda x: filter_dates(x, start_date, end_date))

        #convert datetime fiels back to string because esri shapefile formats do ot support datetime fields
        df_talhao_ccd['start_date'] = df_talhao_ccd['start_date'].dt.strftime('%Y-%m-%d')
        df_talhao_ccd['end_date'] = df_talhao_ccd['end_date'].dt.strftime('%Y-%m-%d')
        df_talhao_ccd['drop_date'] = df_talhao_ccd['drop_date'].dt.strftime('%Y-%m-%d') if df_talhao_ccd['drop_date'] is not None else None

        # Convert tBreak_ddm lists to strings
        df_talhao_ccd['tBreak_ddm'] = df_talhao_ccd['tBreak_ddm'].apply(lambda x: ','.join(x))

        #save
        output_shapefile = str(my_folder/ccd_folder/f'df_ccd_{id_gleba}.shp')
        df_talhao_ccd.to_file(output_shapefile)
        print(f"Id_gleba {id_gleba} saved as {output_shapefile}")



### color 
#id_gleba = '61701-T002_EG'
fn_df_talhao_ccd_with_datedrop = str(my_folder/ccd_folder/f'df_ccd_{id_gleba}.shp')

ccd_table = gpd.read_file(fn_df_talhao_ccd_with_datedrop)
# add column with id_gleba
ccd_table['id_gleba'] = id_gleba
ccd_table.columns
# df_sorted.columns

#add and zoom to layer
layer=iface.addVectorLayer(fn_df_talhao_ccd_with_datedrop,'df_ccd_'+ id_gleba,'ogr')

# set labels according to estimated clear-cut dates 
layer = iface.activeLayer()
#field_name = 'estimated_'
field_name = 'drop_date'
field_index = layer.fields().indexFromName(field_name)
unique_values = sorted(layer.uniqueValues(field_index))

category_list = []
for value in unique_values:
    symbol = QgsSymbol.defaultSymbol(layer.geometryType())
    category = QgsRendererCategory(value, symbol, str(value))
    category_list.append(category)

# color ramp
ramp_name = 'Turbo'
default_style = QgsStyle().defaultStyle()
color_ramp = default_style.colorRamp(ramp_name)
renderer = QgsCategorizedSymbolRenderer(field_name, category_list)
renderer.updateColorRamp(color_ramp)
layer.setRenderer(renderer)
layer.triggerRepaint()


#how many parcels have clear cuts dates from 2018

#select the columns with the first clear cut date
df_talhao['first_start_date'] = pd.to_datetime(df_talhao['first_start_date'], format='%Y-%m-%d')
# filter rows where year is 2018 or later
df_2018 = df_talhao[df_talhao['first_start_date'].dt.year >= 2018]
# count parcels
unique_id_gleba_2018 = len(df_2018['id_gleba'])
print(unique_id_gleba_2018)

#list of id_glebas with first clear cut starting in 2018
id_glebas_list_2018 = df_2018['id_gleba'].tolist()

# filter
gdf_selected_2018 = gdf_nvg[gdf_nvg['id_gleba'].isin(id_glebas_list_2018)]
# Drop specific columns
columns_to_drop = [
    'cod_talhao', 'cod_ug', 'ciclo', 'rotacao', 'dt_referen', 'dt_plant', 
    'ocupacao', 'idade_ref', 'idade_plan', 'area_ha', 'forma_plantacao'
]
gdf_selected_2018 = gdf_selected_2018.drop(columns=columns_to_drop)

# Merge with the second GeoDataFrame to add attributes
attributes_to_copy = ['first_start_date', 'start_date', 'first_end_date','end_date', 'area_ha']
gdf_merged = gdf_selected_2018.merge(df_dates_areas[['id_gleba'] + attributes_to_copy], on='id_gleba', how='left')
# Convert datetime to str only if columns exist and are of datetime type
if 'start_date' in gdf_merged.columns and pd.api.types.is_datetime64_any_dtype(gdf_merged['start_date']):
    gdf_merged['start_date'] = gdf_merged['start_date'].dt.strftime('%Y-%m-%d')
else:
    gdf_merged['start_date'] = gdf_merged['start_date'].astype(str)

if 'end_date' in gdf_merged.columns and pd.api.types.is_datetime64_any_dtype(gdf_merged['end_date']):
    gdf_merged['end_date'] = gdf_merged['end_date'].dt.strftime('%Y-%m-%d')
else:
    gdf_merged['end_date'] = gdf_merged['end_date'].astype(str)

# Save the selected parcels to a new shapefile
fn_all_nvg_2018_ccd = Path(my_folder) / ccd_folder / 'nvg_2018_ccd.gpkg'
gdf_merged.to_file(fn_all_nvg_2018_ccd, driver='GPKG')


#### DATES DISTRIBUTION FOR TILES

#from ccd results per tile, create a new shp file with a 'drop_date' column
# save the file as 'df_ccd_{tilename}'
tilename = 'tile29TNG'
tilename_shp = tilename + '.shp'
output_shapefile, fn_output, df_tile = create_df_ccd_per_tile(my_folder, ccd_folder, tilename_shp, ['id_gleba','start_date','end_date'])

#rename files in S2 CCD folder to dates
s2_tilename = 'S2_T29TNG'
folder_path = str(my_folder/ccd_folder/'tiles'/s2_tilename)
rename_tiff_s2_images_to_dates(folder_path)

# create a list of id_glebas to work as a check list
# Extract the unique values from the 'id_gleba' column
output_csv = f'unique_glebas_{tilename}.csv'
unique_id_glebas_csv = unique_id_glebas_per_tile(df_tile, output_csv)

#join all NVG singlepart files which id_gleba is on the list of visual analysed parcels
fn_filename_output_va = str(my_folder / ccd_folder / 'tiles' / 'merged_glebas_visual_analysis.shp')
merge_visual_analysis_single_shp(my_folder, ccd_folder, output_folder, fn_filename_output_va)


#join all df_ccd files (ccd results per tile - this is at pixel level)
fn_output_merged_ccd = str(tile_folder / 'merged_ccd_visual_analysis.shp')
columns_to_keep = ['id_gleba', 'start_date', 'end_date', 'drop_date', 'geometry']
merge_ccd_shapefiles(tile_folder, columns_to_keep, fn_output_merged_ccd)
### add 'id' column to it
#read shapefile with 'id' column for all id_glebas
fn_nvg_singlepart = str(my_folder/output_folder/'nvg_singlepart.shp')
# field to join
list_fields_to_join = ['id']
fn_output = str(my_folder / ccd_folder / 'tiles'/ 'merged_ccd_visual_analysis_id.shp')
## join attributes by location
att_by_loc = join_field_by_location (fn_output_merged_ccd, fn_nvg_singlepart, list_fields_to_join, fn_output)


#join tables by id gleba - add attributes ECCD1, ECCD2 and NC
attribute_to_copy = ['ECCD1','ECCD2','NC']
fn_merged_glebas_visual_analysis = str(my_folder/ccd_folder /'tiles' / 'merged_glebas_visual_analysis.shp')
result = join_attribute_to_layer(fn_output, 'id', fn_merged_glebas_visual_analysis, 'id', attribute_to_copy)
# result.setName('CCD results All Tiles')
# QgsProject().instance().addMapLayer(result)
result.selectAll()
# export layer
fn_ccd_output = str(my_folder/ccd_folder /'tiles'/'ccd_results_all_tiles_visual_analysis.shp')
result_ccd = processing.run("native:saveselectedfeatures", {'INPUT':result, 'OUTPUT':fn_ccd_output})
result.removeSelection()

### CREATE COLUMNS DATA0 AND DATA1
gdf_filepath = fn_ccd_output
fn_alltiles_ccd_data01 = str(my_folder / ccd_folder / 'tiles'/ 'ccd_results_all_tiles_visual_analysis_data0_data1.shp')
create_time_interval_columns(fn_ccd_output, fn_alltiles_ccd_data01)

## number of subparcels
# Load the GeoDataFrame from the file
gdf = gpd.read_file(fn_ccd_output)
unique_ids = gdf['id'].nunique()
print(f"Number of unique 'id': {unique_ids}")


## Calculate CCD correct subparcels
gdf_ccd_all_tiles = gpd.read_file(fn_alltiles_ccd_data01)
ccd_correct_subparcels1, ccd_correct_subparcels2, total_ccd_correct_sp, percentage_ccd_correct, ccd_eccd1_subparcels, percentage_ccd_eccd = calculate_ccd_correct_subparcels(gdf_ccd_all_tiles)

print("Total CCD Correct Sub-parcels:", total_ccd_correct_sp)
print("Percentage CCD Correct:", percentage_ccd_correct)
print("CCD ECCD1 Sub-parcels:", ccd_eccd1_subparcels)
print("Percentage CCD ECCD1:", percentage_ccd_eccd)


###CALCULATE THE DIFFERENCE BETWEEN DATA0 AND DATA1 AND PLOT HISTOGRAM

#read gdf
gdf_ccd_all_tiles = gpd.read_file(fn_alltiles_ccd_data01)
#output file
output_file = str(my_folder / ccd_folder / 'tiles'/ 'gdf_hist.shp')
#calculate date difference between data0 and data1
subset_gdf = calculate_date_difference(gdf_ccd_all_tiles, 'data0', 'data1', output_file, new_col_name='date_diff')

# parameters for histogram
bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
title = 'Histogram of Date Differences (in days) between data0 and data1'
x_label = 'Days'
y_label = 'Frequency'
# call the hostogram function
create_histogram(data=subset_gdf, column='date_diff', bin_edges=bin_edges, title=title, x_label=x_label, y_label=y_label, log_scale=True)
# calculate stats for histogram subset
total_percentage_diff, perc_equals_10, perc_lessthan_10, perc_between_10_and_30, perc_between_30_and_60, perc_morethan_60 = calculate_date_diff_stats(subset_gdf, 'date_diff')


## ALL STATS
total_gdf = gpd.read_file(fn_alltiles_ccd_data01)
#totals
total_pixels, total_parcels, total_subparcels = calculate_totals(total_gdf)
# NULLS and isolated NULLS
(   total_nulls_count, iso_null_count, non_iso_null_count, percentage_nulls_total, 
    percentage_iso_nulls_total, percentage_iso_within_null, subset_total_nulls, 
    subset_isolated_nulls, subset_nulls_not_iso) = analyze_null_pixels(gdf)

print('Total number of NULL pixels:', total_nulls_count)
print('Total number of isolated NULL pixels:',iso_null_count)
print('Total number of non-isolated NULL pixels:',non_iso_null_count)
print('Percentage of total NULL pixels:',percentage_nulls_total)
print('Percentage of total isolated NULL pixels:',percentage_iso_nulls_total)

# NC stats
nc_stats = analyze_nc_values(subset_total_nulls, subset_nulls_not_iso)
# print NC stats
print("Count of NC = 1:", nc_stats["nc1_count"])
print("Count of subparcels with NC = 1:", nc_stats["subparcels_null_nc1_count"])
print("Count of NC between 0.2 and < 1:", nc_stats["nc_between_02_and_1_count"])
print("Count of NC is NULL:", nc_stats["nc_is_null_count"])













#total nr of pixels
total_pixels_db = len(total_gdf)
#total nr of parcels
total_parcels = total_gdf['id_gleba'].nunique()
#total nr of sub-parcels
total_subparcels = total_gdf['id'].nunique()
#total number of NULL pixels
subset_total_nulls = total_gdf[total_gdf['drop_date'].isnull()]
#total nr of isolated NULLs
subset_isolated_nulls = subset_total_nulls[
    (subset_total_nulls['data0'].isnull() & 
    subset_total_nulls['data1'].isnull() & 
    subset_total_nulls['drop_date'].isnull() &
    subset_total_nulls['ECCD1'].isnull()& 
    subset_total_nulls['ECCD2'].isnull()& 
    subset_total_nulls['NC'].isnull())
    ]
# Subset of nulls that are not isolated
subset_nulls_not_iso = subset_total_nulls[~subset_total_nulls.index.isin(subset_isolated_nulls.index)]


iso_null = len(subset_isolated_nulls)
nulls = len(subset_total_nulls)
nulls_not_iso = nulls - iso_null

#percentage of NULL pixels
percentage_nulls_total = (nulls / total_pixels_db) * 100
#percentage of isolated NULL
percentage_iso_nulls_total = (iso_null / total_pixels_db) * 100
#percentage of isolated NULL within the total NULLS
percentage_iso_within_null = (iso_null / nulls)*100

# from NULL pixels -- calculate NC 
## NC = 1 so this is a CCD-correct sub-parcel
null_nc1 = subset_total_nulls.loc[subset_total_nulls['NC'] == 1]
print(len(null_nc1))
subparcels_null_nc1 = null_nc1.groupby('id').nunique()
print(len(subparcels_null_nc1))

## NC between 0.2 and <1
null_ncnotnull = subset_nulls_not_iso.loc[
    (subset_nulls_not_iso['NC'] >= 0.2) & (subset_nulls_not_iso['NC'] < 1)]

## NC is NULL
null_ncnotnull = subset_nulls_not_iso.loc[
    (subset_nulls_not_iso['NC'].isnull())]











































## min, max and mean of isolated NULLs per sub-parcel

# nr of sub-parcels with isolated NULLs
unique_parcels = subset_isolated_nulls['id'].nunique()
# nr of isolated NULLs per sub-parcel
null_counts_per_parcel = subset_isolated_nulls.groupby('id').size()
# min, max, mean
min_nulls = null_counts_per_parcel.min()
max_nulls = null_counts_per_parcel.max()
avg_nulls = null_counts_per_parcel.mean()

# Display the results
print(f"Number of parcels: {total_parcels}")
print(f"Number of subparcels: {total_subparcels}")
print(f"Number of total pixels: {total_pixels_db}")
print(f"Number of NULL pixels: {nulls}")
print(f"Number of isolated NULL pixels: {iso_nulls}")
print(f"Percentage of NULL pixel in the DB: {percentage_nulls_total}")
print(f"Percentage of isolated NULL in the DB: {percentage_iso_nulls_total}")
print(f"Percentage of isolate NULL within the NULLs: {percentage_iso_within_null}")
print(f"Number of sub-parcels with isolated NULLs: {unique_parcels}")
print(f"Minimum number of NULLs in a parcel: {min_nulls}")
print(f"Maximum number of NULLs in a parcel: {max_nulls}")
print(f"Average number of NULLs per parcel: {avg_nulls:.2f}")

#calculate the average number of pixels per sub-parcel
pixels_per_parcel = gdf_hist.groupby('id').size()
average_pixels_per_parcel = pixels_per_parcel.mean()
## percentage of average isolated NULLS per parcel
perc_iso_nulls_per_sp = (avg_nulls/average_pixels_per_parcel)*100
#nr minimo/max e media de pixels por subparcela
unique_parcels_sp = isolated_nulls['id'].nunique()
null_counts_per_sp = isolated_nulls.groupby('id').size()
min_nulls_sp = null_counts_per_sp.min()
max_nulls_sp = null_counts_per_sp.max()
avg_nulls_sp = null_counts_per_sp.mean()

print(f"Average of pixels per sub-parcel: {average_pixels_per_parcel}")
print(f"Average of isolated NULL pixels per sub-parcel: {perc_iso_nulls_per_sp}")
print(f"Number of unique sub-parcels: {unique_parcels_sp}")
print(f"Minimum number of NULLs in a parcel: {min_nulls_sp}")
print(f"Maximum number of NULLs in a parcel: {max_nulls_sp}")
print(f"Average number of NULLs per parcel: {avg_nulls_sp:.2f}")

## nr de pixels correctly identified by CCD
subset_correct_pixels = isolated_nulls[
    (isolated_nulls['drop_date'].notnull() & 
     isolated_nulls['ECCD1'].isnull() & 
     isolated_nulls['ECCD2'].isnull() & 
     isolated_nulls['NC'].isnull()) 
    |
    (isolated_nulls['drop_date'].isnull() & 
     isolated_nulls['NC'] == 1)
]

print(len(subset_correct_pixels))

###calculate the max proportion of isolated nulls in sub-parcel
# Step 1: Calculate the total number of pixels per sub-parcel
pixels_per_parcel = gdf_hist.groupby('id').size()

# Step 2: Calculate the number of isolated NULL pixels per sub-parcel
null_counts_per_parcel = subset_isolated_nulls.groupby('id').size()

# Step 3: Calculate the percentage of isolated NULL pixels in each sub-parcel
percentage_isolated_nulls_per_parcel = (null_counts_per_parcel / pixels_per_parcel.reindex(null_counts_per_parcel.index)) * 100

# Step 4: Find the maximum percentage and the corresponding sub-parcel ID
max_percentage = percentage_isolated_nulls_per_parcel.max()
max_percentage_id = percentage_isolated_nulls_per_parcel.idxmax()

# Display the results
print(f"The maximum percentage of isolated NULL pixels in any sub-parcel is: {max_percentage:.2f}%")
print(f"The sub-parcel with the maximum percentage of isolated NULLs has the ID: {max_percentage_id}")



#### AUTOMATE THE VISUAL ANALYSIS

from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsTemporalProperties
from PyQt5.QtCore import QDateTime
import os
from typing import Tuple, Optional


import imageio
print(imageio.__version__)


# choose tile
ln_tile = 'tile29TPE'
fn_tile_to_use = str(my_folder/ccd_folder/'tiles'/f'df_ccd_{ln_tile}.shp') #ccd results
# folder with S2 images
ln_s2_folder = 'S2_T29TPE'
fn_s2_folder = str(my_folder/ccd_folder/'tiles'/ ln_s2_folder)
# choose parcel
id_gleba = '53001-T006_EG'
#get parcels shapefile
fn_id_gleba = str(my_folder/output_folder/f'nvg_singlepart_{id_gleba}.shp')

# retrive ccd results for id_gleba
## extract by location 
result = extract_by_location (fn_tile_to_use, fn_id_gleba)
result.setName('parcels CCD results')
QgsProject().instance().addMapLayer(result)

# convert drop date into date time
convert_drop_date_to_datetime(result, 'drop_date', 'drop_date_dt')
# get start and end dates and convert to yyyymmdd format
start_date_yyyymmdd, end_date_yyyymmdd = convert_dates_to_yyyymmdd(result)

# #set legend
result = set_layer_legend(result, 'drop_date', 'Turbo')

# get first and last S2 images to be used in temporal range
# Folders and files
s2_folder = fn_s2_folder  # The folder with S2 images
# start_date = datetime.strptime(start_date_yyyymmdd, '%Y%m%d')
# end_date = datetime.strptime(end_date_yyyymmdd, '%Y%m%d')

first_image, last_image = find_first_last_s2_images(fn_s2_folder, start_date_yyyymmdd, end_date_yyyymmdd)
if first_image and last_image:
    print(f"First Image Date: {first_image}")
    print(f"Last Image Date: {last_image}")
else:
    print("No images found within the specified date range.")







