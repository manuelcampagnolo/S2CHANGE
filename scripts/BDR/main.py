import os
import sys
import csv
from osgeo import ogr
from osgeo import gdal, osr 
from pathlib import Path 
from console.console import _console 
from qgis.core import QgsProject
from qgis.core import QgsProject, QgsExpression, QgsExpressionContext, QgsExpressionContextUtils
from qgis.core import QgsVectorLayer
from qgis.core import QgsField, QgsFields
from PyQt5.QtCore import QVariant
import collections
from PyQt5.QtWidgets import QAction
import processing
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from datetime import datetime, timedelta
import numpy as np
import geopandas as gpd
import pandas as pd
from pathlib import Path 
from console.console import _console 
import fiona
import ee
import geemap
from dateutil.relativedelta import relativedelta

#project and data folders
project_name='database_navigator'
input_folder= 'input'
output_folder='output'
ndvi_folder = 'NDVI'


# Working directory:
# |----myfolder
#    |---- explore_data.py
#    |---- my_functions.py
#    |---- database_navigator.qgz 
#    |---- input_folder
#         |---- NVG_proprios_2015_2023_clean.gpkg
#    |---- output_folder

# Determine path to working directory ("my_folder")
# Find path to the directory where the script is 
script_path = Path(_console.console.tabEditorWidget.currentWidget().path)
my_folder=script_path.parent
# load my_functions.py
exec(Path(my_folder/ "my_functions_main.py").read_text())


# Layers name:
ln_gpkg='NVG_proprios_2015_2023_clean.gpkg'
ln_nvg = 'NVG_2015_2023_Proprios_clean'
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
df_exp = gpd.read_file(fn_gpkg, layer=ln_exploracao).drop(columns='geometry')
gdf_silv = gpd.read_file(fn_gpkg, layer=ln_silvicultura)
df_silv = gdf_silv.drop(columns=gdf_silv.geometry.name)


# NORMALIZATION
#rename columns 
## table exploracao
gdf_exp.columns = gdf_exp.rename(columns={'Id Gleba': 'id_gleba','Id Projeto': 'cod_un', 'Talhão': 'cod_talhao', 'Data Real': 'dt_real'}) #rename column
#table silvicultura
df_silv = df_silv.rename(columns={'Id Projeto': 'cod_un', 'Talhão': 'cod_talhao', 'Data Operação': 'dt_operacao', 'Desc. Atividade': 'desc_atividade'}) #rename column

#create normalized tables
## table nvg
# Split the 'ocupacao' column into two columns based on the '-' sign
# the output is two columns one with "ocupacao" and the other with "forma de plantacao"
gdf_nvg[['ocupacao', 'forma_plantacao']] = gdf_nvg['ocupacao'].str.split(' - ', n=1, expand=True)
fn_nvg_norm = str(my_folder / output_folder / 'nvg_norm.shp')
gdf_nvg.to_file(fn_nvg_norm, encoding='utf-8')
#print(gdf_nvg.columns)

## create table management_units
management_units = pd.DataFrame(gdf_nvg['cod_ug'].unique(), columns=['cod_ug'])
# Save the GeoDataFrame as a shapefile or another format
fn_management_units = str(my_folder / output_folder / 'management_units.csv')
management_units.to_csv(fn_management_units)

##add primary key to tables exploracao 
df_exp.insert(0, 'exploracao_ID', range(1, 1 + len(df_exp)))
exp = pd.DataFrame(df_exp)
# Save the GeoDataFrame as a shapefile or another format
fn_exp_norm = str(my_folder / output_folder / 'exploracao_norm.csv')
exp.to_csv(fn_exp_norm)

## add primary key to table silvicultura
df_silv.insert(0, 'silvicultura_ID', range(1, 1 + len(df_silv)))
# Save the GeoDataFrame to a new shapefile
fn_silv_norm = str(my_folder / output_folder / 'silvicultura_norm.csv')
df_silv.to_csv(fn_silv_norm)

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

####### GOOGLE EARTH ENGINE

#Inputs
#nvg_norm
ln_nvg='nvg_norm.shp'
fn_nvg = str(my_folder/output_folder/ln_nvg)
#final_df_sorted
ln_df_sorted = 'final_df_sorted_no_spaces.csv'
fn_df_sorted = str(my_folder/output_folder/ln_df_sorted)
df_sorted = pd.read_csv(fn_df_sorted)



# Variables
id_gleba = '50445-T002_EG'
# id_gleba = '50550-T001_EG'
# id_gleba = '50002-T001_EG'
# id_gleba = '50033-T001_EG'
# id_gleba = '53739-T001_EO'
cloud_percentage = 10


# 1st step: is to create a singlepart talhao of the id_gleba we want 

ln_talhao_singlepart = 'nvg_singlepart_' + id_gleba + '.shp'
fn_talhao_singlepart = str(my_folder / output_folder / ln_talhao_singlepart)

if not os.path.exists(fn_talhao_singlepart):
    talhao = extract_talhao_from_nvg(fn_nvg, id_gleba)
    talhao_singlepart = multi_to_singlepart(talhao)
    talhao_singlepart_pk, talhao_singlepart_shp, fn_talhao_singlepart = add_primary_key_talhao(talhao_singlepart)

# 2nd step: get start and end date of that same talhao

date_pairs = find_date_pairs(df_sorted, id_gleba)
new_start_dates, new_end_dates, modified_date_pairs = dates_with_two_months_diff(date_pairs)

# 3rd step: get a csv file with median NDVI values from Google Earth Engine

ee.Initialize()
nvg = geemap.shp_to_ee(fn_talhao_singlepart, encoding='latin-1') #encoding para nao ter erro

for i, (start_date, end_date) in enumerate(modified_date_pairs):
    ln_median_ndvi = f'Median_NDVI_{id_gleba}_{i}.csv'
    csv_path = str(my_folder/ndvi_folder/ln_median_ndvi)
    
    if not os.path.exists(csv_path):
        medianNDVI = ndvi_median_gee(start_date, end_date, nvg, cloud_percentage)

        output_dir = str(my_folder/ndvi_folder)

        # Export the result as a CSV file using geemap
        geemap.ee_to_csv(
            ee_object=medianNDVI,
            filename=os.path.join(output_dir, f'Median_NDVI_{id_gleba}_{i}.csv'),
        )

        # get csv file
        ln_median_ndvi = f'Median_NDVI_{id_gleba}_{i}.csv'
        csv_path = str(my_folder/ndvi_folder/ln_median_ndvi)




ln_pivot_table = f'pivot_table_{id_gleba}_{i}.csv'
fn_pivot_table = str(my_folder / output_folder / ln_pivot_table)

if not os.path.exists(fn_pivot_table):
    # Read CSV with pandas DataFrame
    df_median_ndvi = pd.read_csv(csv_path, header=0)
    #estimate clear cuts dates
    pivot_table = convert_to_pivot_table(df_median_ndvi)
    pivot_table_with_estimated_date = calculate_biggest_ndvi_drop_and_estimated_date(pivot_table)
    # Save the GeoDataFrame as a shapefile or another format
    # fn_pivot_table = str(my_folder / output_folder / ('pivot_table_' + str(id_gleba) + '.csv'))
    fn_pivot_table = str(my_folder / output_folder / (f'pivot_table_{id_gleba}_{i}.csv'))
    pivot_table_with_estimated_date.to_csv(fn_pivot_table)




## Label the parcel
# join 'estimated_date' to shapefile and add layer to the project
result = join_attribute_to_layer(fn_talhao_singlepart, 'id', fn_pivot_table, 'id', 'estimated_date')
result.setName('talhao_'+ id_gleba)
QgsProject().instance().addMapLayer(result)


# set labels according to estimated clear-cut dates 
layer = iface.activeLayer()
field_name = 'estimated_date'
field_index = layer.fields().indexFromName(field_name)
unique_values = layer.uniqueValues(field_index)

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


# 5th step: create a dataset with columns of 'data_estimada'

ln_nvg_singlepart = 'nvg_singlepart.shp'
fn_nvg_singlepart = str(my_folder/output_folder/ln_nvg_singlepart)
# read nvg_singlepart as gdf
gdf_nvg_singlepart = gpd.read_file(fn_nvg_singlepart)


fn_expanded = (my_folder / output_folder / 'df_expanded.csv')
expanded_df.to_csv(fn_expanded)

if not os.path.exists(fn_expanded):
    expanded_df = create_expanded_df(df_sorted, gdf_nvg_singlepart, 'id_gleba', 'id', 'id_gleba', 'left')

    # Save the GeoDataFrame as a shapefile or another format
    fn_expanded = (my_folder / output_folder / 'df_expanded.csv')
    expanded_df.to_csv(fn_expanded)


# 6th step: fill the data_estimada columns

# fn_pivot_table = str(my_folder / output_folder / ('pivot_table_' + str(id_gleba) + '.csv'))
fn_pivot_table = str(my_folder / output_folder / (f'pivot_table_{id_gleba}_{i}.csv'))
pivot_table = pd.read_csv(fn_pivot_table)


# Iterate through each row of pivot_table
for index, row in pivot_table.iterrows():
    # Extract id and estimated_date
    id_value = row['id']
    estimated_date = row['estimated_date']
    
    # Find matching row in expanded_df based on id
    matching_row = expanded_df[expanded_df['id'] == id_value]
    
    # Iterate through data_estimada columns to find the matching date
    for col in expanded_df.columns:
        # Check if it's a 'data_estimada' column
        if col.startswith('data_estimada'):
            # Extract the corresponding data column index
            data_index = int(col.split('data_estimada')[1])  # Extract the index after 'data_estimada'
            data_col = 'data' + str(data_index)  # Form the corresponding 'data' column name
            
            # Check if the data value matches the estimated_date
            if expanded_df.at[matching_row.index[0], data_col] == estimated_date:
                # Update corresponding data_estimada column with 1
                expanded_df.at[matching_row.index[0], col] = 1


# Save the GeoDataFrame as a shapefile or another format
fn_expanded = (my_folder / output_folder / 'df_expanded_updated.csv')
expanded_df.to_csv(fn_expanded)




######################################################################





## loop for all



csv_paths = ndvi_mediana_from_gee(start_date, end_date, modified_date_pairs, nvg, cloud_percentage, id_gleba)

# Define ln_pivot_table and fn_pivot_table outside the function
ln_pivot_table = f'pivot_table_{id_gleba}_{i}.csv'
fn_pivot_table = str(my_folder / output_folder / ln_pivot_table)

if not os.path.exists(fn_pivot_table):
    for csv_path in ndvi_mediana_from_gee:
        # Read CSV with pandas DataFrame
        df_median_ndvi = pd.read_csv(csv_path, header=0)
        
        # Estimate clear cuts dates
        pivot_table = convert_to_pivot_table(df_median_ndvi)
        pivot_table_with_estimated_date = calculate_biggest_ndvi_drop_and_estimated_date(pivot_table)
        
        # Save the DataFrame as a CSV file
        ln_pivot_table = f'pivot_table_{id_gleba}_{i}.csv'
        fn_pivot_table = str(my_folder / output_folder / ln_pivot_table)
        pivot_table_with_estimated_date.to_csv(fn_pivot_table)




















#convert nvg from multipart to singlepart
nvg_singlepart = multi_to_singlepart(fn_nvg)
# add primary key to all sub-parcels
nvg_singlepart_pk, nvg_singlepart_shp, fn_nvg_singlepart = add_primary_key_talhao(nvg_singlepart)
#add layer to map 
nvg_singlepart_pk.setName('NVG Singlepart PK')
QgsProject().instance().addMapLayer(nvg_singlepart_pk)



# extract talhao from nvg_singlepart
talhao = extract_talhao_from_nvg(fn_nvg_singlepart, id_gleba)
# talhao_singlepart = multi_to_singlepart(talhao)
ln_talhao = 'nvg_singlepart_' + str(id_gleba) + '.shp'
fn_talhao = str(my_folder/output_folder/ln_talhao)
talhao.selectAll()
# export layer
talhao_shp = processing.run("native:saveselectedfeatures", {'INPUT':talhao, 'OUTPUT':fn_talhao})
talhao.removeSelection()




#### ASSUMINDO QUE USAMOS A PRIMEIRA E ULTIMA DATA DE CORTE
# Variables
id_gleba = '50445-T001_EG'
cloud_percentage = 10

# for a single id_gleba
talhao = extract_talhao_from_nvg(fn_nvg, id_gleba)
talhao_singlepart = multi_to_singlepart(talhao)
# talhao_singlepart_pk, talhao_shp, fn_talhao = add_primary_key_talhao(talhao_singlepart)
ln_talhao = 'nvg_singlepart_' + str(id_gleba) + '.shp'
fn_talhao = str(my_folder/output_folder/ln_talhao)
# get start and end dates
first_start_date, first_end_date = filter_and_select_dates(df_sorted, id_gleba)
#start_date, end_date = start_and_end_dates_two_months (first_start_date, first_end_date)
start_date, end_date = start_and_end_dates_two_months(first_start_date.iloc[0], first_end_date.iloc[0])
# Initialize GEE
ee.Initialize()
# add talhao to feature collection
nvg = geemap.shp_to_ee(fn_talhao)
#calculate median values for each subtalhao
csv_path = calculate_median_from_gee_talhao(fn_talhao, start_date, end_date, cloud_percentage, id_gleba)
# Read CSV with pandas DataFrame
df_median_ndvi = pd.read_csv(csv_path)
#estimate clear cuts dates
pivot_table = convert_to_pivot_table(df_median_ndvi)
calculate_biggest_ndvi_drop_and_estimated_date(pivot_table)
# Save the GeoDataFrame as a shapefile or another format
fn_pivot_table = str(my_folder / output_folder / ('pivot_table_' + str(id_gleba) + '.csv'))
pivot_table.to_csv(fn_pivot_table)





#### ASSUMINDO QUE USAMOS PARES DE DATES COM PERIODO DE 2 ANOS ENTRE START E END DATE
# Variables
id_gleba = '50445-T001_EG'
id_gleba = '50550-T001_EG'
cloud_percentage = 10

# for a single id_gleba
talhao = extract_talhao_from_nvg(fn_nvg, id_gleba)
talhao_singlepart = multi_to_singlepart(talhao)
talhao_singlepart_pk, talhao_shp, fn_talhao = add_primary_key_talhao(talhao_singlepart)
#get start and end dates
date_pairs = extract_date_pairs(df_sorted, id_gleba)
modifies_date_pairs = dates_with_two_months_diff(date_pairs)
new_start_dates, new_end_dates, modified_date_pairs = dates_with_two_months_diff(date_pairs)
# Initialize GEE
ee.Initialize()
# add talhao to feature collection
nvg = geemap.shp_to_ee(fn_talhao)
for i, (start_date, end_date) in enumerate(modified_date_pairs):
    #calculate median values for each subtalhao
    csv_path = calculate_median_from_gee(fn_talhao, start_date, end_date, cloud_percentage, id_gleba, i) 
    # Read CSV with pandas DataFrame
    df_median_ndvi = pd.read_csv(csv_path)
    #estimate clear cuts dates
    pivot_table = convert_to_pivot_table(df_median_ndvi)
    calculate_biggest_ndvi_drop_and_estimated_date(pivot_table)
    # Save the GeoDataFrame as a shapefile or another format
    fn_pivot_table = str(my_folder / output_folder / ('pivot_table_' + str(id_gleba)+ '_'+ str(i) + '.csv'))
    pivot_table.to_csv(fn_pivot_table)
