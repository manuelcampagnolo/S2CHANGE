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
import glob


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

# ####### GOOGLE EARTH ENGINE

# #Inputs
# #nvg_norm
# ln_nvg='nvg_norm.shp'
# fn_nvg = str(my_folder/output_folder/ln_nvg)
# #final_df_sorted
# ln_df_sorted = 'final_df_sorted_no_spaces.csv'
# fn_df_sorted = str(my_folder/output_folder/ln_df_sorted)
# df_sorted = pd.read_csv(fn_df_sorted)


# # Variables
# id_gleba = '50445-T001_EG'
# # id_gleba = '50550-T001_EG'
# # id_gleba = '53010-T001_EG'
# id_gleba = '50307-T002_EG'
# # id_gleba = '50161-T001_EG'
# cloud_percentage = 50


# # 1st step: is to create a singlepart talhao of the id_gleba we want 

# ln_talhao_singlepart = 'nvg_singlepart_' + id_gleba + '.shp'
# fn_talhao_singlepart = str(my_folder / output_folder / ln_talhao_singlepart)

# if not os.path.exists(fn_talhao_singlepart):
#     talhao = extract_talhao_from_nvg(fn_nvg, id_gleba)
#     talhao_singlepart = multi_to_singlepart(talhao)
#     talhao_singlepart_pk, talhao_singlepart_shp, fn_talhao_singlepart = add_primary_key_talhao(talhao_singlepart)

# # 2nd step: get start and end date of that same talhao

# date_pairs = find_date_pairs(df_sorted, id_gleba)
# new_start_dates, new_end_dates, modified_date_pairs = dates_with_two_months_diff(date_pairs)

# # 3rd step: get a csv file with median NDVI values from Google Earth Engine

# ee.Initialize()
# nvg = geemap.shp_to_ee(fn_talhao_singlepart, encoding='latin-1') #encoding para nao ter erro

# for i, (start_date, end_date) in enumerate(modified_date_pairs):
#     medianNDVI = ndvi_median_gee(start_date, end_date, nvg, cloud_percentage)

#     #medianNDVI = ndvi_median_gee_masks2clouds(start_date, end_date, nvg, cloud_percentage)
#     medianNDVIWithProperties = medianNDVI.map(map_features)
    
#     # # Get the first element to inspect properties
#     # sample_feature = medianNDVIWithProperties.first()
    
#     # # Print the properties of the sample feature
#     # print(sample_feature.getInfo())
#     output_dir = str(my_folder/ndvi_folder)


#     # Export the result as a CSV file using geemap
#     geemap.ee_to_csv(
#         ee_object=medianNDVIWithProperties,
#         filename=os.path.join(output_dir, f'Median_NDVI_{id_gleba}_{i}.csv')
#     )

#     # get csv file
#     ln_median_ndvi = f'Median_NDVI_{id_gleba}_{i}.csv'
#     csv_path = str(my_folder/ndvi_folder/ln_median_ndvi)


#     # Read CSV with pandas DataFrame
#     df_median_ndvi = pd.read_csv(csv_path, header=0)
#     #estimate clear cuts dates
#     pivot_table = convert_to_pivot_table(df_median_ndvi)
#     pivot_table_with_estimated_date = calculate_biggest_ndvi_drop_and_estimated_date(df_sorted, id_gleba, pivot_table)
#     # pivot_table_with_estimated_date = calculate_biggest_ndvi_drop_and_estimated_date(pivot_table)
#     # Save the GeoDataFrame as a shapefile or another format
#     # fn_pivot_table = str(my_folder / output_folder / ('pivot_table_' + str(id_gleba) + '.csv'))
#     fn_pivot_table = str(my_folder / output_folder / (f'pivot_table_{id_gleba}_{i}.csv'))
#     pivot_table_with_estimated_date.to_csv(fn_pivot_table)


# ## Label the parcel
# # join 'estimated_date' to shapefile and add layer to the project
# result = join_attribute_to_layer(fn_talhao_singlepart, 'id', fn_pivot_table, 'id', 'estimated_date')
# result.setName('talhao_'+ id_gleba)
# QgsProject().instance().addMapLayer(result)
 

# # set labels according to estimated clear-cut dates 
# layer = iface.activeLayer()
# field_name = 'estimated_date'
# field_index = layer.fields().indexFromName(field_name)
# unique_values = layer.uniqueValues(field_index)

# category_list = []
# for value in unique_values:
#     symbol = QgsSymbol.defaultSymbol(layer.geometryType())
#     category = QgsRendererCategory(value, symbol, str(value))
#     category_list.append(category)

# # color ramp
# ramp_name = 'Greens'
# default_style = QgsStyle().defaultStyle()
# color_ramp = default_style.colorRamp(ramp_name)
# renderer = QgsCategorizedSymbolRenderer(field_name, category_list)
# renderer.updateColorRamp(color_ramp)
# layer.setRenderer(renderer)
# layer.triggerRepaint()


# # 5th step: create a dataset with columns of 'data_estimada'

# ln_nvg_singlepart = 'nvg_singlepart.shp'
# fn_nvg_singlepart = str(my_folder/output_folder/ln_nvg_singlepart)
# # read nvg_singlepart as gdf
# gdf_nvg_singlepart = gpd.read_file(fn_nvg_singlepart)


# fn_expanded = (my_folder / output_folder / 'df_expanded.csv')
# # expanded_df.to_csv(fn_expanded)

# if not os.path.exists(fn_expanded):
#     expanded_df = create_expanded_df(df_sorted, gdf_nvg_singlepart, 'id_gleba', 'id', 'id_gleba', 'left')

#     # Save the GeoDataFrame as a shapefile or another format
#     fn_expanded = (my_folder / output_folder / 'df_expanded.csv')
#     expanded_df.to_csv(fn_expanded)


# # 6th step: fill the data_estimada columns

# # fn_pivot_table = str(my_folder / output_folder / ('pivot_table_' + str(id_gleba) + '.csv'))
# fn_pivot_table = str(my_folder / output_folder / (f'pivot_table_{id_gleba}_{i}.csv'))
# pivot_table = pd.read_csv(fn_pivot_table)

# ##### funciona pd.DataFrame.at !!!!!
# expanded_df = pd.read_csv(fn_expanded)
# # Iterate through each row of pivot_table
# for index, row in pivot_table.iterrows():
#     # Extract id and estimated_date
#     id_value = row['id']
#     estimated_date = row['estimated_date']
    
#     # Find matching row in expanded_df based on id
#     matching_row_index = expanded_df[expanded_df['id'] == id_value].index[0]
    
#     # Initialize a set to store columns already updated for this row
#     updated_columns = set()
    
#     # Iterate through data_estimada columns to find the matching date
#     for col in expanded_df.columns:
#         # Check if it's a 'data_estimada' column and not already updated
#         if col.startswith('data_estimada') and col not in updated_columns:
#             # Extract the corresponding data column index
#             data_index = int(col.split('data_estimada')[1])  # Extract the index after 'data_estimada'
#             data_col = 'data' + str(data_index)  # Form the corresponding 'data' column name
            
#             # Check if the data value matches the estimated_date
#             if expanded_df.at[matching_row_index, data_col] == estimated_date:
#                 # Update corresponding data_estimada column with 1
#                 expanded_df.at[matching_row_index, col] = 1
#                 # Add the updated column to the set
#                 updated_columns.add(col)


# # Save the GeoDataFrame as a shapefile or another format
# fn_expanded = (my_folder / output_folder / 'df_expanded_updated.csv')
# expanded_df.to_csv(fn_expanded)


# ######################################################################
# #Create expanded_df
# fn_expanded = my_folder / output_folder / 'df_expanded.csv'

# if not os.path.exists(fn_expanded):
#     expanded_df = create_expanded_df(df_sorted, gdf_nvg_singlepart, 'id_gleba', 'id', 'id_gleba', 'left')
#     expanded_df.to_csv(fn_expanded)
#     #read as pd 
#     expanded_df = pd.read_csv(fn_expanded)
# else:
#     # Load the existing expanded_df from the CSV file
#     expanded_df = pd.read_csv(fn_expanded)

# id_gleba_list = extract_unique_id_gleba_from_nvg(fn_nvg)
# id_gleba_dates = {}

# for id_gleba in id_gleba_list:
#     date_pairs = filter_and_select_dates1(df_sorted, id_gleba)
    
#     # Check if date_pairs is not empty
#     if date_pairs:
#         id_gleba_dates[id_gleba] = date_pairs
#     else:
#         pass #print(f"No 'CORTE' activity found for ID {id_gleba}")

# print(len(id_gleba_dates))
# print(len(id_gleba_list))


# ## loop for all
# cloud_percentage = 75
# id_gleba_list = ['50211-T001_EG','50445-T002_EG','50445-T001_EG','50550-T001_EG','50002-T001_EG']
# id_gleba_list = list(id_gleba_dates)
# print(id_gleba_list)


# fn_exp_cortes = str(my_folder / output_folder / 'pivot_table_exp_cortes_2015.csv')
# df_cortes = pd.read_csv(fn_exp_cortes)
# id_gleba_list = df_cortes['id_gleba'].tolist()
# unique_id_gleba_count = df_cortes['id_gleba'].nunique()

# fn_tile = str(my_folder / output_folder / 'nvg_tile.shp')
# gdf_tile = gpd.read_file(fn_tile)
# id_gleba_list = gdf_tile['id_gleba'].tolist()
# unique_id_gleba_count = gdf_tile['id_gleba'].nunique()


# # Initialize Earth Engine
# ee.Initialize()

for id_gleba in id_gleba_list:
    ln_talhao_singlepart = 'nvg_singlepart_' + id_gleba + '.shp'
    fn_talhao_singlepart = str(my_folder / output_folder / ln_talhao_singlepart)

    if not os.path.exists(fn_talhao_singlepart):
        talhao = extract_talhao_from_nvg(fn_nvg, id_gleba)
        talhao_singlepart = multi_to_singlepart(talhao)
        talhao_singlepart_pk, talhao_shp, fn_talhao = add_primary_key_talhao(talhao_singlepart)

    #get start and end dates
    date_pairs = find_date_pairs(df_sorted, id_gleba)
    if not date_pairs:
        print(f"no date pairs found for {id_gleba}")
        continue
    new_start_dates, new_end_dates, modified_date_pairs = dates_with_two_months_diff(date_pairs)
    # add talhao to feature collection
    nvg = geemap.shp_to_ee(fn_talhao_singlepart, encoding = 'latin-1')
    for i, (start_date, end_date) in enumerate(modified_date_pairs):

        medianNDVI = ndvi_median_gee_masks2clouds(start_date, end_date, nvg, cloud_percentage)
        medianNDVIWithProperties = medianNDVI.map(map_features)

        output_dir = str(my_folder / ndvi_folder)
        # Export the result as a CSV file using geemap
        geemap.ee_to_csv(
            ee_object=medianNDVIWithProperties,
            # filename=os.path.join(output_dir, 'Median_NDVI_' + id_gleba + '.csv')
            filename=os.path.join(output_dir, f'Median_NDVI_{id_gleba}_{i}.csv')
        )
        # Get CSV file path
        # ln_median_ndvi = 'Median_NDVI_' + id_gleba + '.csv'
        ln_median_ndvi = f'Median_NDVI_{id_gleba}_{i}.csv'
        csv_path = str(my_folder / ndvi_folder / ln_median_ndvi)
        if not os.path.exists(csv_path):
            print(f'no data for {id_gleba}')
            continue
        # Read CSV with pandas DataFrame
        df_median_ndvi = pd.read_csv(csv_path, header=0)

        #estimate clear cuts dates
        pivot_table = convert_to_pivot_table(df_median_ndvi)
        pivot_table_with_estimated_date = calculate_biggest_ndvi_drop_and_estimated_date(df_sorted, id_gleba, pivot_table)
        # Save the GeoDataFrame as a shapefile or another format
        # fn_pivot_table = str(my_folder / output_folder / ('pivot_table_' + str(id_gleba) + '.csv'))
        fn_pivot_table = str(my_folder / output_folder / (f'pivot_table_{id_gleba}_{i}.csv'))
        pivot_table_with_estimated_date.to_csv(fn_pivot_table)

        # Read the pivot table CSV file
        pivot_table = pd.read_csv(fn_pivot_table)
            
        # Iterate through each row of pivot_table
        for index, row in pivot_table.iterrows():
            # Extract id and estimated_date
            id_value = row['id']
            estimated_date = row['estimated_date']
            
            # Find matching row in expanded_df based on id
            matching_row_index = expanded_df[expanded_df['id'] == id_value].index[0]
            
            # Initialize a set to store columns already updated for this row
            updated_columns = set()
            
            # Iterate through data_estimada columns to find the matching date
            for col in expanded_df.columns:
                # Check if it's a 'data_estimada' column and not already updated
                if col.startswith('data_estimada') and col not in updated_columns:
                    # Extract the corresponding data column index
                    data_index = int(col.split('data_estimada')[1])  # Extract the index after 'data_estimada'
                    data_col = 'data' + str(data_index)  # Form the corresponding 'data' column name
                    
                    # Check if the data value matches the estimated_date
                    if expanded_df.at[matching_row_index, data_col] == estimated_date:
                        # Update corresponding data_estimada column with 1
                        expanded_df.at[matching_row_index, col] = 1
                        # Add the updated column to the set
                        updated_columns.add(col)

# # Save expanded_df as expanded_df_updated
# expanded_df_updated = expanded_df.copy()


# # Save the GeoDataFrame as a shapefile or another format
# fn_expanded = (my_folder / output_folder / 'df_expanded_updated_all.csv')
# expanded_df_updated.to_csv(fn_expanded)






### TILE 29TNE  


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
df_ndvi.columns

filtered_df_ndvi = df_ndvi.groupby('id_gleba').filter(lambda x: not x['median'].isna().all())
fn_fdf = str(my_folder / ndvi_folder / 'Median_NDVI_filtered.csv')
filtered_df_ndvi.to_csv(fn_fdf)

id_gleba_list_f = extract_unique_id_gleba_from_nvg(fn_fdf, 'id')
id_gleba_list = extract_unique_id_gleba_from_nvg(fn_csv, 'id')
print(len(id_gleba_list_f))
print(len(id_gleba_list))


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
new_gdf['first_start_date'] = np.nan
new_gdf['first_end_date'] = np.nan

for index, row in new_gdf.iterrows():
    id_gleba = row['id_gleba']
    date_pairs = find_date_pairs(df_sorted, id_gleba)
    
    if date_pairs:
        first_start_date, first_end_date = date_pairs[0]
        new_gdf.at[index, 'first_start_date'] = first_start_date
        new_gdf.at[index, 'first_end_date'] = first_end_date

new_gdf['start_date'] = np.nan
new_gdf['end_date'] = np.nan

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
nvg_dates_cuts.to_csv(fn_merged_df)

#join total area of id_gleba
result = join_attribute_to_layer(fn_nvg_dates_and_cuts, 'id_gleba', fn_nvg, 'id_gleba', 'area_ha')
#save
ln_nvg_dates_cuts_and_area = 'nvg_with_dates_cuts_and_area.csv'
fn_nvg_dates_cuts_and_area = str(my_folder/output_folder/ln_nvg_dates_cuts_and_area)
result.selectAll()
result = processing.run("native:saveselectedfeatures", {'INPUT':result, 'OUTPUT':fn_nvg_dates_cuts_and_area})

# calcular a area de cada talho por numero de cortes
df_dates_areas = pd.read_csv(fn_nvg_dates_cuts_and_area)

df_dates_areas['area_per_cuts'] = df_dates_areas['area_ha'] / df_dates_areas['nr_clear_cuts']
fn_csv_dates = str(my_folder / output_folder / 'nvg_with_dates_cuts_and_area.csv')
df_dates_areas.to_csv(fn_csv_dates)


# select rows where 'first_start_data' is not null
filtered_df = df_dates_areas[df_dates_areas['start_date'].notnull()]
#save
ln_nvg_dates_filtered = 'nvg_with_dates_filtered.csv'
fn_nvg_dates_filtered = str(my_folder/output_folder/ln_nvg_dates_filtered)
filtered_gdf.to_csv(fn_nvg_dates_filtered)



## TILE 29TNE

fn_nvg_singlepart = str(my_folder/output_folder/'nvg_singlepart.shp')

#extract by location all sub-parcels within tile 29TNE (id_gleba and id)
parcels_tile = processing.run("native:extractbylocation", 
 {'INPUT':fn_nvg_singlepart,
 'PREDICATE':[6],
 'INTERSECT':fn_tile,
 'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
QgsProject().instance().addMapLayer(parcels_tile)

# select all features of layer to export
ln_glebas_tile = 'id_glebas_tile.shp'
fn_glebas_tile = str(my_folder/tile29_folder/ln_glebas_tile)
parcels_tile.selectAll()
# export layer
parcels_tile = processing.run("native:saveselectedfeatures", {'INPUT':parcels_tile, 'OUTPUT':fn_glebas_tile})

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



# retrieve id_glebas where area/nr cuts is more than 0.5ha
less_than_05_area_per_cuts_count = (df_dates_areas['area_per_cuts'] < 0.5).sum()
less_than_05_df = df_dates_areas[df_dates_areas['area_per_cuts'] < 0.5]
id_glebas_list_less_than_05_ha = less_than_05_df['id_gleba'].tolist()
id_glebas_list_less_than_05_ha_to_save = pd.DataFrame(id_glebas_list_less_than_05_ha, columns=['id_gleba'])

fn = str(my_folder/tile29_folder/'id_gleba_list_area_per_cuts_higher_05_ha.csv')
id_glebas_list_less_than_05_ha_to_save.to_csv(fn)









###
#CREATED PIVOT TABLES FOR TILE 29 IN VSCODE
###
id_gleba = '56025-T002_EG'
ln_talhao_singlepart = 'nvg_singlepart_' + id_gleba + '.shp'
fn_talhao_singlepart = str(my_folder / output_folder / ln_talhao_singlepart)
vscode_folder = 's2change'
fn_pivot_table = str(my_folder / vscode_folder / 'tile_29' /'tile_29'/'pivot_tables'/(f'pivot_table_{id_gleba}.csv'))

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



### id_gleba com area menor a 0.5 ha

fn_merged_df = str(my_folder / output_folder / 'nvg_with_dates_merged.csv')
df = pd.read_csv(fn_merged_df)

fn_nvg_dates = str(my_folder/output_folder/ln_nvg_dates)


result = join_attribute_to_layer(fn_merged_df, 'id_gleba', fn_nvg_dates, 'id_gleba', 'area_ha')
result.setName('nvg_dates_area')
QgsProject().instance().addMapLayer(result)
result.selectAll()
# export layer
ln_csv_dca = 'nvg_with_dates_and_areas.csv'
fn_csv_dca = str(my_folder / output_folder / ln_csv_dca)
result_csv = processing.run("native:saveselectedfeatures", {'INPUT':result, 'OUTPUT':fn_csv_dca})
result.removeSelection()

df_dates_areas = pd.read_csv(fn_csv)

df_dates_areas['area_per_cuts'] = df_dates_areas['area_ha'] / df_dates_areas['count_activities']
fn_csv_dates = str(my_folder / output_folder / 'nvg_with_area_per_cut.csv')
df_dates_areas.to_csv(fn_csv_dates)

# number of rows with area_per_cuts less than 0.5
less_than_05_area_per_cuts_count = (df_dates_areas['area_per_cuts'] < 0.5).sum()
less_than_05_df = df_dates_areas[df_dates_areas['area_per_cuts'] < 0.5]
id_glebas_list_less_than_05_ha = less_than_05_df['id_gleba'].tolist()


fn_id_gleba_tile = str(my_folder/tile29_folder/'id_gleba_list_tile29.csv')
# Read the id_gleba_list_tile29.csv file into a DataFrame
id_gleba_df = pd.read_csv(fn_id_gleba_tile)

# Filter out the id_glebas that belong to id_glebas_list
filtered_id_glebas_df = id_gleba_df[~id_gleba_df['id_gleba'].isin(id_glebas_list)]

# Extract the filtered list of id_glebas
filtered_id_glebas_list = filtered_id_glebas_df['id_gleba'].tolist()
filtered_id_glebas_list_to_save = pd.DataFrame(filtered_id_glebas_list, columns=['id_gleba'])
fn = str(my_folder/tile29_folder/'id_gleba_list_tile29_area_higher_05_ha.csv')
filtered_id_glebas_list_to_save.to_csv(fn)




### LIMITACOES

# JOIN ALL PIVOT TABLES 
fn_folder = str(my_folder / vscode_folder / 'tile_29' /'nvg_dataset'/'all_pivot_tables')
all_files = os.listdir(fn_folder)
csv_files = [f for f in all_files if f.endswith('_count.csv')]
df_list = []
## to delete files, if needed
# files_to_delete = glob.glob(os.path.join(fn_folder, '*count.csv'))
# # Iterate over the list of files and delete them
# for file_path in files_to_delete:
#     try:
#         os.remove(file_path)
#         print(f"Deleted: {file_path}")
#     except Exception as e:
#         print(f"Error deleting {file_path}: {e}")

# create new files with columns with count number of all S2 images, empty and non empty NDVI values
for csv_file in csv_files:
    # Construct the full file path
    file_path = os.path.join(fn_folder, csv_file)
    df = pd.read_csv(file_path)

    date_columns = [col for col in df.columns if col.startswith('date_20')]
    
    # count the number of empty and non-empty cells for each row
    df['nr_empty_cells'] = df[date_columns].isna().sum(axis=1)
    df['nr_non_empty_cells'] = df[date_columns].notna().sum(axis=1)
    df['nr_s2_dates'] = df['nr_empty_cells'] + df['nr_non_empty_cells']
    
    new_file_name = os.path.splitext(csv_file)[0] + '_count.csv'
    new_file_path = os.path.join(fn_folder, new_file_name)
    # save
    df.to_csv(new_file_path, index=False)

print("files have been saved with '_count' suffix.")

for csv in csv_files:
    file_path = os.path.join(fn_folder, csv)
    df = pd.read_csv(file_path)
    df_list.append(df[['biggest_drop_NDVI','date_of_biggest_drop', 'estimated_date','nr_empty_cells', 'nr_non_empty_cells','nr_s2_dates']])

all_pivot_tables = pd.concat(df_list, ignore_index=True)
date_columns = [col for col in all_pivot_tables.columns if col.startswith('date_20')]
all_pivot_tables = all_pivot_tables.drop(columns=date_columns)
#save
ln_all_pivot_tables = 'all_pivot_tables.csv'
fn_all_pivot_tables = str(my_folder / output_folder / ln_all_pivot_tables)
all_pivot_tables.to_csv(fn_all_pivot_tables)
#join attributes (dates, areas and nr of clear cuts) to the joined pivot tables from 
result = join_attribute_to_layer(fn_all_pivot_tables, 'id_gleba', fn_csv_dates, 'id_gleba', ['area_ha','start_date','end_date','date_difference_days','nr_clear_cuts'])
result.setName('ALL')
QgsProject().instance().addMapLayer(result)
result.selectAll()
# export layer
ln_nvg_pt = 'all_nvg_pivot_tables.csv'
fn_nvg_pt = str(my_folder / output_folder / ln_nvg_pt)
result_layer = processing.run("native:saveselectedfeatures", {'INPUT':result, 'OUTPUT':fn_nvg_pt})
result.removeSelection()

# update df
df_pt = pd.read_csv(fn_csv)
df_pt.dropna(subset=['id'], inplace=True)
df_pt.to_csv(fn_csv)



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

fn_s2_mean = str(my_folder/output_folder/'id_gleba_s2_mean_month.csv')
join_df.to_csv(fn_s2_mean, index=False)


# Create a scatterplot
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
scatter = plt.scatter(join_df['nr_clear_cuts'], join_df['nr_s2_dates'], c=join_df['mean_month'], cmap='hsv', alpha=0.7)

# Set labels and title
plt.xlabel('Nr Clear Cuts')
plt.ylabel('Nr S2 Dates')
plt.title('Scatterplot of Nr S2 Dates vs. Nr Clear Cuts')

# Set limits for x-axis and y-axis
plt.xlim(0, 20)
plt.ylim(0, 60)

# Add colorbar
plt.colorbar(scatter, ticks=range(1, 13), label='Mean Month')

# Display the plot
plt.grid(True)
plt.show()


### CALCULAR AREA DOS SUB-TALHOES

ln_nvg_singlepart = 'nvg_singlepart.shp'
fn_nvg_singlepart = str(my_folder/output_folder/ln_nvg_singlepart)

layer = QgsVectorLayer(fn_nvg_singlepart, 'My Layer', 'ogr')

area_singlepart = processing.run("native:addfieldtoattributestable", 
 {'INPUT':layer,
 'FIELD_NAME':'area_ha_subtalhao',
 'FIELD_TYPE':1,
 'FIELD_LENGTH':5,
 'FIELD_PRECISION':2,
 'FIELD_ALIAS':'',
 'FIELD_COMMENT':'',
 'OUTPUT':'TEMPORARY_OUTPUT'})

output_layer = area_singlepart['OUTPUT']

# Ensure the output layer is added to the QGIS project
if output_layer not in QgsProject.instance().mapLayers().values():
    QgsProject.instance().addMapLayer(output_layer)

# Calculate the area of the polygons and update the new field
if output_layer.isEditable() or output_layer.startEditing():
    for feature in output_layer.getFeatures():
        geom = feature.geometry()
        area = geom.area() / 10000  # Calculate the area in hectares (assuming CRS is in meters)
        feature['area_ha_subtalhao'] = area
        output_layer.updateFeature(feature)
    
    # Commit the changes
    output_layer.commitChanges()

fn_output_layer = str(my_folder/output_folder/'nvg_singlepart_area.shp')
save = QgsVectorFileWriter.writeAsVectorFormat(output_layer, fn_output_layer, "UTF-8", output_layer.crs(), "ESRI Shapefile")


### CRIAR TABELAS DE ATRIBUTO AO NIVEL DO TALHAO E SUB-TALHAO

#NEW FOLDER
new_folder_name = 'entregavel2_2'
new_folder_path = my_folder / new_folder_name
new_folder_path.mkdir(parents=True, exist_ok=True)
entregavel_folder = 'entregavel2_2'

# AO NIVEL DO TALHAO
df_nvg_select = df_nvg[['id_gleba', 'nr_s2_dates',
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
df_nvg_select_id = df_nvg[['id_gleba', 'id',
       'first_start_date', 'first_end_date', 'date_of_biggest_drop','estimated_date', 'nr_empty_cells',
       'nr_non_empty_cells', 'nr_s2_dates']]

df_nvg_select_id = df_nvg_select_id.rename(columns={'estimated_date': 'first_estimated_date'})

fn_sub_talhao = str(my_folder/entregavel_folder/'nvg_sub_talhao.csv')
df_nvg_select_id.to_csv(fn_sub_talhao, index=False)

result =  processing.run("native:joinattributestable", 
{'INPUT':fn_sub_talhao,
'FIELD':'id',
'INPUT_2':fn_output_layer,
'FIELD_2':'id',
'FIELDS_TO_COPY':['area_ha_su'],
'METHOD':1,
'DISCARD_NONMATCHING':False,
'PREFIX':'',
'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']


# result = join_attribute_to_layer(fn_sub_talhao, 'id', fn_output_layer, 'id', ['area_ha_sub'])
result.setName('sub-talhoes com area')
QgsProject().instance().addMapLayer(result)
result.selectAll()
# export layer
result_csv = processing.run("native:saveselectedfeatures", {'INPUT':result, 'OUTPUT':fn_sub_talhao})
result.removeSelection()





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

# apply function to uodate estimated date
df_subtalhao = df_subtalhao.apply(update_first_estimated_date, axis=1)
df_subtalhao.to_csv(fn_sub_talhao, index=False)


# Number of sub-parcels
num_subparcels = df_subtalhao.shape[0]
print("Number of sub-parcels:", num_subparcels)

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
even_condition = (df_subtalhao['nr_s2_dates'] % 2 == 0) & (df_subtalhao['nr_empty_cells'] >= df_subtalhao['nr_non_empty_cells'])
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

nr_total_s2 = df_subtalhao['nr_s2_dates'].sum()
nr_total_empty_cells = df_subtalhao['nr_empty_cells'].sum()
perc_empty_cells = (nr_total_empty_cells*100)/nr_total_s2
nr_total_non_empty_cells = df_subtalhao['nr_non_empty_cells'].sum()
perc_non_empty_cells = (nr_total_non_empty_cells*100)/nr_total_s2

print("Total number of S2 images:", nr_total_s2)
print("Total number of S2 images with no NDVI values:", nr_total_empty_cells)
print("Percentage of S2 images with no NDVI values:", perc_empty_cells)
print("Total number of S2 images with NDVI values:", nr_total_non_empty_cells)
print("Percentage of S2 images with NDVI values:", perc_non_empty_cells)

# mean_month
df_subtalhao2 = df_nvg_select_id[['id_gleba','id','nr_empty_cells','nr_non_empty_cells','nr_s2_dates']]
df_sub_talhao_month = join_df[['id_gleba', 'mean_month']]

df_st_month = pd.merge(df_subtalhao2, df_sub_talhao_month, on='id_gleba')
df_st_month.drop_duplicates(inplace=True)

fn_st_month = str(my_folder/output_folder/'id_gleba_st_month.csv')
df_st_month.to_csv(fn_st_month, index=False)
df_st_month.columns


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
plt.ylabel('Number of Features')
plt.ylim(0, 50)
plt.xlim(0, 50)
plt.title('Scatter Plot of Number of Features vs Number of Clear Cuts')
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
##### colored by date differences


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












