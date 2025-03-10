import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pathlib import Path
from datetime import datetime
import ccd
from mpi4py import MPI
from shared.utils import fromParamsReturnName

#%% Source variables
data_source_folder = 'GEE' # choose variable: THEIA or GEE
s2_tile_folder = 'T29SPB' # choose S2 tile
roi_filename = 'NAV' # choose variable: DGT or NAV

# Base path
data_path = Path('/projects/F202410004CPCAA1/')

#%% Inputs
input_config = {
    'data_source_folder': data_source_folder,
    's2_tile_folder': s2_tile_folder,
    'data_path': data_path,
    'roi': data_path / 'CCDC_Mask_dissolve.gpkg',
    'theia_images': data_path / 'imagens_Theia',
    'gee_images': data_path / 's2_images'
}

# Determine tile path based on selected data source
input_config['tiles'] = (
    input_config['theia_images'] / s2_tile_folder 
    if data_source_folder == 'THEIA' 
    else input_config['gee_images'] / s2_tile_folder
)

#%% Pre-processing
input_bands = ['B3', 'B4', 'B8', 'B12']
bands_dict = {1:'B3', 2:'B4', 3:'B8', 4:'B12'} # NDVI band is only added later in the processPointData function
bandas_desejadas = list(bands_dict.keys())

preprocessing_config = {
    'min_year': 2017, # CCD start year
    'max_date': datetime(2024, 12, 31), # CCD end date
    'input_bands': ['B3', 'B4', 'B8', 'B12'],
    'bandas_desejadas': bandas_desejadas,
    'nodata_value': 65535,
    'max_value_ndvi': 10000,
    'execute_plot': False, # False will not execute plot
    'row_index': 8, # Chooses the CSV row to plot to
    'batch_size': 1000, 
    'img_collection': input_config['tiles'].parts[-2],
    'crs_theia': 32629,
    'wgs84': 4326
}

#%% Outputs
# Output paths
outputs_config = {
    'output_path': Path(f'{data_path}/outputs_ROI'),
    'folders': {}
}

# Create output subfolders
def create_directory_if_not_exists(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {path}")

for folder_type in ['numpy', 'plots', 'tabular', 'shapefiles']:
    outputs_config['folders'][folder_type] = outputs_config['output_path'] / folder_type / s2_tile_folder
    create_directory_if_not_exists(outputs_config['folders'][folder_type])

#%% CCD/PyCCD Parameters
ccd_config = {
    'parameters': ccd.parameters.defaults,
    'alpha': ccd.parameters.defaults['ALPHA']
}

# Build filename
filename = fromParamsReturnName(
    preprocessing_config['img_collection'], 
    ccd_config['parameters'], 
    (s2_tile_folder, input_config['tiles']), 
    roi_filename, 
    preprocessing_config['min_year'], 
    preprocessing_config['max_date']
)
ccd_config['filename'] = filename

# Set output file path
outputs_config['output_file'] = outputs_config['folders']['numpy'] / f"{filename}.h5"
