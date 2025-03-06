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
DATA_SOURCE = 'GEE' # choose variable: THEIA or GEE
S2_TILE = 'T29SPB' # choose S2 tile
ROI = 'NAV' # choose variable: DGT or NAV

# Base path
data_path = Path('/projects/F202410004CPCAA1/')

#%% Inputs
input_config = {
    'data_source': DATA_SOURCE,
    'roi': ROI,
    's2_tile': S2_TILE,
    'data_path': data_path,
    'validation_path': Path('/home/scaetano/CCDC_Mask_dissolve.gpkg'),
    'theia_images': data_path / 'imagens_Theia',
    'gee_images': data_path / 's2_images'
}

# Determine tile path based on selected data source
input_config['tiles'] = (
    input_config['theia_images'] / S2_TILE 
    if DATA_SOURCE == 'THEIA' 
    else input_config['gee_images'] / S2_TILE
)

#%% Pre-processing
preprocessing_config = {
    'min_year': 2017, # CCD start year
    'max_date': datetime(2024, 12, 31), # CCD end date
    'input_bands': ['B3', 'B4', 'B8', 'B12'],
    'bands_dict': {1: 'NDVI', 2: 'B3', 3: 'B4', 4: 'B8', 5: 'B12'},
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
    'output_path': Path('/projects/F202410004CPCAA1/outputs_RI'),
    'folders': {}
}

# Create output subfolders
def create_directory_if_not_exists(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {path}")

for folder_type in ['numpy', 'plots', 'tabular', 'shapefiles']:
    outputs_config['folders'][folder_type] = outputs_config['output_path'] / folder_type / S2_TILE
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
    (S2_TILE, input_config['tiles']), 
    ROI, 
    preprocessing_config['min_year'], 
    preprocessing_config['max_date']
)
ccd_config['filename'] = filename

# Set output file path
outputs_config['output_file'] = outputs_config['folders']['numpy'] / f"{filename}.h5"
