# PyCCD v01 - 24/02/2025 / Contrato N.ยบ 3044 DGT/ISA/CEXC/2152/2023
#%% Imports and Path Setup
# Standard library imports
import os
import sys
import platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import warnings

# Set up path for custom modules
if platform.system() == "Windows":
    user_profile = os.environ['USERPROFILE']
    directory_path = os.path.join(user_profile, 'Desktop', 'S2CHANGE')
else:  # Linux
    user_home = os.path.expanduser("~")
    directory_path = os.path.join(user_home, 'CCD_yml_win')
os.chdir(directory_path)

# PyCCD script folder path
PASTA_DE_SCRIPTS = Path(__name__).parent.absolute() / 'scripts' / 'pyccd'
if PASTA_DE_SCRIPTS not in sys.path:
    sys.path.append(str(PASTA_DE_SCRIPTS))

# Third-party libraries
import pandas as pd
import h5py
from tqdm import tqdm

# PyCCD module imports
from shared.processing import runDetectionForPoint, explode_columns
from shared.preprocessing import check_or_initialize_file

# Configurations
from config.config import input_config, preprocessing_config, outputs_config, ccd_config

# Suppress warnings
warnings.filterwarnings('ignore')
#%%

def main(batch_size):
    # Verificar a existência do arquivo .npy e inicializar ou carregar os dados
    # Create npy/h5 file if "output_file doesn't exist
    # obs: calculation of raster_path should be included in check_or_initialize_file
    tif_dates_ord, N = check_or_initialize_file(
        outputs_config['output_file'], 
        input_config['tiles'], 
        input_config['data_source_folder'], 
        input_config['s2_tile_folder'], 
        preprocessing_config['min_year'], 
        preprocessing_config['max_date'], 
        input_config['roi'], 
        preprocessing_config['bandas_desejadas']
        )
    
    # Carregar os dados numpy para o processamento em lotes
    h5_file = h5py.File(outputs_config['output_file'], 'r')
    sel_values = h5_file['values']
    xs = h5_file['xs']
    ys = h5_file['ys']
    
    # Criar os batches # N is the total number of pixels; 
    batches = [
        (sel_values[:, :, start:end], xs[start:end], ys[start:end], tif_dates_ord)
        for start, end in zip(range(0, N, batch_size), range(batch_size, N + batch_size, batch_size))
    ]
    
    
    dfs = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        tqdm_bar = tqdm(total=len(batches))
        
        # Processar os batches paralelamente
        # function process_batch defined below
        for batch_results in executor.map(process_batch, batches):
            dfs.extend(batch_results)  # Adiciona os resultados processados; dataframe or list of dataframes
            tqdm_bar.update(1)
        
        tqdm_bar.close()
    
    # Concatenar os resultados de todos os lotes num único DataFrame  // dataframe could be problematic 
    # try to write this as a fucntion with imnput: dfs; output: parquet file path
    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)
        result_df = explode_columns(result_df)
        print(f"Saving the parquet file with {len(result_df)} records.")
        result_df.to_parquet(outputs_config['folders']['tabular'] / '{}.parquet'.format(ccd_config['filename']), index=False)
    
    #if EXECUTAR_PLOT:
    #    plotFromCSV(FOLDER_PARQUET / '{}.parquet'.format(filename), ROW_INDEX, save_dir=FOLDER_PLOTS / '{}_RowIndex{}.png'.format(filename, ROW_INDEX))


# Função auxiliar para processar um batch
def process_batch(args):
    """
    Processes a batch of points for change detection.

    This function takes a batch of selected values and processes each point in the batch 
    by calling the `runDetectionForPoint` function. It creates a list of arguments for each 
    point in the batch and returns the results for all points.

    Args:
        - args (tuple): A tuple containing the following elements:
            - i (int): Index of the point being processed within the batch. Used to iterate through each point in the batch.
            - sel_values_block (ndarray): 3D array of selected values for the batch (number of images x number of bands x batch_size).
            - xs_slice (ndarray): Array of x coordinates for the batch.
            - ys_slice (ndarray): Array of y coordinates for the batch.
            - tif_dates_ord (ndarray): Array of ordered dates corresponding to the images.
            - NODATA_VALUE (float): Value representing missing data in the dataset.
            - MAX_VALUE_NDVI (float): Maximum possible NDVI value used for scaling.
            - PASTA_DE_OUTPUTS (str): Directory to save the output results.
            - CRS_THEIA (str): Coordinate Reference System (CRS) for the Theia dataset.
            - CRS_WGS84 (str): Coordinate Reference System (CRS) for the WGS84 standard.
            - img_collection (ndarray): Collection of Sentinel-2 images.

    Returns:
        - list: A list of results from `runDetectionForPoint` for each point in the batch.
    """
    sel_values_block, xs_slice, ys_slice, tif_dates_ord = args
    arg_list = [
        (i, sel_values_block, tif_dates_ord, xs_slice, ys_slice, preprocessing_config['nodata_value'],
         preprocessing_config['max_value_ndvi'], outputs_config['output_path'], preprocessing_config['crs_theia'], 
         preprocessing_config['wgs84'], preprocessing_config['img_collection'])
        for i in range(sel_values_block.shape[2])]
    # Retorna resultados para todos os pontos no batch
    return [runDetectionForPoint(arg) for arg in arg_list]
#%%

#%%
# Executar o código
if __name__ == '__main__':
    main(preprocessing_config['batch_size'])
