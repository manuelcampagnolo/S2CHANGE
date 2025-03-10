# -*- coding: utf-8 -*-
#%% Imports and Path Setup
# Standard library imports
import os
import sys
import platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import warnings

# Set up path for custom modules
if platform.system() == "Windows":
    user_profile = os.environ['USERPROFILE']
    directory_path = os.path.join(user_profile, 'Desktop', 'S2CHANGE')
else:  # Linux
    user_home = os.path.expanduser("~")
    directory_path = os.path.join(user_home, 'S2CHANGE')
os.chdir(directory_path)

# PyCCD script folder path
PASTA_DE_SCRIPTS = Path(__name__).parent.absolute() / 'scripts' / 'pyccd'
if PASTA_DE_SCRIPTS not in sys.path:
    sys.path.append(str(PASTA_DE_SCRIPTS))

# Third-party libraries
import pandas as pd
import h5py
from tqdm import tqdm
from mpi4py import MPI

# PyCCD module imports
from shared.processing import runDetectionForPoint, explode_columns
from shared.preprocessing import check_or_initialize_file
from config.config import input_config, preprocessing_config, outputs_config, ccd_config

# Suppress warnings
warnings.filterwarnings('ignore')
#%%
def main(batch_size=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # root rank
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
        
        N=1000000
        
        # Dividir os ÃƒÂ­ndices de dados
        indices = list(range(0, N, batch_size))
        batches = [
            (start, min(start + batch_size, N)) for start in indices
        ]
        print(f"[Rank {rank}] Total de batches criados: {len(batches)}")

        # Dividir lotes entre ranks
        batches_per_rank = [batches[i::size] for i in range(size)]
    else:
        tif_dates_ord = None
        batches_per_rank = None

    # Share data between processes
    tif_dates_ord = comm.bcast(tif_dates_ord, root=0)
    my_batches = comm.scatter(batches_per_rank, root=0)

    # Create shared progress bar
    with Manager() as manager:
        progress = manager.Value('i', 0)  # Shared counter
        lock = manager.Lock()
        total_batches = len(my_batches)  # Total number of lots

        local_results = []

        with tqdm(total=total_batches, desc=f"Processo {rank}") as pbar:
            def update_progress(lock):
                with lock:
                    progress.value += 1
                    pbar.update(1)
        
            for batch in my_batches:
                result = process_batch(batch, outputs_config['output_file'], tif_dates_ord)
                local_results.extend(result)
                update_progress(lock)

    # Saving results locally per process
    if local_results:
        result_df = pd.concat(local_results, ignore_index=True)
        result_df = explode_columns(result_df)
        result_df.to_parquet(outputs_config['folders']['tabular'] / f"{ccd_config['filename']}_rank_{rank}.parquet", index=False)

    # Sync all ranks before continuing
    comm.Barrier()
    
    if rank == 0:
        print(f"All batches were processed by {size} ranks.")

def process_batch(batch, sel_values_path, tif_dates_ord):
    """
    Processes a batch of data from an HDF5 file, extracting the required slices 
    and passing them for further analysis.

    Parameters:
        batch (tuple): A tuple containing the start and end indices of the batch.
        sel_values_path (str): Path to the HDF5 file containing the data.
        tif_dates_ord (list): List of ordered dates associated with the dataset.

    Returns:
        list: A list of processed results from `runDetectionForPoint`.
    """
    start, end = batch
    
    h5_file = h5py.File(sel_values_path, 'r')
    sel_values_block = h5_file['values'][:, :, start:end]
    xs_slice = h5_file['xs'][start:end]
    ys_slice = h5_file['ys'][start:end]

    arg_list = [
        (i, 
        sel_values_block, 
        tif_dates_ord, 
        xs_slice, 
        ys_slice, 
        preprocessing_config['nodata_value'], 
        preprocessing_config['max_value_ndvi'], 
        outputs_config['output_path'], 
        preprocessing_config['crs_theia'], 
        preprocessing_config['wgs84'], 
        preprocessing_config['img_collection'])
        for i in range(sel_values_block.shape[2])
    ]

    return [runDetectionForPoint(arg) for arg in arg_list]

if __name__ == '__main__':
    main(preprocessing_config['batch_size'])
