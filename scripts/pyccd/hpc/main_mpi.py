# -*- coding: utf-8 -*-
#%% Imports and Path Setup
# Standard library imports
import os
import sys
import platform
from pathlib import Path
import warnings
import time

# Third-party libraries
import pandas as pd
import h5py
from mpi4py import MPI

# PyCCD module imports
sys.path.append(str(Path(__file__).parents[1]))
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
        
        N=100000 # Number of pixels to be processed
        
        indices = list(range(0, N, batch_size))
        batches = [
            (start, min(start + batch_size, N)) for start in indices
        ]
        print(f"[Rank {rank}] Total batches created: {len(batches)}")

        # Split batches between ranks
        batches_per_rank = [batches[i::size] for i in range(size)]
    else:
        tif_dates_ord = None
        batches_per_rank = None

    # Share data between processes
    tif_dates_ord = comm.bcast(tif_dates_ord, root=0)
    my_batches = comm.scatter(batches_per_rank, root=0)

    start_time = time.time()

    local_results = []
    total_batches = len(my_batches)

    for i, batch in enumerate(my_batches):
        result = process_batch(batch, outputs_config['output_file'], tif_dates_ord, rank)
        local_results.extend(result)

        elapsed_time = time.time() - start_time
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60

        print(f"[Rank {rank}] Processed {i+1}/{total_batches} batches "
              f"({(i+1)/total_batches*100:.2f}%) - Elapsed Time: {int(minutes)}m {int(seconds)}s")

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
