# -*- coding: utf-8 -*-
#%% Imports and Path Setup
# Standard library imports
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK"))
from concurrent.futures import ProcessPoolExecutor
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
        
        indices = list(range(0, N, batch_size))
        batches = [(start, min(start + batch_size, N)) for start in indices]
        print(f"[Rank {rank}] Total batches created: {len(batches)}", flush=True)

        batches_per_rank = [batches[i::size] for i in range(size)]
    else:
        tif_dates_ord = None
        batches_per_rank = None

    tif_dates_ord = comm.bcast(tif_dates_ord, root=0)
    my_batches = comm.scatter(batches_per_rank, root=0)
    print(f"[Rank {rank}] Received {len(my_batches)} batches", flush=True)

    start_time = time.time()
    local_results = []

    for i, batch in enumerate(my_batches):
        print(f"[Rank {rank}] Starting batch {i+1}/{len(my_batches)}: indices {batch}", flush=True)
        try:
            result = process_batch(batch, outputs_config['output_file'], tif_dates_ord, rank)
            if result:
                local_results.extend(result)
        except Exception as e:
            print(f"[Rank {rank}] Error in batch {i+1}: {repr(e)}", flush=True)

        elapsed = time.time() - start_time
        print(f"[Rank {rank}] Finished batch {i+1}/{len(my_batches)} "
              f"({(i+1)/len(my_batches)*100:.2f}%) - Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s", flush=True)

    if local_results:
        result_df = pd.concat(local_results, ignore_index=True)
        result_df = explode_columns(result_df)
        parquet_path = outputs_config['folders']['tabular'] / f"{ccd_config['filename']}_rank_{rank}.parquet"
        print(f"[Rank {rank}] Saving results to {parquet_path}", flush=True)
        result_df.to_parquet(parquet_path, index=False, engine='pyarrow')
#%%
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

    try:
        h5_file = h5py.File(sel_values_path, 'r')
    except Exception as e:
        print(f"[Rank {rank}] Failed to open HDF5 file: {repr(e)}", flush=True)
        return []

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

    results = []
    for args in arg_list:
        try:
            result = runDetectionForPoint(args)
            results.append(result)
        except Exception as e:
            print(f"[Rank {rank}] Error processing point {args[0]}: {repr(e)}", flush=True)

    return results
#%%
if __name__ == '__main__':
    main(preprocessing_config['batch_size'])
