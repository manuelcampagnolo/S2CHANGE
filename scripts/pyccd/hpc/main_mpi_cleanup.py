# -*- coding: utf-8 -*-
#region Imports and Path Setup
# Standard library imports
import os
import sys
import platform
from datetime import datetime
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
from shared.processing import runDetectionForPoint
from shared.preprocessing import check_or_initialize_file
from config.config import input_config, preprocessing_config, outputs_config, ccd_config

# Environment variables
cpus_slurm = int(os.getenv('SLURM_NTASKS', os.cpu_count()))

# Suppress warnings
warnings.filterwarnings('ignore')
#endregion

#%% ConfiguraÃ§Ãµes MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#%%

def main(batch_size=None):
    if rank == 0:
        # root rank
        tif_dates_ord, N = check_or_initialize_file(outputs_config['output_file'], 
                                                    input_config['tiles'], 
                                                    input_config['data_source'], 
                                                    input_config['s2_tile'], 
                                                    preprocessing_config['min_year'], 
                                                    preprocessing_config['max_date'], 
                                                    input_config['validation_path'], 
                                                    preprocessing_config['bandas_desejadas'], 
                                                    outputs_config['output_path'], 
                                                    preprocessing_config['img_collection'], 
                                                    preprocessing_config['nodata_value'])
                                                    # raster_path - This variable is not actually used in the function

        # Split data indexes
        indices = list(range(0, N, batch_size))
        batches = [
            (start, min(start + batch_size, N)) for start in indices
        ]
        print(f"[Rank {rank}] Total de batches criados: {len(batches)}")

        # Split batches between rans
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
        total_batches = len(my_batches)  # Total number of lots

        with tqdm(total=total_batches, desc=f"Processo {rank}") as pbar:
            def update_progress(_):
                pbar.n = progress.value  # Update progress bar with shared value
                pbar.refresh()

            # Process allocated batches using ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=cpus_slurm) as executor:
                futures = [
                    executor.submit(
                        process_single_batch,
                        batch,
                        outputs_config['output_file'],
                        tif_dates_ord,
                        progress
                    )
                    for batch in my_batches
                ]
                for future in futures:
                    future.add_done_callback(update_progress)

        local_results = [future.result() for future in futures]

    # Saving results locally per process
    dfs = [df for results in local_results for df in results]
    if dfs:
        # Create a Parquet for each process
        rank_parquet_filename = outputs_config['folders']['tabular'] / f'{ccd_config['filename']}_rank_{rank}.parquet'
        result_df = pd.concat(dfs, ignore_index=True)
        for col in result_df.columns:
            if result_df[col].apply(lambda x: isinstance(x, list)).any():
                result_df = result_df.explode(col)
        result_df.to_parquet(rank_parquet_filename, index=False)

    comm.Barrier()  # Sync all ranks before continuing

def process_batch(args):
    sel_values_block, xs_slice, ys_slice, tif_dates_ord = args
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

def process_single_batch(batch, sel_values_path, tif_dates_ord, progress):
    start, end = batch

    # Load only the specific block into the batch
    h5_file = h5py.File(sel_values_path, 'r')
    sel_values_block = h5_file['values'][:, :, start:end]
    xs_slice = h5_file['xs'][start:end]
    ys_slice = h5_file['ys'][start:end]

    # Process the specific bloc
    result = process_batch((sel_values_block, xs_slice, ys_slice, tif_dates_ord))

    # Update the shared progress bar
    progress.value += 1
    return result


if __name__ == '__main__':
    #if rank == 0:
        # print(f"Numero total de pixels processados: {N}")
        # print(f"Executando com batch_size = {BATCH_SIZE} e N = {N}")
        # print(f'Numero de CPUs para o ProcessPoolExecutor: {cpus_slurm}')
    main(preprocessing_config['batch_size'])
