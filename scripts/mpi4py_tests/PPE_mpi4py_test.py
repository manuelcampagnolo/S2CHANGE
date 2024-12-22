from mpi4py import MPI
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import math
import os
from tqdm import tqdm
from datetime import datetime
import time

def process_point(args):
    """
    Simulates processing a single point and returns monitoring information
    """
    point_index, node_id, cpu_id = args
    time.sleep(0.1)  # Simulate processing time
    
    # Create a row of data similar to the original function's output
    return pd.DataFrame([{
        'point_index': point_index,
        'node_id': node_id,
        'cpu_id': cpu_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'value': np.random.rand()  # Simulate some computed value
    }])

def monitor_parallel_processing(batch_size=None):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create monitoring dictionary
    monitoring_data = {
        'node_id': rank,
        'total_nodes': size,
        'cpu_cores': int(os.environ.get('SLURM_CPUS_ON_NODE', 1)),
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'batch_processing_times': []
    }
    
    # Simulate some data for testing
    if rank == 0:
        # Create sample data
        N = 1000  # Total points to process
        total_points = N
        points_per_node = math.ceil(total_points / size)
        
        # Create dummy arrays for compatibility with original function
        sel_values = np.random.rand(10, 5, N)
        xs = np.arange(N)
        ys = np.arange(N)
        
        # Broadcast necessary data to all nodes
        comm.bcast((total_points, points_per_node, batch_size), root=0)
        
        # Distribute data chunks to other nodes
        for i in range(1, size):
            start_idx = i * points_per_node
            end_idx = min(start_idx + points_per_node, total_points)
            
            if start_idx < total_points:
                node_data = {
                    'sel_values': sel_values[:, :, start_idx:end_idx],
                    'xs': xs[start_idx:end_idx],
                    'ys': ys[start_idx:end_idx],
                    'start_index': start_idx
                }
                comm.send(node_data, dest=i)
                
        # Set data for master node
        start_idx = 0
        end_idx = points_per_node
        node_data = {
            'sel_values': sel_values[:, :, start_idx:end_idx],
            'xs': xs[start_idx:end_idx],
            'ys': ys[start_idx:end_idx],
            'start_index': start_idx
        }
    else:
        # Other nodes receive broadcast data
        total_points, points_per_node, batch_size = comm.bcast(None, root=0)
        # Receive node-specific data
        node_data = comm.recv(source=0)
    
    # Set batch size if not provided
    if batch_size is None:
        batch_size = 100  # Smaller default batch size for testing
    
    # Calculate local workload
    local_points = len(node_data['xs'])
    num_local_batches = math.ceil(local_points / batch_size)
    
    # Initialize list to store DataFrames
    dfs = []
    
    # Process batches using local CPU cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        progress_bar = tqdm(total=num_local_batches, disable=rank != 0)
        
        for batch_index in range(num_local_batches):
            batch_start_time = time.time()
            start_idx = batch_index * batch_size
            end_idx = min(start_idx + batch_size, local_points)
            
            # Create work items for the batch
            work_items = [(i + node_data['start_index'], rank, cpu_id) 
                         for i in range(start_idx, end_idx)
                         for cpu_id in range(os.cpu_count())]
            
            # Process batch
            batch_results = list(executor.map(process_point, work_items))
            
            # Combine batch results
            if batch_results:
                batch_df = pd.concat(batch_results, ignore_index=True)
                dfs.append(batch_df)
            
            # Record monitoring data
            batch_end_time = time.time()
            monitoring_data['batch_processing_times'].append({
                'batch_id': batch_index,
                'points_processed': end_idx - start_idx,
                'processing_time': batch_end_time - batch_start_time
            })
            
            if rank == 0:
                progress_bar.update(1)
        
        if rank == 0:
            progress_bar.close()
    
    # Combine all DataFrames from this node
    local_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    # Add monitoring information to the DataFrame
    local_df['node_total_batches'] = num_local_batches
    local_df['node_total_points'] = local_points
    
    # Gather results from all nodes
    all_dfs = comm.gather(local_df, root=0)
    all_monitoring_data = comm.gather(monitoring_data, root=0)
    
    # Master node combines and returns the results
    if rank == 0:
        # Combine DataFrames from all nodes
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Create monitoring report
        monitoring_report = {
            'total_nodes': size,
            'total_points': total_points,
            'points_per_node': points_per_node,
            'batch_size': batch_size,
            'node_reports': all_monitoring_data,
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return final_df, monitoring_report
    
    return None, None

def print_monitoring_report(df, report):
    if report is None:
        return
        
    print("\n=== Parallel Processing Monitoring Report ===")
    print(f"Total Nodes: {report['total_nodes']}")
    print(f"Total Points Processed: {report['total_points']}")
    print(f"Points per Node: {report['points_per_node']}")
    print(f"Batch Size: {report['batch_size']}")
    print(f"Start Time: {report['node_reports'][0]['start_time']}")
    print(f"End Time: {report['end_time']}")
    
    print("\nNode Processing Details:")
    for node_data in report['node_reports']:
        print(f"\nNode {node_data['node_id']}:")
        print(f"  CPU Cores Used: {node_data['cpu_cores']}")
        print(f"  Batches Processed: {len(node_data['batch_processing_times'])}")
        
        avg_time = np.mean([t['processing_time'] 
                           for t in node_data['batch_processing_times']])
        print(f"  Average Batch Processing Time: {avg_time:.2f} seconds")
    
    print("\nDataFrame Summary:")
    print(f"Total Records: {len(df)}")
    print("\nPoints Processed per Node:")
    print(df.groupby('node_id')['point_index'].count())
    print("\nPoints Processed per CPU core (across all nodes):")
    print(df.groupby('cpu_id')['point_index'].count())

if __name__ == "__main__":
    final_df, report = monitor_parallel_processing(batch_size=100)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print_monitoring_report(final_df, report)
