import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import time
from sklearn.linear_model import Lasso
import pandas as pd


# clf = Lasso(alpha=0.1)

# N = 10000
#dates = np.random.rand(100)
# sel_values = np.random.rand(10,100,N)

def doSomething(args):
    clf = Lasso(alpha=0.1)
    i = args[0]
    sel_values_block = args[1]
    dates = args[2]
    aux = sel_values_block[:,:,i]
    #return aux**2 #arbitrary operation to simulate some processing
    clf.fit(dates.reshape(-1,1),aux[0,:])


#without parallelization
def runSequential(array, dates, print_bool=True):
    result = []
    t = time.time()
    for i in range(array.shape[-1]):
        result.append(doSomething((i, array, dates)))
    if print_bool:
        print("Without Parallelization: Execution took {} seconds".format(time.time()-t))
    return time.time()-t


#with parallelization
def runParallel(batch_size):
    results = []
    t = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        for batch_index, start_index in enumerate(range(0, N, batch_size)):
            end_index = min(start_index + batch_size, N)
            
            sel_values_block = sel_values[:, :, start_index:end_index]
            
            arg_list = [(i, sel_values_block, dates) for i in range(sel_values_block.shape[2])]
            
            for result in executor.map(doSomething, arg_list):
                results.append(result)
    print("With Parallelization (batch_size={}): Execution took {} seconds".format(batch_size,time.time()-t))
    return time.time()-t

def helper(args):
    runSequential(args[0], args[1], print_bool=False)

def runParallelBatches(batch_size, dates):
    #clf = Lasso(alpha=0.1)
    results = []
    t = time.time()

    batches = [sel_values[:, :, i:i+batch_size] for i in range(0, N, batch_size)]

    args = [(batch, dates) for batch in batches]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        for result in executor.map(helper, args):
            results.append(result)
        
    print("With Batch Parallelization (batch_size={}): Execution took {} seconds".format(batch_size,time.time()-t))
    return time.time()-t


if __name__== '__main__':

    df = pd.DataFrame(columns=['N','Type','Batch','Exec_time'])

    batch_sizes = [10, 100, 1000]#[100, 500, 1000]

    #N = 1000
    for N in [20000, 100000, 250000, 500000, 750000, 1000000]:#[10000, 100000]:

        dates = np.random.rand(100)
        sel_values = np.random.rand(10,100,N)

        s = runSequential(sel_values, dates)
        p1 = runParallel(batch_sizes[0])
        pb1 = runParallelBatches(batch_sizes[0], dates)
        p2 = runParallel(batch_sizes[1])
        pb2 = runParallelBatches(batch_sizes[1], dates)
        p3 = runParallel(batch_sizes[2])
        pb3 = runParallelBatches(batch_sizes[2], dates)

        df = pd.concat([df, pd.DataFrame({'N':[N]*7,
                                         'Type':['Seq']+['Par','ParBatch']*3,
                                         'Batch':[N]+[batch_sizes[0]]*2+[batch_sizes[1]]*2+[batch_sizes[2]]*2,
                                         'Exec_time':[s,p1,pb1,p2,pb2,p3,pb3]})])
        
    df.to_parquet('df_testes_parallelization2.parquet')
