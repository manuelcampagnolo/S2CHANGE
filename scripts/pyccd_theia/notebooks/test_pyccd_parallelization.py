import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import time
from sklearn.linear_model import Lasso

clf = Lasso(alpha=0.1)

N = 10000
dates = np.random.rand(100)
sel_values = np.random.rand(10,100,N)

def doSomething(args):
    i = args[0]
    sel_values_block = args[1]
    aux = sel_values_block[:,:,i]
    #return aux**2 #arbitrary operation to simulate some processing
    clf.fit(dates.reshape(-1,1),aux[0,:])


#without parallelization
def runSequential():
    result = []
    t = time.time()
    for i in range(N):
        result.append(doSomething((i, sel_values)))
    print("Without Parallelization: Execution took {} seconds".format(time.time()-t))


#with parallelization
def runParallel(batch_size):
    results = []
    t = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        for batch_index, start_index in enumerate(range(0, N, batch_size)):
            end_index = min(start_index + batch_size, N)
            
            sel_values_block = sel_values[:, :, start_index:end_index]
            
            arg_list = [(i, sel_values_block) for i in range(sel_values_block.shape[2])]
            
            for result in executor.map(doSomething, arg_list):
                results.append(result)
    print("With Parallelization (batch_size={}): Execution took {} seconds".format(batch_size,time.time()-t))

if __name__== '__main__':
    runSequential()
    runParallel(100)
    runParallel(10)
        
        

