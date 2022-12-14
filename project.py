import numpy as np
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import scipy.signal as scisg
import pycuda.autoinit

import kernels

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

class LZSS:
    def __init__(self):
        # Use this space to define the thread dimensions if required, or it can be incorporated into main function
        # You can also define a lambda function to compute grid dimensions if required.
        self.getSourceModule()

    def getSourceModule(self):

        kw_constant = kernels.kw_constant
        typedef = kernels.typedef
        kw_matchfunc = kernels.kw_matchfunc
        kw_kernel_encode = kernels.kw_kernel_encode
        kw_kernel_decode = kernels.kw_kernel_decode

        self.module_encode = SourceModule(kw_constant+typedef+kw_matchfunc+kw_kernel_encode)
        self.module_decode = SourceModule(kw_constant+typedef+kw_kernel_decode)

        # If you wish, you can also include additional compiled kernels and compile-time defines that you may use for debugging without modifying the above three compiled kernel.

    def CPU_Compress(self,input):
        # implement this, note you can change the function signature (arguments and return type)
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        # TODO: CPU function implementation

        end.record()
        cuda.Context.synchronize()
        t = start.time_till(end)
        return res,t

    def GPU_Compress(self,input_string,length):
        # implement this, note you can change the function signature (arguments and return type)
        SECTION_SIZE = 1024
        #Event objects to indicate starts and ends
        start = cuda.Event()
        end = cuda.Event()

        #initial list
        X = input_string
        Y = np.zeros_like(X)

        #memory allocate
        X_gpu = cuda.mem_alloc(X.size * X.dtype.itemsize)
        Y_gpu = cuda.mem_alloc(Y.size * Y.dtype.itemsize)


        #memory transfer
        start.record()
        cuda.memcpy_htod(X_gpu,X)
        
        #Call Kernel Func
        prg = self.module_encode.get_function("EncodeKernel")

        #Set block and grid size
        block = (128,1,1)
        grid = (int(np.ceil(length/128)),1,1)


        #Run func
        prg(X_gpu,Y_gpu,length,block=block,grid=grid)


        #Copy Back
        cuda.memcpy_dtoh(Y,Y_gpu)

        end.record()
        #cuda.Context.synchronize()
        #t = start.time_till(end)
        t=0;
        return Y,t


if __name__ == "__main__":
    #Main Code

    #Open Test file
    with open('wordlist.txt','r') as f:
        content = f.read()
    file_arr_r = np.array(content)
    #Open write file
    w_f = open('result.txt','w')


    # Create an instance of the CudaModule class
    PS = LZSS()

    #Initialize the arrays take times:
    avg_total_cpu_time = np.array([])
    avg_total_inef_time = np.array([])
    avg_total_ef_time = np.array([])

    #Nested loops:
    #InputDimensions->Iterations->Methods
    print("Testing Started!")
    print("----------------")

    #Initialize the running time array
    total_cpu_time = np.array([])
    total_gpu_time = np.array([])

    #run
    result,t = PS.GPU_Compress(file_arr_r,np.intc(10000))

    print(result.shape)
    w_f.write(np.array2string(result))
    f.close()
    w_f.close()


