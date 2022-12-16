# Global Modules
import numpy as np
from io import BytesIO
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import scipy.signal as scisg
import pycuda.autoinit

# Local Modules
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

        self.module_encode = SourceModule(kw_constant+typedef+kw_matchfunc+kw_kernel_encode+kw_kernel_decode)

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
        #Event objects to indicate starts and ends
        start = cuda.Event()
        end = cuda.Event()

        #initial list
        X = input_string
        Y = np.append(np.zeros_like(X),np.zeros_like(X))
        #Z = np.zeros_like(Y)

        #memory allocate
        X_gpu = cuda.mem_alloc(X.size * X.dtype.itemsize)
        Y_gpu = cuda.mem_alloc(Y.size * Y.dtype.itemsize)
        #Z_gpu = cuda.mem_alloc(Z.size * Z.dtype.itemsize)

        #Call Kernel Func
        prg1 = self.module_encode.get_function("EncodeKernel")
        #prg2 = self.module_encode.get_function("DecodeKernel")

        #memory transfer
        start.record()
        cuda.memcpy_htod(X_gpu,X)

        #Set block and grid size
        block = (128,1,1)
        grid = (int(np.ceil(length/128/32)),1,1)


        #Run func
        prg1(X_gpu,Y_gpu,block=block,grid=grid)
        #prg2(Y_gpu,Z_gpu,block=block,grid=grid)

        #Copy Back
        cuda.memcpy_dtoh(Y,Y_gpu)
        out = self.CPU_GPU(Y)
        

        end.record()
        #cuda.Context.synchronize()
        #t = start.time_till(end)
        t=0;
        return out,t
    
    def CPU_GPU(self,input):
        length = int(len(input)/2)-1
        out = []
        step = 0
        print(len(input))
        
        #print(input[:223])

        for i in range(length):
            if (input[i*2] != b''):
                if (input[i*2] == b'\x01'):
                    out.append(input[2*i+1])
                    #print(input[2*i+1])
                else:
                    #print(1)
                    #temp =  str(int.from_bytes(input[2*i+1], "big"))  + str(int.from_bytes(input[2*i], "big"))
                    #print(type(temp))
                    out.append(input[2*i+1])
                    out.append(input[2*i])

                    #step += 4
        #print(input)
        return out


if __name__ == "__main__":
    #Main Code

    #Open Test file
    with open('wordlist.txt','r',encoding='utf-8') as f:
    #with open('wordlist.txt','r',encoding='utf-8') as f:
        content = f.read()
    file_list_r = [*content]
    file_arr_r = np.array(file_list_r).astype(bytes)
    print(file_arr_r)
    print(file_arr_r.shape)
    #Open write file
    w_f = open('result.txt','wb')


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
    result,t = PS.GPU_Compress(file_arr_r,len(file_arr_r))
    #result = result.astype('<U1')
    #res = "".join(result)
    print(result)
    print(len(result))
    for elem in result:
        elem = bytes(elem)
        w_f.write(elem)
    f.close()
    w_f.close()


