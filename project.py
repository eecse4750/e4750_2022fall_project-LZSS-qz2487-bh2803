# Global Modules
import numpy as np
from io import BytesIO
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import scipy.signal as scisg
import pycuda.autoinit
import lzss
import LZSSCPU

# Local Modules
import kernels_old as kernels

DEBUG = False
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

    def CPU_Compress_lzss(self,input,length):
        # implement this, note you can change the function signature (arguments and return type)
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        # TODO: CPU function implementation
        res = lzss.compress(input)
        end.record()
        cuda.Context.synchronize()
        t = start.time_till(end)
        return res,t

    def CPU_Compress_naive(self,input,length):
        # implement this, note you can change the function signature (arguments and return type)
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        # TODO: CPU function implementation
        #print(len(input))
        encoder = LZSSCPU.LZSSCPU()
        res = encoder.compress(input)
        end.record()
        cuda.Context.synchronize()
        t = start.time_till(end)
        return res,t

    def GPU_Compress(self,input,length):
        #Event objects to indicate starts and ends
        start = cuda.Event()
        end = cuda.Event()

        #initial list
        X = input
        Y = np.append(np.zeros_like(X),np.zeros_like(X))
        #Z = np.zeros_like(Y)

        #memory allocate
        X_gpu = cuda.mem_alloc(X.size * X.dtype.itemsize)
        Y_gpu = cuda.mem_alloc(Y.size * Y.dtype.itemsize)
        #Z_gpu = cuda.mem_alloc(Z.size * Z.dtype.itemsize)

        #Call Kernel Func
        #prg1 = self.module_encode.get_function("kw_encode")
        prg1 = self.module_encode.get_function("EncodeKernel")

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
        out = self.Compress(Y)
        

        end.record()
        cuda.Context.synchronize()
        t = start.time_till(end)
        return out,t
    
    def Compress(self,input):
        length = int(len(input)/2)
        out = []
        step = 0
        #print(input)
        if (DEBUG):
            print("Uncompressed Result:")
            print(input[:100])
            print()
        i = 0
        while i < length:
            #print(i)
            if (input[i*2] != b''):
                if (input[i*2] == b'\x01'):
                    out.append(input[2*i+1])
                    #print(input[2*i+1])
                else:
                    offset = input[2*i+1]
                    Maxlength = input[2*i]
                    i += ord(Maxlength)
                    out.append(bytes(Maxlength))
                    out.append(offset)
            i += 1
        
        #step += 4
        #print(input)
        #return out
        return out


if __name__ == "__main__":
    #Filenames
    filenames = ['sequency.txt','gistfile1.txt','big.txt']
    #Operations
    operations = ['gpu','cpu','naive']


    for p,filename in enumerate(filenames):
        with open(filename,'r',encoding='utf-8') as f:
            content = f.read()
        file_list_r = [*content]
        file_arr_r = np.array(file_list_r).astype(bytes)
        input_len = len(file_arr_r)

        if(DEBUG):
            print("First 50 elements:")
            print(file_arr_r[:50])
            print("Input Shape")
            print(file_arr_r.shape)

        #Open write file
        w_f = open('result_{:s}.bin'.format(filename),'wb')


        # Create an instance of the CudaModule class
        PS = LZSS()

        print("Testing Started For File: %s" %filename)
        print("----------------")
        print("File Size in Bytes:%d"%input_len)

        #Compression Ratio array
        cpu_ratio = np.array([])
        gpu_ratio = np.array([])
        naive_ratio = np.array([])

        #Initialize the arrays take times:
        cpu_time = np.array([])
        gpu_time = np.array([])
        naive_time = np.array([])

        #run
        NAIVE_ACTIVE = False
        for operation in operations:
            if (operation == 'cpu'):
                result,t = PS.CPU_Compress_lzss(file_arr_r,len(file_arr_r))
                res_len = len(result)
                ratio = res_len/input_len
                cpu_ratio = np.append(cpu_ratio,ratio)
                cpu_time = np.append(cpu_time,t)
                print("CPU running time for file(%s): %f"%(filename,t))
                print("Compression Ratio: %.4f"%ratio)


            else:
                if (operation == 'gpu'):
                    result,t = PS.GPU_Compress(file_arr_r,len(file_arr_r))
                    res_len = len(result)
                    ratio = res_len/input_len
                    gpu_ratio = np.append(gpu_ratio,ratio)
                    gpu_time = np.append(gpu_time,t)
                    print("GPU running time for file(%s): %f"%(filename,t))
                    print("Compression Ratio: %.4f"%ratio)
                else:
                    if (NAIVE_ACTIVE == True):
                        result,t = PS.CPU_Compress_naive(file_arr_r,len(file_arr_r))
                        res_len = len(result)
                        ratio = res_len/input_len
                        print("CPU Naive running time for file(%s): %f"%(filename,t))
                        print("Compression Ratio: %.4f"%ratio)
        
        print("----------------")
        print("Testing Finished For file %s" %filename)
        
        print()

        if(DEBUG):
            print("Compressed Result")
            print(result[:100])
            print("Shape of Result:")
            print(len(result))

        for elem in result:
            elem = bytes(elem)
            w_f.write(elem)
        f.close()
        w_f.close()


