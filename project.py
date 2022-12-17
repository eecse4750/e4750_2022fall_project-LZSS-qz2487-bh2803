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
import sys,getopt

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

    #MACRO to change different version test
    '''
    Enter "python project.py -d" to activate different file mode
    Enter "python project.py -a" to activate naive mode
    Enter "python project.py -f FILENAME" to run a DIY file
    '''
    DIFF_FILE = False
    NAIVE_ACTIVE = False
    DIY_FILE = False

    argumentList = sys.argv[1:]
    options = "daf:"
    try:
        arguments,values = getopt.getopt(argumentList,options)
        for currentArgument,currentValue in arguments:
            if (currentArgument == "-d"):
                DIFF_FILE = True
            elif (currentArgument == "-a"):
                NAIVE_ACTIVE = True
            elif (currentArgument == "-f"):
                filenames = [currentValue]
                DIY_FILE = True
    except getopt.error as err:
        print(str(err))


    #Filenames
    if (DIY_FILE==False):
        if (DIFF_FILE == False):
            filenames = ['random1mb','random5mb','random10mb','random50mb']
            #filenames = ['random1mb']
        else:
            filenames = ['SHA1SUM','alice29.txt','asyoulik.txt','fields.c','grammar.lsp',\
                     'lcet10.txt','plrabn12.txt','xargs.1']
    #Operations
    operations = ['gpu','cpu','naive']
    #operations = ['cpu']

    #Size Array
    size_arr = np.array([])

    #Compression Ratio array
    cpu_ratio = np.array([])
    gpu_ratio = np.array([])
    naive_ratio = np.array([])

    #Initialize the arrays take times:
    cpu_time = np.array([])
    gpu_time = np.array([])
    naive_time = np.array([])

    for p,filename in enumerate(filenames):
        if (DIY_FILE == False):
            if (DIFF_FILE == False):
                with open("./testfile/%s.txt"%filename,'r',encoding='utf-8') as f:
                    content = f.read()
            else:
                with open("./canterbury/%s"%filename,'r',encoding='utf-8') as f:
                    content = f.read()
        else:
            with open(filename,'r',encoding='utf-8') as f:
                content = f.read()         

        file_list_r = [*content]
        file_arr_r = np.array(file_list_r).astype(bytes)
        input_len = len(file_arr_r)
        size_arr = np.append(size_arr,input_len)
        if(DEBUG):
            print("First 50 elements:")
            print(file_arr_r[:50])
            print("Input Shape")
            print(file_arr_r.shape)




        # Create an instance of the CudaModule class
        PS = LZSS()

        print("Testing Started For File: %s" %filename)
        print("----------------")
        print("File Size in Bytes:%d"%input_len)



        #FLAG to decide whether giving NAIVE version
        
        for operation in operations:
            if (operation == 'cpu'):
                #Open write file
                if (DIY_FILE == False):
                    w_f = open('./result/result_%s_%s.bin'%(filename,operation),'wb')
                else:
                    w_f = open('./result/result_DIY.bin','wb')
                res_cpu,t = PS.CPU_Compress_lzss(file_arr_r,len(file_arr_r))
                w_f.write(res_cpu)

                res_len = len(res_cpu)
                ratio = res_len/input_len
                cpu_ratio = np.append(cpu_ratio,ratio)
                cpu_time = np.append(cpu_time,t)
                w_f.close()
                print("CPU running time for file(%s): %f"%(filename,t))
                print("Compression Ratio: %.4f"%ratio)


            elif (operation == 'gpu'):
                #Open write file
                if (DIY_FILE == False):
                    w_f = open('./result/result_%s_%s.bin'%(filename,operation),'wb')
                else:
                    w_f = open('./result/result_DIY.bin','wb')
                res_gpu,t = PS.GPU_Compress(file_arr_r,len(file_arr_r))
                res_len = len(res_gpu)
                ratio = res_len/input_len
                gpu_ratio = np.append(gpu_ratio,ratio)
                gpu_time = np.append(gpu_time,t)
                #Write in File
                for elem in res_gpu:
                    elem = bytes(elem)
                    w_f.write(elem)
                w_f.close()
                print("GPU running time for file(%s): %f"%(filename,t))
                print("Compression Ratio: %.4f"%ratio)
            else:
                if (NAIVE_ACTIVE == True):
                    if (DIY_FILE == False):
                        w_f = open('./result/result_%s_%s.bin'%(filename,operation),'wb')
                    else:
                        w_f = open('./result/result_DIY.bin','wb')
                    res_naive,t = PS.CPU_Compress_naive(file_arr_r,len(file_arr_r))
                    res_len = len(res_naive)
                    ratio = res_len/input_len
                    naive_ratio = np.append(naive_ratio,ratio)
                    naive_time = np.append(naive_time,t)
                    #Write in File
                    for elem in res_naive:
                        elem = bytes(elem)
                        w_f.write(elem)
                    w_f.close()
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


        f.close()
        

    #Plot
    if (DIY_FILE == False):

        if (DIFF_FILE == False):
            #Running Time
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.plot(size_arr,cpu_time,'.-',label='CPU time')
            ax.plot(size_arr,gpu_time,'x-',label='GPU time')
            if(NAIVE_ACTIVE):
                ax.plot(size_arr,naive_time,'o-',label='Naive Time')
            ax.grid()
            ax.legend(loc=0)
            ax.set_xlabel("Size(bit)")
            ax.set_yscale('log')
            ax.set_ylabel("Time(ms)") 
            ax.set_title('Running Time for Rdm Num Files')
            plt.savefig('./image/RunningTime.jpg')
            #Ratio
            fig2 = plt.figure(2)
            ax2 = fig2.add_subplot(111)
            print(cpu_ratio)
            ax2.plot(size_arr.astype(str),cpu_ratio,'.-',label='CPU Ratio')
            ax2.plot(size_arr.astype(str),gpu_ratio,'x-',label='GPU Ratio')
            if(NAIVE_ACTIVE):
                ax2.plot(size_arr,naive_ratio,'o-',label='Naive Ratio')

            ax2.set_ylabel("Compression Ratio")
            ax2.legend(loc=0)
            ax2.grid()
            ax2.set_xlabel("Size(bit)")
            ax2.set_ylim(0,1)
            ax2.set_title('Compression Ratio for Rdm Num Files')
            plt.savefig('./image/CompressionRatio.jpg')
        else:
            #Running Time
            fig = plt.figure(1,figsize=(8,8))
            ax = fig.add_subplot(111)
            ax.plot(filenames,cpu_time,'.-',label='CPU time')
            ax.plot(filenames,gpu_time,'x-',label='GPU time')
            if(NAIVE_ACTIVE):
                ax.plot(filenames,naive_time,'o-',label='Naive Time')
            ax.grid()
            ax.legend(loc=0)
            ax.set_xlabel("Size(bit)")
            ax.set_yscale('log')
            ax.set_ylabel("Time(ms)") 
            ax.set_title('Running Time for Canterbury files')
            plt.savefig('./image/RunningTime_canterbury.jpg')

            #Ratio
            x = np.arange(8)
            fig2 = plt.figure(2,figsize=(8,8))
            ax2 = fig2.add_subplot(111)
            WIDTH = 0.4
            if(NAIVE_ACTIVE):
                WIDTH = 0.2
                ax2.bar(x,naive_ratio,width=WIDTH,label='Naive Ratio')
            ax2.bar(x-0.2,cpu_ratio,width=WIDTH,label='CPU Ratio')
            ax2.bar(x+0.2,gpu_ratio,width=WIDTH,label='GPU Ratio')


            ax2.set_ylabel("Compression Ratio")
            ax2.set_xticks(x)
            ax2.set_xticklabels(filenames)
            ax2.legend(loc=0)
            ax2.grid()
            ax2.set_xlabel("Filename")
            #ax2.set_ylim(0,1)
            ax2.set_title('Compression Ratio for canterbury files')
            if(NAIVE_ACTIVE):
                plt.savefig('./image/CompressionRatio_canterbury_with_naive.jpg')
            else:
                plt.savefig('./image/CompressionRatio_canterbury.jpg')




