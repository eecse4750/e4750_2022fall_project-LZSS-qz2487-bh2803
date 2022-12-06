import numpy as np
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import scipy.signal as scisg
import pycuda.autoinit

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
class PrefixSum:
    def __init__(self):
        # Use this space to define the thread dimensions if required, or it can be incorporated into main function
        # You can also define a lambda function to compute grid dimensions if required.
        self.getSourceModule()

    def getSourceModule(self):
        typedef = r"""
        typedef struct en_str_t
            {
                int offset;     /* offset addr to start of longest match */
                int length;     /* length of longest match */
            } en_str_t;
        """

        kw_matchfunc = r"""
        __device__ en_str_t StringMatch(
            unsigned char *StrWindow, //Sliding Window String
            unsigned char *StrAhead,  //Uncoded LookAhead String
            unsigned short lenWindow, //Length of sliding window
            unsigned short lenAhead,  //length of uncoded lookahead String
            unsigned int WindowStart; //Start point of sliding window
            unsigned int UncodedStart;//Start point of uncoded lookahead
            unsigned int *MatchInd,
            unsigned int lastcheck,
            unsigned int tx,){
                /*Match Info Initialization*/
                en_str_t Match;
                Match.offset = 1;
                Match.length = 1;

                bool isMatch = false;

                int maxcheck;
                maxcheck = MAX_CODED - tx * lastcheck;

                int loop_i = 0;
                int tmp = 0;
                int i = WindowStart; // index of sliding window string
                int j = 0; // index of Lookahead string

                /*Main Loop to get offset and length of matching string*/
                while (loop_i < WINDOWSIZE){

                    /*Indicate if win[i] and lookahead are matching*/
                    if (StrWindow[i] == StrAhead[(UncodedStart + j)%(WINDOWSIZE + MAX_CODED)]){
                        j++; //add to index of lookahead 
                        isMatch = true; //indicate string is matching
                    }
                    else{
                        // what if the previous one still match but j is bigger than length of matching part
                        if (isMatch && j > Match.length){
                            Match.length = j;
                            tmp = i - j; 
                            if (tmp < 0)
                                tmp += WINDOWSIZE+MAX_CODED;
                            Match.offset = tmp;
                        }

                        j = 0;
                        isMatch = false;

                    }

                    i = (i + 1)%(WINDOWSIZE + MAX_CODED);
                    loop_i ++;

                    if (loop >= maxcheck - 1){
                        loop = WINDOWSIZE;//Stop the loop
                    }

                }

                if (j > Match.length && isMatch){
                    Match.length = j;
                    tmp = i - j;

                    if (tmp < 0)
                        tmp += WINDOWSIZE + MAX_CODED;

                    Match.offset = tmp;
                }

                return Match
        }
        """



        self.module_eff = SourceModule()

        # If you wish, you can also include additional compiled kernels and compile-time defines that you may use for debugging without modifying the above three compiled kernel.

    def CPU_Compress(self,input_list,length):
        # implement this, note you can change the function signature (arguments and return type)
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        res = np.cumsum(input_list,dtype=float)
        end.record()
        cuda.Context.synchronize()
        t = start.time_till(end)
        return res,t

    def GPU_Compress(self,input_list,length):
        # implement this, note you can change the function signature (arguments and return type)
        SECTION_SIZE = 1024
        #Event objects to indicate starts and ends
        start = cuda.Event()
        end = cuda.Event()
        lengthS = np.intc(np.ceil(length/SECTION_SIZE))
        #initial list
        X = input_list
        Y = np.zeros_like(X)
        S = np.zeros((lengthS,),dtype=np.float32)
        S_prime = np.zeros((int(np.ceil(length/SECTION_SIZE/SECTION_SIZE)),),dtype=np.float32)

        #memory allocate
        X_gpu = cuda.mem_alloc(X.size * X.dtype.itemsize)
        Y_gpu = cuda.mem_alloc(Y.size * Y.dtype.itemsize)
        S_gpu = cuda.mem_alloc(S.size * S.dtype.itemsize)
        S_prime_gpu = cuda.mem_alloc(S_prime.size * S_prime.dtype.itemsize)

        #memory transfer
        start.record()
        cuda.memcpy_htod(X_gpu,X)
        
        #Call Kernel Func
        prg_inef = self.module_ineff.get_function("work_scan")
        prg_S = self.module_ineff.get_function("scanS")
        prg_out = self.module_ineff.get_function("write_output")

        #Set block and grid size
        block = (1024,1,1)
        grid = (int(np.ceil(length/1024)),1,1)

        #print(block)
        if (lengthS <= 1024):
            #Phase 1
            #Run the kernel function
            prg_inef(X_gpu,Y_gpu,S_gpu,length,block=block,grid=grid)

            #Phase 2
            prg_S(S_gpu,S_gpu,lengthS,block=block,grid=grid)

            #Phase 3
            prg_out(S_gpu,Y_gpu,length,block=block,grid=grid)
        else:
            lengthSS = np.intc(np.ceil(length/SECTION_SIZE))
            #Phase 1
            #Run the kernel function
            prg_inef(X_gpu,Y_gpu,S_gpu,length,block=block,grid=grid)

            #Phase 2
            prg_inef(S_gpu,S_gpu,S_prime_gpu,lengthS,block=block,grid=grid)
            prg_S(S_prime_gpu,S_prime_gpu,lengthSS,block=block,grid=grid)
            prg_out(S_prime_gpu,S_gpu,lengthS,block=block,grid=grid)

            #Phase 3
            prg_out(S_gpu,Y_gpu,length,block=block,grid=grid)

        #Copy Back
        cuda.memcpy_dtoh(Y,Y_gpu)

        #cuda.memcpy_dtoh(S,S_gpu)
        #print(S[-100:])
        end.record()
        cuda.Context.synchronize()
        t = start.time_till(end)

        return Y,t

    def prefix_sum_gpu_work_efficient(self,input_list,length):
        # implement this, note you can change the function signature (arguments and return type)
        SECTION_SIZE = 2048
        lengthS = np.intc(np.ceil(length/SECTION_SIZE))
        #Event objects to indicate starts and ends
        start = cuda.Event()
        end = cuda.Event()

        #initial list
        X = input_list
        Y = np.zeros_like(X)
        S = np.zeros((int(np.ceil(length/SECTION_SIZE)),),dtype=np.float32)
        S_prime = np.zeros((int(np.ceil(length/SECTION_SIZE/SECTION_SIZE)),),dtype=np.float32)

        #memory allocate
        X_gpu = cuda.mem_alloc(X.size * X.dtype.itemsize)
        Y_gpu = cuda.mem_alloc(Y.size * Y.dtype.itemsize)
        S_gpu = cuda.mem_alloc(S.size * S.dtype.itemsize)
        S_prime_gpu = cuda.mem_alloc(S_prime.size * S_prime.dtype.itemsize)

        #memory transfer
        start.record()
        cuda.memcpy_htod(X_gpu,X)

        #Call Kernel Func
        prg_ef = self.module_eff.get_function("work_scan")
        prg_S = self.module_eff.get_function("scanS")
        prg_out = self.module_eff.get_function("write_output")

        #Set block and grid size
        block = (1024,1,1)
        grid = (int(np.ceil(length/1024)),1,1)
        
        if (lengthS <= 1024):
            #Phase 1
            #Run the kernel function
            prg_ef(X_gpu,Y_gpu,S_gpu,length,block=block,grid=grid)

            #Phase 2
            prg_S(S_gpu,S_gpu,lengthS,block=block,grid=grid)

            #Phase 3
            prg_out(S_gpu,Y_gpu,length,block=block,grid=grid)
        else:
            lengthSS = np.intc(np.ceil(length/SECTION_SIZE))
            #Phase 1
            #Run the kernel function
            prg_ef(X_gpu,Y_gpu,S_gpu,length,block=block,grid=grid)

            #Phase 2
            prg_ef(S_gpu,S_gpu,S_prime_gpu,lengthS,block=block,grid=grid)
            prg_S(S_prime_gpu,S_prime_gpu,lengthSS,block=block,grid=grid)
            prg_out(S_prime_gpu,S_gpu,lengthS,block=block,grid=grid)

            #Phase 3
            prg_out(S_gpu,Y_gpu,length,block=block,grid=grid)

        #transfer output back
        cuda.memcpy_dtoh(Y,Y_gpu)
        #cuda.memcpy_dtoh(S,S_gpu)
        #print(S)
        end.record()
        cuda.Context.synchronize()
        t = start.time_till(end)

        return Y,t

    def test_prefix_sum_python(self):
        # implement this, note you can change the function signature (arguments and return type)
        test_list_1 = np.array([1,2,3,4,5])
        test_list_2 = np.array([3,1,7,0,4])
        test_result_1 = np.array([1,3,6,10,15])
        test_result_2 = np.array([3,4,11,11,15])
        result_1 = self.prefix_sum_python(test_list_1)
        result_2 = self.prefix_sum_python(test_list_2)
        if (np.allclose(result_1,test_result_1) and np.allclose(result_2,test_result_2)):
            print("CPU RESULT TEST PASSED")
        else:
            raise Exception("TEST FAILED")

    def test_prefix_sum_gpu_work_inefficient(self,input_list,length):
        # implement this, note you can change the function signature (arguments and return type)
        
        pass

    def test_prefix_sum_gpu_work_efficient(self,input_list,length):
        # implement this, note you can change the function signature (arguments and return type)
        pass

if __name__ == "__main__":
    #Main Code
    #All methods
    all_methods = ['cpu','inefficient','efficient']
    #all_methods = ['cpu','efficient']
    # Create an instance of the CudaModule class
    PS = PrefixSum()

    #Iteration Index
    ITERA = 10
    iteration_idx = np.arange(1,ITERA+1)

    #TEST CPU CASES
    #PS.test_prefix_sum_python()

    #Case length
    #case_lengths = np.array([134215680],dtype=np.intc)
    case_lengths = np.array([128, 2048, 16384, 262144, 134215680],dtype=np.intc)
    #Initialize the arrays take times:
    avg_total_cpu_time = np.array([])
    avg_total_inef_time = np.array([])
    avg_total_ef_time = np.array([])

    #Nested loops:
    #InputDimensions->Iterations->Methods
    print("Testing Started!")
    print("----------------")
    for current_len in case_lengths:
        #Initialize the input matrix with different dimensions:
        #input_list = np.float32(np.random.randint(low=1, high=1, size=(int(current_len),1)))
        input_list = np.ones((current_len,),dtype=np.float32)

        #Initialize the running time array
        total_cpu_time = np.array([])
        total_inef_time = np.array([])
        total_ef_time = np.array([])

        for iteration in iteration_idx:
            #print(iteration)
            for current_method in all_methods:
                if (current_method == 'cpu'):
                    print()
                    print("-------------------")
                    print("CPU case Started!")
                    res_cpu,t_cpu = PS.prefix_sum_python(input_list,current_len)
                    total_cpu_time = np.append(total_cpu_time,t_cpu)
                    print("CPU case Done!")
                    print("-------------------")
                    print()
                    #print(res_cpu[-100:])
                else:
                    if (current_method == 'inefficient'):
                        print()
                        print("-------------------")
                        print("Inefficient case Started!")
                        res_inef,t_inef = PS.prefix_sum_gpu_work_inefficient(input_list,current_len)
                        total_inef_time = np.append(total_inef_time,t_inef)
                        #print(res_inef[:100000])
                        #Compare with CPU results
                        sum_diff = res_cpu - res_inef
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print ("Inefficient case sum mismatch")
                            print (total_diff)
                            print ("close enough?")
                            print (np.allclose(res_inef,res_cpu))
                        else:
                            print("Inefficient Case Verified!")
                        print("------------------------")


                    elif (current_method == 'efficient'):
                        print()
                        print("-------------------")
                        print("Efficient case Started!")
                        res_ef,t_ef = PS.prefix_sum_gpu_work_efficient(input_list,current_len)
                        total_ef_time = np.append(total_ef_time,t_ef)
                        #print(res_ef)
                        #Compare with CPU results
                        sum_diff = res_cpu - res_ef
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print ("Efficient sum mismatch")
                            print (total_diff)
                            print ("close enough?")
                            print (np.allclose(res_ef,res_cpu))
                        else:
                            print("Efficient Case Verified!")
                        print("------------------------")



        avg_total_cpu_time = np.append(avg_total_cpu_time,total_cpu_time.sum()/ITERA)
        avg_total_inef_time = np.append(avg_total_inef_time,total_inef_time.sum()/ITERA)
        avg_total_ef_time = np.append(avg_total_ef_time,total_ef_time.sum()/ITERA)

        print("Length %d is done" %current_len)

    print("The running time for CPU with different input matrix size is:")
    print(avg_total_cpu_time)
    print("The running time for inefficient case with different input matrix size is:")
    print(avg_total_inef_time)
    print("The running time for efficient with different input matrix size is:")
    print(avg_total_ef_time)

    fig,ax1 = plt.subplots()
    ax1.plot(case_lengths,avg_total_cpu_time,'.-',label='CPU')
    ax1.plot(case_lengths,avg_total_inef_time,'.--',label='Inefficient')
    ax1.plot(case_lengths,avg_total_ef_time,'x-',label='Efficient')

    ax1.set_title('Plot of running time for CUDA')
    ax1.set_xlabel('Input Array Sizes')
    ax1.set_ylabel('Avg Running Time (ms)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    plt.grid()
    fig.savefig('hw3_PyCUDA.jpg')

