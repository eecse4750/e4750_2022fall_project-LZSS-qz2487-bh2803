import numpy as np
import pycuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import scipy.signal as scisg
import pycuda.autoinit

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
class LZSS:
    def __init__(self):
        # Use this space to define the thread dimensions if required, or it can be incorporated into main function
        # You can also define a lambda function to compute grid dimensions if required.
        self.getSourceModule()

    def getSourceModule(self):
        kw_constant = r"""

        
        
        #define MAX_UNCODED     2
        #define MAX_CODED       128
        #define WINDOW_SIZE      128
        #define PCKTSIZE        4096


        """



        typedef = r"""
        typedef struct en_str_t
            {
                int offset;     /* offset addr to start of longest match */
                int length;     /* length of longest match */
            } en_str_t;

        typedef struct de_string_t
            {
                int offset;     /* offset to start of longest match */
                int length;     /* length of longest match */
            } de_string_t;
        
        """

        kw_matchfunc = r"""
        __device__ en_str_t StringMatch(
            unsigned char *StrWindow, //Sliding Window String
            unsigned char *StrAhead,  //Uncoded LookAhead String
            unsigned int WindowStart; //Start point of sliding window
            unsigned int UncodedStart;//Start point of uncoded lookahead
            unsigned int lastcheck,
            unsigned int tx){

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
                while (loop_i < WINDOW_SIZE){

                    /*Indicate if win[i] and lookahead are matching*/
                    if (StrWindow[i] == StrAhead[(UncodedStart + j)%(WINDOW_SIZE + MAX_CODED)]){
                        j++; //add to index of lookahead 
                        isMatch = true; //indicate string is matching
                    }
                    else{
                        // what if the previous one still match but j is bigger than length of matching part
                        if (isMatch && j > Match.length){
                            Match.length = j;
                            tmp = i - j; 
                            if (tmp < 0)
                                tmp += WINDOW_SIZE+MAX_CODED;
                            Match.offset = tmp;
                        }

                        j = 0;
                        isMatch = false;

                    }

                    i = (i + 1)%(WINDOW_SIZE + MAX_CODED);
                    loop_i ++;

                    if (loop >= maxcheck - 1){
                        loop = WINDOW_SIZE;//Stop the loop
                    }

                }

                if (j > Match.length && isMatch){
                    Match.length = j;
                    tmp = i - j;

                    if (tmp < 0)
                        tmp += WINDOW_SIZE + MAX_CODED;

                    Match.offset = tmp;
                }

                return Match
        }
        """

        kw_kernel_encode = r"""
            __global__ void EncodeLZSS (unsigned char *in, unsigned char *out, int SIZE){
                __shared__ unsigned char SlidingWindow[WINDOW_SIZE+MAX_CODED];
                __shared__ unsigned char LookAhead[MAX_CODED * 2];
                __shared__ unsigned char encodedData[MAX_CODED * 2];

                en_str_t mactchstring;

                #initialize parameters
                int WindowStart, UncodedStart;
                int readPosition;
                int position_W;
                int position_R;
                int LoadCounter;
                int bx = blockIdx.x;
                int tx = threadIdx.x;

                SlidingWindow[tx] = '';
                WindowStart = tx;
                UncodedStart = tx;
                position_R = 0;
                position_W = 0;
                lastCheck = 0;
                LoadCounter = 0;

                #Synchronize after writing to shared memory
                __syncthreads();

                #Copy data from input
                LookAhead[tx] = in[bx * PCKTSIZE + tx];
                position_R += MAX_CODED;

                SlidingWindow[(WindowStart + WINDOW_SIZE) % (WINDOW_SIZE + MAX_CODED)] = LookAhead[UncodedStart];

                __syncthreads();

                LookAhead[MAX_CODED + tx] = in[bx * PCKSIZE + tx];
                position_R += MAX_CODED;

                __syncthreads();

                LoadCounter++;


                //Look for match string
                matchstring = StringMatch(
                    SlidingWindow, //Sliding Window String
                    LookAhead,  //Uncoded LookAhead String
                    WindowStart; //Start point of sliding window
                    UncodedStart;//Start point of uncoded lookahead
                    0,
                    tx);

                __syncthreads();

                // now encoded the rest of the file until an EOF is read //
                while ((position_R) <= PCKTSIZE && !lastcheck)
                {		
                
                    
                    
                    if (matchstring.length >= MAX_CODED)
                        {
                                // garbage beyond last data happened to extend match length //
                                matchstring.length = MAX_CODED-1;
                        }

                    if (matchstring.length <= MAX_UNCODED)
                    {
                        // not long enough match.  write uncoded byte //
                        matchstring.length = 1;   // set to 1 for 1 byte uncoded //
                        encodedData[tx*2] = 1;
                        encodedData[tx*2 + 1] = uncodedLookahead[uncodedHead];
                    }
                    else if(matchstring.length > MAX_UNCODED)
                    {	
                        // match length > MAX_UNCODED.  Encode as offset and length. //
                        encodedData[tx*2] = (unsigned char)matchstring.length;
                        encodedData[tx*2+1] = (unsigned char)matchstring.offset;			
                    }

                        
                    //write out the encoded data into output
                    out_d[bx * PCKTSIZE*2 + position_W + tx*2] = encodedData[tx*2];
                    out_d[bx * PCKTSIZE*2 + position_W + tx*2 + 1] = encodedData[tx*2+1];
                    
                    //update written pointer and heads
                    position_W = position_W + MAX_CODED*2;
                    
                    windowHead = (windowHead + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
                    uncodedHead = (uncodedHead + MAX_CODED) % (MAX_CODED*2);
                    
                    __syncthreads(); 	

                            
                    //if(lastcheck==1)
                    //{
                    //	break;			
                    //}	
                    
                    //if(!lastcheck)
                    {
                        if(position_R<PCKTSIZE){
                            //uncodedLookahead[(uncodedHead+ MAX_CODED)% (MAX_CODED*2)] = tex1Dfetch(in_d_tex, bx * PCKTSIZE + position_R + tx);
                            uncodedLookahead[(uncodedHead+ MAX_CODED)% (MAX_CODED*2)] = in_d[bx * PCKTSIZE + position_R + tx];
                            position_R+=MAX_CODED;
                            
                            //find the location for the thread specific view of window
                            slidingWindow[ (windowHead + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedLookahead[uncodedHead];
                            //__syncthreads(); 	
                        }
                        else{
                            lastcheck++;				
                            slidingWindow[(windowHead + MAX_CODED ) % (WINDOW_SIZE+MAX_CODED)] = '^';		
                        }
                        __syncthreads(); 	
                        
                        loadcounter++;
                        matchstring = FindMatch(windowHead, uncodedHead,slidingWindow,uncodedLookahead,tx,bx, position_W, lastcheck,loadcounter);
                    }
                    
                } //while
                
                if(lastcheck==1)
                    {
                        if(matchstring.length > (MAX_CODED - tx))
                            matchstring.length = MAX_CODED - tx;
                    }
                    
                if (matchstring.length >= MAX_CODED)
                    {
                        // garbage beyond last data happened to extend match length //
                        matchstring.length = MAX_CODED-1;
                    }

                if (matchstring.length <= MAX_UNCODED)
                    {
                        // not long enough match.  write uncoded byte //
                        matchstring.length = 1;   // set to 1 for 1 byte uncoded //
                        encodedData[tx*2] = 1;
                        encodedData[tx*2 + 1] = uncodedLookahead[uncodedHead];
                    }
                else if(matchstring.length > MAX_UNCODED)
                    {	
                        // match length > MAX_UNCODED.  Encode as offset and length. //
                        encodedData[tx*2] = (unsigned char)matchstring.length;
                        encodedData[tx*2+1] = (unsigned char)matchstring.offset;			
                    }

                        
                //write out the encoded data into output
                out_d[bx * PCKTSIZE*2 + position_W + tx*2] = encodedData[tx*2];
                out_d[bx * PCKTSIZE*2 + position_W + tx*2 + 1] = encodedData[tx*2+1];
                
                //update written pointer and heads
                position_W = position_W + MAX_CODED*2;
                
                windowHead = (windowHead + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
                uncodedHead = (uncodedHead + MAX_CODED) % (MAX_CODED*2);
                    
            }

            }
        """
        kw_kernel_decode = r'''
    __global__ void DecodeKernel(unsigned char * in_d, unsigned char * out_d, int * error_d, int * sizearr_d, int SIZEBLOCK)
    {

        // cyclic buffer sliding window of already read characters //
        unsigned char slidingWindow[WINDOW_SIZE];
        unsigned char uncodedLookahead[MAX_CODED];
        //unsigned char writebuf[8];

        int nextChar;                       /* next char in sliding window */
        de_string_t code;              /* offset/length code for string */

        // 8 code flags and encoded strings //
        unsigned char flags, flagsUsed;
        //int nextEncoded;                // index into encodedData //
        //de_string_t matchData;
        int i, c;

        //initialize variables
        flags = 0;
        flagsUsed = 7;
        nextChar = 0;

        //long lSize=PCKTSIZE;
        int filepoint=0;
        int wfilepoint=0;

        int bx = blockIdx.x;
        int tx = threadIdx.x; 
        
        int sizeinpckt = 0, startadd = 0;
        startadd = sizearr_d[bx * SIZEBLOCK + tx]; //read the size of the packet
        sizeinpckt = sizearr_d[bx * SIZEBLOCK + tx + 1] -  startadd;

        //bigger than a packet hold-compression took more space for that packet
        //REPORT 
        if(sizeinpckt > PCKTSIZE){
            (*error_d)++;
        }

        // ************************************************************************
        //* Fill the sliding window buffer with some known vales.  EncodeLZSS must
        //* use the same values.  If common characters are used, there's an
        //* increased chance of matching to the earlier strings.
        //************************************************************************ /
        for (i = 0; i < WINDOW_SIZE; i++)
        {
            slidingWindow[i] = ' ';
        }

        while (TRUE)
            {
                flags >>= 1;
                flagsUsed++;

                if (flagsUsed == 8)
                {
                    // shifted out all the flag bits, read a new flag //
                    if (filepoint >= sizeinpckt)
                    {
                        break;
                    }
                    c=in_d[startadd + filepoint]; //packet*PCKTSIZE 
                    filepoint++;
                    flags = c & 0xFF;
                    flagsUsed = 0;
                }

                if (flags & 0x01)
                {
                    // uncoded character //
                    if (filepoint >= sizeinpckt)
                    {
                        break;
                    }
                    
                    // write out byte and put it in sliding window //
                    out_d[bx * SIZEBLOCK * PCKTSIZE + tx*PCKTSIZE+wfilepoint]=in_d[startadd +filepoint];
                    wfilepoint++;
                    slidingWindow[nextChar] = in_d[startadd +filepoint];
                    nextChar = (nextChar + 1) % WINDOW_SIZE;
                    filepoint++;
                }
                else 
                {
                    // offset and length //
                    if (filepoint >= sizeinpckt)
                    {
                        break;
                    }
                    code.length=in_d[startadd +filepoint];
                    filepoint++;

                    if (filepoint >= sizeinpckt)
                    {
                        break;
                    }
                    code.offset =in_d[startadd +filepoint];
                    filepoint++;
                    
                    // ****************************************************************
                    //* Write out decoded string to file and lookahead.  It would be
                    //* nice to write to the sliding window instead of the lookahead,
                    ////* but we could end up overwriting the matching string with the
                    ///* new string if abs(offset - next char) < match length.
                    //**************************************************************** /
                    for (i = 0; i < code.length; i++)
                    {
                        c = slidingWindow[(code.offset + i) % WINDOW_SIZE];
                        out_d[bx * SIZEBLOCK * PCKTSIZE + tx*PCKTSIZE + wfilepoint]=c;
                        wfilepoint++;
                        uncodedLookahead[i] = c;
                    }

                    // write out decoded string to sliding window //
                    for (i = 0; i < code.length; i++)
                    {
                        slidingWindow[(nextChar + i) % WINDOW_SIZE] =
                            uncodedLookahead[i];
                    }

                    nextChar = (nextChar + code.length) % WINDOW_SIZE;
                }
            }
            
    }
            '''


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

        #memory allocate
        X_gpu = cuda.mem_alloc(X.size * X.dtype.itemsize)
        Y_gpu = cuda.mem_alloc(Y.size * Y.dtype.itemsize)


        #memory transfer
        start.record()
        cuda.memcpy_htod(X_gpu,X)
        
        #Call Kernel Func
        prg = self.module_encode.get_function("EncodeLZSS")

        #Set block and grid size
        block = (512,1,1)
        grid = (int(np.ceil(length/512)),1,1)


        #Run func
        prg(X_gpu,Y_gpu,length,block=block,grid=grid)


        #Copy Back
        cuda.memcpy_dtoh(Y,Y_gpu)

        #cuda.memcpy_dtoh(S,S_gpu)
        #print(S[-100:])
        end.record()
        cuda.Context.synchronize()
        t = start.time_till(end)

        return Y,t


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

