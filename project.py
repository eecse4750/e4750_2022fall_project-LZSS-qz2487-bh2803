import numpy as np
from io import BytesIO
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
        #define MAX_CODED       512
        #define WINDOW_SIZE     512
        #define PCKTSIZE        4096
        #define TRUE            1
        #define FALSE           0


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
        __device__ en_str_t FindMatch(
            int windowHead, 
            int uncodedHead, 
            unsigned char* slidingWindow, 
            unsigned char* uncodedLookahead,
            int tx, 
            int bx, 
            int wfilepoint, 
            int lastcheck, 
            int loadcounter)
        {
            en_str_t matchData;
            int i, j;
            int maxcheck;
            int matchingState=0;
            int loop=0;
            
            matchData.length = 1; // make it 1 in the 0 case, it will be returned as 1, 0 gives problems
            matchData.offset = 1; // make it 1 in the 0 case, it will be returned as 1, 0 gives problems
            i = windowHead ;  // start at the beginning of the sliding window //
            j = 0; //counter for matchings

            
            //if(lastcheck) 
                maxcheck = MAX_CODED - tx*lastcheck;
            //else
            //	maxcheck = MAX_CODED;
            
            int tempi=0;
            while (loop<WINDOW_SIZE)
            {
                if (slidingWindow[i] == uncodedLookahead[(uncodedHead+j)% (WINDOW_SIZE+MAX_CODED)])
                {
                    j++;
                    matchingState=1;		
                }
                else
                {
                    if(matchingState && j > matchData.length)
                    {
                        matchData.length = j;
                        tempi=i-j;
                        if(tempi<0)
                            tempi+=WINDOW_SIZE+MAX_CODED;
                        matchData.offset = tempi;
                    }
                    
                    j=0;
                    matchingState=0;		
                }
            
                i = (i + 1) % (WINDOW_SIZE+MAX_CODED);
                loop++;	
                if (loop >= maxcheck-1)
                {
                    /// we wrapped around ///
                    loop = WINDOW_SIZE; //break;
                }
            }
            
            if(j > matchData.length && matchingState )
            {
                matchData.length = j;
                tempi=i-j;
                if(tempi<0)
                    tempi+=WINDOW_SIZE+MAX_CODED;

                matchData.offset = tempi;
            }
                
            
            return matchData;
        }
        """

        kw_kernel_encode = r"""
        __global__ void EncodeKernel(unsigned char * in_d, unsigned char * out_d, unsigned int SIZEBLOCK)
        {


            /* cyclic buffer sliding window of already read characters */
            __shared__ unsigned char slidingWindow[WINDOW_SIZE+(MAX_CODED)];
            __shared__ unsigned char uncodedLookahead[MAX_CODED*2];
            __shared__ unsigned char encodedData[MAX_CODED*2];
            en_str_t matchData;
        
            int windowHead, uncodedHead;    // head of sliding window and lookahead //
            int filepoint;			//file index pointer for reading
            int wfilepoint;			//file index pointer for writing
            int lastcheck;			//flag for last run of the packet
            int loadcounter=0;
            
            int bx = blockIdx.x;
            int tx = threadIdx.x; 
            
            
        //***********************************************************************
        // * Fill the sliding window buffer with some known values.  DecodeLZSS must
        // * use the same values.  If common characters are used, there's an
        // * increased chance of matching to the earlier strings.
        // *********************************************************************** //
            //printf("%d",WINDOW_SIZE+(MAX_CODED));
            slidingWindow[tx] = ' ';
            windowHead = tx;
            uncodedHead = tx;
	
            filepoint=0;
            wfilepoint=0;
            lastcheck=0;
            
            __syncthreads();

            
            //***********************************************************************
            //* Copy MAX_CODED bytes from the input file into the uncoded lookahead
            //* buffer.
            //*********************************************************************** //
        
            //uncodedLookahead[tx] = tex1Dfetch(in_d_tex, bx * PCKTSIZE + tx); //in_d[bx * PCKTSIZE + tx];
            uncodedLookahead[tx] = in_d[bx * PCKTSIZE + tx];
            filepoint+=MAX_CODED;
            
            slidingWindow[ (windowHead + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedLookahead[uncodedHead];
            //tex1Dfetch(in_d_tex, bx * PCKTSIZE + tx);//uncodedLookahead[uncodedHead];
            
            __syncthreads(); 
            
            //uncodedLookahead[MAX_CODED+tx] = tex1Dfetch(in_d_tex, bx * PCKTSIZE + filepoint + tx); //in_d[bx * PCKTSIZE + filepoint + tx];

            uncodedLookahead[MAX_CODED+tx] = in_d[bx * PCKTSIZE + filepoint + tx];
            
            filepoint+=MAX_CODED;
            

            
            __syncthreads();
            
            loadcounter++;
            // Look for matching string in sliding window //	
            matchData = FindMatch(windowHead, uncodedHead,slidingWindow,uncodedLookahead,  tx, bx, 0, 0,loadcounter);
            __syncthreads();  
            
            // now encoded the rest of the file until an EOF is read //
            while ((filepoint) <= PCKTSIZE && !lastcheck)
            {		
            
                
                
                if (matchData.length >= MAX_CODED)
                    {
                            // garbage beyond last data happened to extend match length //
                            matchData.length = MAX_CODED-1;
                    }

                if (matchData.length <= MAX_UNCODED)
                {
                    // not long enough match.  write uncoded byte //
                    matchData.length = 1;   // set to 1 for 1 byte uncoded //
                    encodedData[tx*2] = 1;
                    encodedData[tx*2 + 1] = uncodedLookahead[uncodedHead];
                }
                else if(matchData.length > MAX_UNCODED)
                {	
                    // match length > MAX_UNCODED.  Encode as offset and length. //
                    encodedData[tx*2] = (unsigned char)matchData.length;
                    encodedData[tx*2+1] = (unsigned char)matchData.offset;			
                }

                    
                //write out the encoded data into output
                out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2] = encodedData[tx*2];
                out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedData[tx*2+1];
                //printf("%c",encodedData[tx*2]);
                //printf("%c",encodedData[tx*2+1]);
                
                
                //update written pointer and heads
                wfilepoint = wfilepoint + MAX_CODED*2;
                
                windowHead = (windowHead + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
                uncodedHead = (uncodedHead + MAX_CODED) % (MAX_CODED*2);
                
                __syncthreads(); 	

                        
                //if(lastcheck==1)
                //{
                //	break;			
                //}	
                
                //if(!lastcheck)
                {
                    if(filepoint<PCKTSIZE){
                        //uncodedLookahead[(uncodedHead+ MAX_CODED)% (MAX_CODED*2)] = tex1Dfetch(in_d_tex, bx * PCKTSIZE + filepoint + tx);
                        uncodedLookahead[(uncodedHead+ MAX_CODED)% (MAX_CODED*2)] = in_d[bx * PCKTSIZE + filepoint + tx];
                        filepoint+=MAX_CODED;
                        
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
                    matchData = FindMatch(windowHead, uncodedHead,slidingWindow,uncodedLookahead,tx,bx, wfilepoint, lastcheck,loadcounter);
                }
                
            } //while
            
                if(lastcheck==1)
                {
                    if(matchData.length > (MAX_CODED - tx))
                        matchData.length = MAX_CODED - tx;
                }
                
                if (matchData.length >= MAX_CODED)
                    {
                        // garbage beyond last data happened to extend match length //
                        matchData.length = MAX_CODED-1;
                    }

                if (matchData.length <= MAX_UNCODED)
                {
                    // not long enough match.  write uncoded byte //
                    matchData.length = 1;   // set to 1 for 1 byte uncoded //
                    encodedData[tx*2] = 1;
                    encodedData[tx*2 + 1] = uncodedLookahead[uncodedHead];
                }
                else if(matchData.length > MAX_UNCODED)
                {	
                    // match length > MAX_UNCODED.  Encode as offset and length. //
                    encodedData[tx*2] = (unsigned char)matchData.length;
                    encodedData[tx*2+1] = (unsigned char)matchData.offset;			
                }

                    
                //write out the encoded data into output
                out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2] = encodedData[tx*2];
                out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedData[tx*2+1];
                //printf("%c",encodedData[tx*2]);
                //printf("%c",encodedData[tx*2+1]);

                //update written pointer and heads
                wfilepoint = wfilepoint + MAX_CODED*2;
                
                windowHead = (windowHead + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
                uncodedHead = (uncodedHead + MAX_CODED) % (MAX_CODED*2);
                
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
        block = (512,1,1)
        grid = (int(np.ceil(length/512)),1,1)


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
    with open('gistfile1.txt','r',encoding='utf-8') as f:
    #with open('wordlist.txt','r',encoding='utf-8') as f:
        content = f.read()
    file_list_r = [*content]
    file_arr_r = np.array(file_list_r).astype(bytes)
    #print(file_arr_r)
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
    result,t = PS.GPU_Compress(file_arr_r,np.intc(10000))
    #result = result.astype('<U1')
    #res = "".join(result)
    #print(result)
    for elem in result:
        elem = bytes(elem)
        w_f.write(elem)
    f.close()
    w_f.close()


