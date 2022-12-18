
kw_constant = r'''
    #define MAX_UNCODED     2
    #define MAX_CODED       128
    #define WINDOW_SIZE     128
    #define PCKTSIZE        4096
    #define TRUE            1
    #define FALSE           0
'''


typedef = r'''
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
'''


kw_matchfunc = r'''
    __device__ en_str_t FindMatch(
        int StartPoint_win, 
        int StartPoint_uncoded, 
        unsigned char* searchWindow, 
        unsigned char* uncodedWindow,
        int tx, 
        int isLAST)
    {
        en_str_t matchInfo;
        int i, j;
        int isMAX;
        int matchingState=0;
        int loop=0;
        
        matchInfo.length = 1; // make it 1 in the 0 case, it will be returned as 1, 0 gives problems
        matchInfo.offset = 1; // make it 1 in the 0 case, it will be returned as 1, 0 gives problems
        i = StartPoint_win ;  // start at the beginning of the sliding window //
        j = 0; //counter for matchings

        
        isMAX = MAX_CODED - tx*isLAST;
        
        int tempi=0;
        while (loop<WINDOW_SIZE)
        {
            if (searchWindow[i] == uncodedWindow[(StartPoint_uncoded+j)% (WINDOW_SIZE+MAX_CODED)])
            {
                j++;
                matchingState=1;		
            }
            else
            {
                if(matchingState && j > matchInfo.length)
                {
                    matchInfo.length = j;
                    tempi=i-j;
                    if(tempi<0)
                        tempi+=WINDOW_SIZE+MAX_CODED;
                    matchInfo.offset = tempi;
                }
                
                j=0;
                matchingState=0;		
            }
        
            i = (i + 1) % (WINDOW_SIZE+MAX_CODED);
            loop++;	
            if (loop >= isMAX-1)
            {
                loop = WINDOW_SIZE; //stop while loop
            }
        }
        
        if(j > matchInfo.length && matchingState )
        {
            matchInfo.length = j;
            tempi=i-j;
            if(tempi<0)
                tempi+=WINDOW_SIZE+MAX_CODED;

            matchInfo.offset = tempi;
        }
            
        
        return matchInfo;
    }
'''


kw_kernel_encode = r'''
    __global__ void EncodeKernel(unsigned char * in_d, unsigned char * out_d)
        {

        __shared__ unsigned char searchWindow[WINDOW_SIZE+(MAX_CODED)];
        __shared__ unsigned char uncodedWindow[MAX_CODED*2];
        __shared__ unsigned char encodedBuf[MAX_CODED*2];
        en_str_t matchInfo;
    
        int StartPoint_win, StartPoint_uncoded;    // head of sliding window and lookahead //
        int filepoint;			//file index pointer for reading
        int wfilepoint;			//file index pointer for writing
        int isLAST;			//flag for last run of the packet
        int loadcounter=0;
        
        int bx = blockIdx.x;
        int tx = threadIdx.x; 
        
        searchWindow[tx] = ' ';
        StartPoint_win = tx;
        StartPoint_uncoded = tx;

        filepoint=0;
        wfilepoint=0;
        isLAST=0;
        
        __syncthreads();

        uncodedWindow[tx] = in_d[bx * PCKTSIZE + tx];
        filepoint+=MAX_CODED;
        
        //write data to shared memory
        searchWindow[ (StartPoint_win + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedWindow[StartPoint_uncoded];
        
        __syncthreads(); 

        uncodedWindow[MAX_CODED+tx] = in_d[bx * PCKTSIZE + filepoint + tx];
        
        filepoint+=MAX_CODED;

        __syncthreads();
        
        // Look for matching string//	
        matchInfo = FindMatch(StartPoint_win, StartPoint_uncoded,searchWindow,uncodedWindow,  tx, 0);
        __syncthreads();  
        
        //encode the rest of the file
        while ((filepoint) <= PCKTSIZE && !isLAST)
        {		
            if (matchInfo.length >= MAX_CODED)
                {
                        matchInfo.length = MAX_CODED-1;
                }

            //write ended bytes to encoded memory
            if (matchInfo.length <= MAX_UNCODED){
                matchInfo.length = 1;   // set to 1 for 1 byte uncoded //
                encodedBuf[tx*2] = 1;
                encodedBuf[tx*2 + 1] = uncodedWindow[StartPoint_uncoded];
            }
            else if(matchInfo.length > MAX_UNCODED){
                encodedBuf[tx*2] = (unsigned char)matchInfo.length;
                
                encodedBuf[tx*2+1] = (unsigned char)matchInfo.offset;			
            }

                

            //write the encoded data into output
            out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2] = encodedBuf[tx*2];
            out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedBuf[tx*2+1];
            
            
            wfilepoint = wfilepoint + MAX_CODED*2;
            
            StartPoint_win = (StartPoint_win + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
            StartPoint_uncoded = (StartPoint_uncoded + MAX_CODED) % (MAX_CODED*2);
            
            __syncthreads(); 	


            //update shared memory
            if(filepoint<PCKTSIZE){
                uncodedWindow[(StartPoint_uncoded+ MAX_CODED)% (MAX_CODED*2)] = in_d[bx * PCKTSIZE + filepoint + tx];
                filepoint+=MAX_CODED;
                //find the location for the thread specific view of window
                searchWindow[ (StartPoint_win + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedWindow[StartPoint_uncoded];
             }

            else{
                isLAST++;				
                searchWindow[(StartPoint_win + MAX_CODED ) % (WINDOW_SIZE+MAX_CODED)] = '^';		
            }
            __syncthreads(); 	
                
            loadcounter++;
            matchInfo = FindMatch(StartPoint_win, StartPoint_uncoded,searchWindow,uncodedWindow,tx, isLAST);
            
        } //end while
        

            if(isLAST==1)
            {
                if(matchInfo.length > (MAX_CODED - tx))
                    matchInfo.length = MAX_CODED - tx;
            }
            
            if (matchInfo.length >= MAX_CODED)
                {
                    // garbage beyond last data happened to extend match length //
                    matchInfo.length = MAX_CODED-1;
                }

            if (matchInfo.length <= MAX_UNCODED)
            {
                matchInfo.length = 1;   // set to 1 for 1 byte uncoded //
                encodedBuf[tx*2] = 1;
                encodedBuf[tx*2 + 1] = uncodedWindow[StartPoint_uncoded];
            }
            else if(matchInfo.length > MAX_UNCODED)
            {	
                encodedBuf[tx*2] = (unsigned char)matchInfo.length;
                encodedBuf[tx*2+1] = (unsigned char)matchInfo.offset;			
            }


            out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2] = encodedBuf[tx*2];
            out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedBuf[tx*2+1];
            //printf("%c",encodedBuf[tx*2]);
            //printf("%c",encodedBuf[tx*2+1]);

            //update written pointer and heads
            wfilepoint = wfilepoint + MAX_CODED*2;
            
            StartPoint_win = (StartPoint_win + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
            StartPoint_uncoded = (StartPoint_uncoded + MAX_CODED) % (MAX_CODED*2);
            
    }
'''

kw_kernel_decode = r'''
    __global__ void DecodeKernel(unsigned char * in_d, unsigned char * out_d)
    {  
        //unsigned int encodingFLAG;
        //__shared__ unsigned char buffer[32];

        //Take 32 bytes at a time.
        //Use 0 to indicate this is original data
        //Use 1 to indicate this is pointer data
        //For instance: 0b00111100 

        //encodingFLAG = 0x00000000;
        //int tx = threadIdx.x;
        //int bx = blockIdx.x; 

        int idx = blockIdx.x*blockDim.x + threadIdx.x;//blockDim : 32*1*1


        if (in_d[2*idx]==1){
            out_d[idx] = in_d[2*idx+1];
        }
        else{
            out_d[idx] = in_d[2*idx+1];
            out_d[idx+1] = in_d[2*idx];
        }            
    }
'''

