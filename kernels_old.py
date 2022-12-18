
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
        i = StartPoint_win ;  // start at the beginning of the sliding window //
        j = 0; //counter for matchings

        
        //if(lastcheck) 
            maxcheck = MAX_CODED - tx*lastcheck;
        //else
        //	maxcheck = MAX_CODED;
        
        int tempi=0;
        while (loop<WINDOW_SIZE)
        {
            if (searchWindow[i] == uncodedLookahead[(StartPoint_uncoded+j)% (WINDOW_SIZE+MAX_CODED)])
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
'''


kw_kernel_encode = r'''
    __global__ void EncodeKernel(unsigned char * in_d, unsigned char * out_d)
        {

        __shared__ unsigned char searchWindow[WINDOW_SIZE+(MAX_CODED)];
        __shared__ unsigned char uncodedLookahead[MAX_CODED*2];
        __shared__ unsigned char encodedBuf[MAX_CODED*2];
        en_str_t matchData;
    
        int StartPoint_win, StartPoint_uncoded;    // head of sliding window and lookahead //
        int filepoint;			//file index pointer for reading
        int wfilepoint;			//file index pointer for writing
        int lastcheck;			//flag for last run of the packet
        int loadcounter=0;
        
        int bx = blockIdx.x;
        int tx = threadIdx.x; 
        
        searchWindow[tx] = ' ';
        StartPoint_win = tx;
        StartPoint_uncoded = tx;

        filepoint=0;
        wfilepoint=0;
        lastcheck=0;
        
        __syncthreads();

        //uncodedLookahead[tx] = tex1Dfetch(in_d_tex, bx * PCKTSIZE + tx); //in_d[bx * PCKTSIZE + tx];
        uncodedLookahead[tx] = in_d[bx * PCKTSIZE + tx];
        filepoint+=MAX_CODED;
        
        searchWindow[ (StartPoint_win + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedLookahead[StartPoint_uncoded];
        //tex1Dfetch(in_d_tex, bx * PCKTSIZE + tx);//uncodedLookahead[StartPoint_uncoded];
        
        __syncthreads(); 
        
        //uncodedLookahead[MAX_CODED+tx] = tex1Dfetch(in_d_tex, bx * PCKTSIZE + filepoint + tx); //in_d[bx * PCKTSIZE + filepoint + tx];

        uncodedLookahead[MAX_CODED+tx] = in_d[bx * PCKTSIZE + filepoint + tx];
        
        filepoint+=MAX_CODED;
        

        
        __syncthreads();
        
        loadcounter++;
        // Look for matching string in sliding window //	
        matchData = FindMatch(StartPoint_win, StartPoint_uncoded,searchWindow,uncodedLookahead,  tx, bx, 0, 0,loadcounter);
        __syncthreads();  
        
        //encode the rest of the file
        while ((filepoint) <= PCKTSIZE && !lastcheck)
        {		
        
            
            
            if (matchData.length >= MAX_CODED)
                {
                        matchData.length = MAX_CODED-1;
                }

            if (matchData.length <= MAX_UNCODED)
            {
                matchData.length = 1;   // set to 1 for 1 byte uncoded //
                encodedBuf[tx*2] = 1;
                encodedBuf[tx*2 + 1] = uncodedLookahead[StartPoint_uncoded];
            }
            else if(matchData.length > MAX_UNCODED)
            {	//printf("%d\n",matchData.offset);
                encodedBuf[tx*2] = (unsigned char)matchData.length;
                
                encodedBuf[tx*2+1] = (unsigned char)matchData.offset;			
            }

                
            //write out the encoded data into output
            out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2] = encodedBuf[tx*2];
            out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedBuf[tx*2+1];
            
            
            //update written pointer and heads
            wfilepoint = wfilepoint + MAX_CODED*2;
            
            StartPoint_win = (StartPoint_win + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
            StartPoint_uncoded = (StartPoint_uncoded + MAX_CODED) % (MAX_CODED*2);
            
            __syncthreads(); 	

            {
                if(filepoint<PCKTSIZE){
                    uncodedLookahead[(StartPoint_uncoded+ MAX_CODED)% (MAX_CODED*2)] = in_d[bx * PCKTSIZE + filepoint + tx];
                    filepoint+=MAX_CODED;
                    
                    //find the location for the thread specific view of window
                    searchWindow[ (StartPoint_win + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedLookahead[StartPoint_uncoded];
                    //__syncthreads();                 }
                else{
                    lastcheck++;				
                    searchWindow[(StartPoint_win + MAX_CODED ) % (WINDOW_SIZE+MAX_CODED)] = '^';		
                }
                __syncthreads(); 	
                
                loadcounter++;
                matchData = FindMatch(StartPoint_win, StartPoint_uncoded,searchWindow,uncodedLookahead,tx,bx, wfilepoint, lastcheck,loadcounter);
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
                matchData.length = 1;   // set to 1 for 1 byte uncoded //
                encodedBuf[tx*2] = 1;
                encodedBuf[tx*2 + 1] = uncodedLookahead[StartPoint_uncoded];
            }
            else if(matchData.length > MAX_UNCODED)
            {	
                encodedBuf[tx*2] = (unsigned char)matchData.length;
                encodedBuf[tx*2+1] = (unsigned char)matchData.offset;			
            }

                
            //write out the encoded data into output

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

