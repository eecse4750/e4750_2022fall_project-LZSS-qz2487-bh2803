
kw_constant = r'''
    #define MIN_CODE_NUM     2
    #define MaxMatch       128
    #define WINDOW_SIZE     128
    #define PKTSIZE        4096
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
        
        matchInfo.length = 1; 
        matchInfo.offset = 1; 
        i = StartPoint_win ;  
        j = 0; //counter for matchings

        
        isMAX = MaxMatch - tx*isLAST;
        
        int tempi=0;
        while (loop<WINDOW_SIZE)
        {
            if (searchWindow[i] == uncodedWindow[(StartPoint_uncoded+j)% (WINDOW_SIZE+MaxMatch)])
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
                        tempi+=WINDOW_SIZE+MaxMatch;
                    matchInfo.offset = tempi;
                }
                
                j=0;
                matchingState=0;		
            }
        
            i = (i + 1) % (WINDOW_SIZE+MaxMatch);
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
                tempi+=WINDOW_SIZE+MaxMatch;

            matchInfo.offset = tempi;
        }
            
        
        return matchInfo;
    }
'''


kw_kernel_encode = r'''
    __global__ void EncodeKernel(unsigned char * in_d, unsigned char * out_d)
        {

        __shared__ unsigned char searchWindow[WINDOW_SIZE+(MaxMatch)];
        __shared__ unsigned char uncodedWindow[MaxMatch*2];
        __shared__ unsigned char encodedBuf[MaxMatch*2];
        en_str_t matchInfo;
    
        int StartPoint_win, StartPoint_uncoded;
        int filepoint;
        int wfilepoint;
        int isLAST;	
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

        uncodedWindow[tx] = in_d[bx * PKTSIZE + tx];
        filepoint+=MaxMatch;
        
        //write data to shared memory
        searchWindow[ (StartPoint_win + WINDOW_SIZE ) % (WINDOW_SIZE + MaxMatch) ] = uncodedWindow[StartPoint_uncoded];
        
        __syncthreads(); 

        uncodedWindow[MaxMatch+tx] = in_d[bx * PKTSIZE + filepoint + tx];
        
        filepoint+=MaxMatch;

        __syncthreads();
        
        // Look for matching string//	
        matchInfo = FindMatch(StartPoint_win, StartPoint_uncoded,searchWindow,uncodedWindow,  tx, 0);
        __syncthreads();  
        
        //encode the rest of the file
        while ((filepoint) <= PKTSIZE && !isLAST)
        {		
            if (matchInfo.length >= MaxMatch)
                {
                        matchInfo.length = MaxMatch-1;
                }

            //write ended bytes to encoded memory
            if (matchInfo.length <= MIN_CODE_NUM){
                matchInfo.length = 1;   // set to 1 for 1 byte uncoded //
                encodedBuf[tx*2] = 1;
                encodedBuf[tx*2 + 1] = uncodedWindow[StartPoint_uncoded];
            }
            else if(matchInfo.length > MIN_CODE_NUM){
                encodedBuf[tx*2] = (unsigned char)matchInfo.length;
                
                encodedBuf[tx*2+1] = (unsigned char)matchInfo.offset;			
            }

                

            //write the encoded data into output
            out_d[bx * PKTSIZE*2 + wfilepoint + tx*2] = encodedBuf[tx*2];
            out_d[bx * PKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedBuf[tx*2+1];
            
            
            wfilepoint = wfilepoint + MaxMatch*2;
            
            StartPoint_win = (StartPoint_win + MaxMatch) % (WINDOW_SIZE+MaxMatch);
            StartPoint_uncoded = (StartPoint_uncoded + MaxMatch) % (MaxMatch*2);
            
            __syncthreads(); 	


            //update shared memory
            if(filepoint<PKTSIZE){
                uncodedWindow[(StartPoint_uncoded+ MaxMatch)% (MaxMatch*2)] = in_d[bx * PKTSIZE + filepoint + tx];
                filepoint+=MaxMatch;
                searchWindow[ (StartPoint_win + WINDOW_SIZE ) % (WINDOW_SIZE + MaxMatch) ] = uncodedWindow[StartPoint_uncoded];
             }

            else{
                isLAST++;				
                searchWindow[(StartPoint_win + MaxMatch ) % (WINDOW_SIZE+MaxMatch)] = '^';		
            }
            __syncthreads(); 	
                
            loadcounter++;
            matchInfo = FindMatch(StartPoint_win, StartPoint_uncoded,searchWindow,uncodedWindow,tx, isLAST);
            
        } //end while
        

            if(isLAST==1)
            {
                if(matchInfo.length > (MaxMatch - tx))
                    matchInfo.length = MaxMatch - tx;
            }
            
            if (matchInfo.length >= MaxMatch)
                {
                    // garbage beyond last data happened to extend match length //
                    matchInfo.length = MaxMatch-1;
                }

            if (matchInfo.length <= MIN_CODE_NUM)
            {
                matchInfo.length = 1;   // set to 1 for 1 byte uncoded //
                encodedBuf[tx*2] = 1;
                encodedBuf[tx*2 + 1] = uncodedWindow[StartPoint_uncoded];
            }
            else if(matchInfo.length > MIN_CODE_NUM)
            {	
                encodedBuf[tx*2] = (unsigned char)matchInfo.length;
                encodedBuf[tx*2+1] = (unsigned char)matchInfo.offset;			
            }


            out_d[bx * PKTSIZE*2 + wfilepoint + tx*2] = encodedBuf[tx*2];
            out_d[bx * PKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedBuf[tx*2+1];
            //printf("%c",encodedBuf[tx*2]);
            //printf("%c",encodedBuf[tx*2+1]);

            //update written pointer and heads
            wfilepoint = wfilepoint + MaxMatch*2;
            
            StartPoint_win = (StartPoint_win + MaxMatch) % (WINDOW_SIZE+MaxMatch);
            StartPoint_uncoded = (StartPoint_uncoded + MaxMatch) % (MaxMatch*2);
            
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

