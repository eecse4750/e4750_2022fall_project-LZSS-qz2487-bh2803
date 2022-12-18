
kw_constant = r'''
    //initialize some constant value such as size for window and threasholds
    #define MIN_CODE_NUM     2
    #define MaxMatch       128
    #define WINDOW_SIZE     128
    #define PKTSIZE        4096
    #define TRUE            1
    #define FALSE           0
'''


typedef = r'''
    //define a struct here to combine the coded info
    typedef struct en_str_t
        {
            //offset i.e. start point address
            int offset;
            /length of longest match
            int length;
        } en_str_t;

'''


kw_matchfunc = r'''
    __device__ en_str_t MatchFunc(
        int StartPoint_win, 
        int StartPoint_uncoded, 
        unsigned char* searchWindow, 
        unsigned char* uncodedWindow,
        int tx, 
        int isLAST)
    {
        //Parameters initialization
        en_str_t matchInfo;
        int i, j;
        int isMAX;
        int matchingState=0;
        int loop=0;
        
        //encoded string has default value 1
        //when length == 1, we give the original data
        matchInfo.length = 1; 
        matchInfo.offset = 1; 
        i = StartPoint_win ;  
        j = 0; //counter for matchings

        //TEST if this strng is the last one.
        isMAX = MaxMatch - tx*isLAST;
        
        //temp value to record offset value
        int tempi=0;
        while (loop<WINDOW_SIZE)
        {   //if match we increse j and set the match flag to 1
            if (searchWindow[i] == uncodedWindow[(StartPoint_uncoded+j)% (WINDOW_SIZE+MaxMatch)])
            {
                j++;
                matchingState=1;		
            }
            else
            {   //If doesn't match now but previous state is match 
                //and j is bigger than encoded length
                //we set j as the encoded legnth
                //we set tempi as the encoded offset
                //temp i 
                if(matchingState && j > matchInfo.length)
                {
                    matchInfo.length = j;
                    tempi=i-j;
                    if(tempi<0)
                        tempi+=WINDOW_SIZE+MaxMatch;
                    matchInfo.offset = tempi;
                }
                
                //reset j and matching state
                j=0;
                matchingState=0;
            }
            //update i and loop condition value
            i = (i + 1) % (WINDOW_SIZE+MaxMatch);
            loop++;	
            //stop while when loop parameter is bigger than max value
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
    __global__ void EncodeKernel(unsigned char * input, unsigned char * output)
        {
        //initiate search window and uncoded window in shared memory
        __shared__ unsigned char searchWindow[WINDOW_SIZE+(MaxMatch)];
        __shared__ unsigned char uncodedWindow[MaxMatch*2];
        __shared__ unsigned char encodedBuf[MaxMatch*2];
        en_str_t matchInfo;

        //initialize parameters
        int StartPoint_win, StartPoint_uncoded;
        int filepoint;
        int wfilepoint;
        int isLAST;	
        int loadcounter=0;

        //intialize thread and block id
        int bx = blockIdx.x;
        int tx = threadIdx.x; 

        //fill search window array with some known value
        searchWindow[tx] = ' ';
        StartPoint_win = tx;
        StartPoint_uncoded = tx;

        filepoint=0;
        wfilepoint=0;
        isLAST=0;
        
        __syncthreads();

        //We have two modules and we combine them together for all input data
        //Module 1: to fill in the data to the window
        //Module 2: to encode data and write encoded data back to output array

        //Module 1: fill the window with input data
        /****************************************/
        uncodedWindow[tx] = input[bx * PKTSIZE + tx];
        //add start file point
        filepoint+=MaxMatch;
        
        //write data to shared memory
        searchWindow[ (StartPoint_win + WINDOW_SIZE ) % (WINDOW_SIZE + MaxMatch) ] = uncodedWindow[StartPoint_uncoded];
        
        __syncthreads(); 

        uncodedWindow[MaxMatch+tx] = input[bx * PKTSIZE + filepoint + tx];
        
        filepoint+=MaxMatch;

        __syncthreads();
        /*****************************************/

        // Look for match info//
        matchInfo = MatchFunc(StartPoint_win, StartPoint_uncoded,searchWindow,uncodedWindow,  tx, 0);
        __syncthreads();  
        
        //encode the rest of the file
        while ((filepoint) <= PKTSIZE && !isLAST)
        {   //module 2: check Match and encode data
            /***********************************************/
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
                output[bx * PKTSIZE*2 + wfilepoint + tx*2] = encodedBuf[tx*2];
                output[bx * PKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedBuf[tx*2+1];
                
                
                //update start point and write file point
                wfilepoint = wfilepoint + MaxMatch*2;
                StartPoint_win = (StartPoint_win + MaxMatch) % (WINDOW_SIZE+MaxMatch);
                StartPoint_uncoded = (StartPoint_uncoded + MaxMatch) % (MaxMatch*2);
                
                __syncthreads(); 
            }
            /************************************************/


            //Module 1:fill the window with input data
            /************************************/
            {
                //update shared memory
                if(filepoint<PKTSIZE){
                    uncodedWindow[(StartPoint_uncoded+ MaxMatch)% (MaxMatch*2)] = input[bx * PKTSIZE + filepoint + tx];
                    filepoint+=MaxMatch;
                    searchWindow[ (StartPoint_win + WINDOW_SIZE ) % (WINDOW_SIZE + MaxMatch) ] = uncodedWindow[StartPoint_uncoded];
                }

                else{//LAST one
                    searchWindow[(StartPoint_win + MaxMatch ) % (WINDOW_SIZE+MaxMatch)] = '^';
                    isLAST++;
                }
                __syncthreads(); 
            }
            /*************************************/

            // Look for match info//
            matchInfo = MatchFunc(StartPoint_win, StartPoint_uncoded,searchWindow,uncodedWindow,tx, isLAST);
            
        } //end while
        
            //Module 2:check Match and encode data last time
            /****************************************/
            {
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
            
                output[bx * PKTSIZE*2 + wfilepoint + tx*2] = encodedBuf[tx*2];
                output[bx * PKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedBuf[tx*2+1];
                //printf("%c",encodedBuf[tx*2]);
                //printf("%c",encodedBuf[tx*2+1]);
            
                //update written pointer and heads
                StartPoint_win = (StartPoint_win + MaxMatch) % (WINDOW_SIZE+MaxMatch);
                StartPoint_uncoded = (StartPoint_uncoded + MaxMatch) % (MaxMatch*2);
                wfilepoint = wfilepoint + MaxMatch*2;
            }
            /********************************************/
    }
'''

kw_kernel_decode = r'''
    __global__ void DecodeKernel(unsigned char * input, unsigned char * output)
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


        if (input[2*idx]==1){
            output[idx] = input[2*idx+1];
        }
        else{
            output[idx] = input[2*idx+1];
            output[idx+1] = input[2*idx];
        }            
    }
'''

