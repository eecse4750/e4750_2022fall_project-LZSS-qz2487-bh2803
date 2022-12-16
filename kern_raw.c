
// Window size is how far back each pattern tries to match
// Lookahead is also length of maximum match
#define WINDOW_SIZE 1024
#define LOOKAHEAD_SIZE 255
#define MIN_MATCH 4

/*
Each thread matches pattern starting at a certain index.
At this point, the output is always 2x input size.

Another kernel taesk this output and sequentially
constructs the final compressed output.
*/
// ASSUME WINDOW_SIZE EQUALS NUM THREADS PER BLOCK
__global__ void kw_encode(char* input, char* output, int input_size) {
    // Indexing Stuff
    int baseIdx = blockIdx.x * blockDim.x;
    int gIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared Memory Optimization
    __shared__ char in_s[WINDOW_SIZE*3];

    for (int i=0; i<3; i++) {
        int rIdx = gIdx + (i-1) * WINDOW_SIZE;
        if(rIdx > 0 && rIdx < input_size){
            int wIdx = threadIdx.x + i * WINDOW_SIZE;
            in_s[wIdx] = input[rIdx];
        }
    }

    // Do Encode

    // Index for this thread's initial location in shared memory
    int sharedBase = WINDOW_SIZE + threadIdx.x;

    int bestMatchLength = 0;
    int bestMatchOffset = 0;
    int cMatchLen = 0;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        __syncthreads();
        // See if pattern continues to match
        char curWindowChar = in_s[threadIdx.x + i];             // From search window
        char curLookaheadChar = in_s[sharedBase + cMatchLen];   // From lookahead
        
        if(curWindowChar == curLookaheadChar){
            cMatchLen += 1;                                     // Matched
        }else{
            if(curWindowChar == in_s[sharedBase]){              // Reset cMatch
                cMatchLen = 1;                                  
            }else{
                cMatchLen = 0;
            }
        }

        if(cMatchLen > bestMatchLength){                        // Update if better than best
            bestMatchLength = cMatchLen;
            bestMatchOffset = WINDOW_SIZE - i + cMatchLen - 1;
        }
    }

    // Write to Output
    // Output is Twice Length of Input
    // Pairs of Offset and Length
    // Even Index Offset | Odd Index Length
    if(bestMatchLenth < MIN_MATCH){
        output[2*gIdx] = 0;
    }else{
        output[2*gIdx] = bestMatchOffset;
    }
    output[2*gIdx + 1] = bestMatchLength;

    return;

}