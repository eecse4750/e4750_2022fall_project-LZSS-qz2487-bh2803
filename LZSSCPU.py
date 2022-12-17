import numpy as np

class LZSSCPU():
    ENCODING = "utf-8"
    MAX_WINDOW_SIZE = 1024
    #ENCODED_FLAG = 0b1

    def __init__(self):
        self.DEBUG_LEVEL = 0

    def elements_in_array(self,check_elements, elements):
        i = 0
        offset = 0

        for element in elements:
            if len(check_elements) <= offset:
                return i - len(check_elements)
            
            if check_elements[offset] == element:
                offset += 1
            else:
                offset = 0

            i += 1
        return -1


    def compress(self, inStr):

        #print("INPUT:\n", inStr)

        # in data is string and convert it to bytestring
        if (isinstance(inStr,str)):
            inBytes = inStr.encode(self.ENCODING)
        elif (isinstance(inStr,np.ndarray)):
            inBytes = inStr.astype(bytes)
        if(self.DEBUG_LEVEL > 0):
            print("Input Bytes:\n",inBytes)
            print("Input Bytes Type:\n",type(inBytes[0]))

        #Initiate Buffer and Output array
        searchBuf = []
        checkBuf = []
        output = []

        i=0
        for char in inBytes:
            index = self.elements_in_array(checkBuf, searchBuf)
            if(self.DEBUG_LEVEL > 0):
                print("\n\t->|",char,type(char),i)

            if self.elements_in_array(checkBuf + [char], searchBuf) == -1 or i == len(inBytes) - 1:
                if(self.DEBUG_LEVEL > 0):
                    print("\t\t >_< End of Match")
                
                if i == len(inBytes) - 1 and self.elements_in_array(checkBuf + [char], searchBuf) != -1:
                    checkBuf.append(char)

                if len(checkBuf) > 1:
                    index = self.elements_in_array(checkBuf, searchBuf)
                    offset = i - index - len(checkBuf)
                    length = len(checkBuf)

                    token = f"~{offset},{length}"

                    if len(token) > length:
                        if(self.DEBUG_LEVEL > 0):
                            print("\n\t\tAdd Literal:", checkBuf)
                        output.extend(checkBuf)
                    else:
                        if(self.DEBUG_LEVEL > 0):
                            print("\n\t\tAdd Token:", token.encode(self.ENCODING))
                        output.extend(token.encode(self.ENCODING))

                    searchBuf.extend(checkBuf)
                else:
                    output.extend(checkBuf)
                    if(self.DEBUG_LEVEL > 0):
                        print("\t\tAdd Literal",output, "<-", checkBuf)
                    searchBuf.extend(checkBuf)

                checkBuf = []

            checkBuf.append(char)

            if i == len(inBytes) - 1 and len(checkBuf) > 0:
                output.extend(checkBuf)

            if len(searchBuf) > self.MAX_WINDOW_SIZE:
                searchBuf = searchBuf[1:]

            i += 1


        return bytes(output)


if __name__ == "__main__":
    teststr = r'''
    I AM SAM. I AM SAM. SAM I AM.

    THAT SAM-I-AM! THAT SAM-I-AM! I DO NOT LIKE THAT SAM-I-AM!

    DO WOULD YOU LIKE GREEN EGGS AND HAM?

    I DO NOT LIKE THEM,SAM-I-AM.
    I DO NOT LIKE GREEN EGGS AND HAM.
    '''

    #teststr = "abcdef abcdef12"
    #teststr = testnum.teststr
    #teststr = shpt.shpt
    #teststr = sashimi.sashimi

    encoder = LZSSCPU()
    file_list_r = [*teststr]
    file_arr_r = np.array(file_list_r).astype(bytes)
    out = encoder.compress(teststr)
    print("\n= = = = = = = = = =\n")
    print(out)

    
    print("\n= = = = = = = = = =\n")
    print("Input Size: %d\t\tOutput Size: %d\n"%(len(teststr), len(out)))

