
TEST = False

class LZSSC:
    ENCODING = "utf-8"
    MAX_WINDOW_SIZE = 128
    ENCODED_FLAG = 0b1

    def __init__(self):
        pass

    def compress2(self, inData):
        inBytes = inData.encode(self.ENCODING)
        print(inBytes)

    def compress(self, inStr):

        print("INPUT:\n", inStr)

        def elements_in_array(check_elements, elements):
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

        # in data is string
        inBytes = inStr.encode(self.ENCODING)
        print("Input Bytes:\n",inBytes)
        print("Input Bytes Type:\n",type(inBytes[0]))

        searchBuf = []
        checkBuf = []
        output = []

        i=0
        for char in inBytes:
            index = elements_in_array(checkBuf, searchBuf)
            print("\n\t->|",char,type(char),i)

            if elements_in_array(checkBuf + [char], searchBuf) == -1 or i == len(inBytes) - 1:
                print("\t\t >_< End of Match")
                
                if i == len(inBytes) - 1 and elements_in_array(checkBuf + [char], searchBuf) != -1:
                    checkBuf.append(char)

                if len(checkBuf) > 1:
                    index = elements_in_array(checkBuf, searchBuf)
                    offset = i - index - len(checkBuf)
                    length = len(checkBuf)

                    token = f"~{offset},{length}"

                    if len(token) > length:
                        print("\n\t\tAdd Literal:", checkBuf)
                        output.extend(checkBuf)
                    else:
                        print("\n\t\tAdd Token:", token.encode(self.ENCODING))
                        output.extend(token.encode(self.ENCODING))

                    searchBuf.extend(checkBuf)
                else:
                    output.extend(checkBuf)
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


if TEST:
    teststr = r'''
    I AM SAM. I AM SAM. SAM I AM.

    THAT SAM-I-AM! THAT SAM-I-AM! I DO NOT LIKE THAT SAM-I-AM!

    DO WOULD YOU LIKE GREEN EGGS AND HAM?

    I DO NOT LIKE THEM,SAM-I-AM.
    I DO NOT LIKE GREEN EGGS AND HAM.
    '''

   # teststr = "abcdef abcdef12"

    encoder = LZSSC()
    out = encoder.compress(teststr)
    print("\n= = = = = = = = = =\n")
    print(out)

