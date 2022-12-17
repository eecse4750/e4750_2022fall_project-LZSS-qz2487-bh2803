# e4750_2022Fall_Project_LZSS_qz2487_bh2803
This is the repo for our final project of EECSE4750 Heterogeneous Computing. Our topic is LZSS parallel compression. We use a parallel implementation of LZSS algorithm to comrpess data.

## File Organization
```
├── project.py
├── README.md
├── LZSSCPU.py
├── testfile
│   ├── big.txt
│   ├── random1mb.txt
│   ├── random5mb.txt
|   ├── random10mb.txt
│   └── random50mb.txt
├── result
│   └── ... (result bin files)
├── image
│   └── ... (result jpg files)
├── canterbury
│   ├── SHA1SUM
│   ├── alice29.txt
│   ├── cp.html
│   ├── fields.c
│   ├── grammar.lsp
│   ├── kennedy.xls
│   ├── lcet10.txt
│   ├── plrabn12.txt
│   ├── ptt5
│   ├── sum
│   └── xargs.1
├── kernel_old.py
└── kernel.py
```

# Detailed instructions to run our project:
1. ``kernel_old.py`` and ``kernel.py`` contains all the cuda kernel code in our project.
2. To test our code,  
* run ``python project.py`` to test random number file with different sizes
* run ``python project.py -d`` to test different kinds of files
* add ``-a`` like ``python project.py -a`` to give a test on random number files with extra naive CPU test.
**Note**: Naive implementation is very slow since it used dynamic arrays.
* add ``-f`` like ``python project.py -f FILENAME`` to test your own file.
**Note**: DIY file doesn't give a plot back to folder ``image``
## File Details
1. Canterbury Corpus data is contained in ``canterbury`` folder. The Canterbury Corpus is a benchmark to enable researchers to evaluate lossless compression methods. This site includes test files and compression test results for many research compression methods.
2. ``testfile`` folder contains the random number data with differnet sizes.
3. ``result`` folder contains the result bin files which are not able to be opened. It can be decoded back to original file with proper implementation.
4. ``iamge`` folder returns the image file we produced during execution.
