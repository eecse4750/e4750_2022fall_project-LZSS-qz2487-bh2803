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

# Detailed instructions:
1. ``kernel_old.py`` and ``kernel.py`` contains all the cuda kernel code in our project.
2. To test our code,  
run ``python project.py`` to test random number file with different sizes
run ``python project.py -d`` to test different kinds of files
You can also add ``-a`` like ``python project.py -a`` to give a test on random number files with extra naive CPU test.

&ensp;&ensp;&ensp;&ensp;**Note**: Naive implementation is very slow since it has dynamic array.

3. The project code has to be submitted inside this repository 

## (Re)naming of the project repository shared by multiple students (TODO students)
INSTRUCTIONS for (re)naming the students' solution repository for assignments with more students, such as the final project. Students need to use a 4-letter groupID: 
* Template: e4750-2022Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4750-2022Fall-Project-MEME-zz9999-aa9999-aa0000.

## Other Instructions
1. See detailed instructions in the E4750.2022Fall.ProjectInstructions
2. The report as a pdf file has to submitted twice - once in the github, once uploaded into the courseworks
3. The code has to be submitted only in github
4. Final project slides have to be google slides in google/liondrive E4750_2022FallProjects
5. There will be no due date extensions, slip days do not apply for project
