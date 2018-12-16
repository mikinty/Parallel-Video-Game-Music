# Markov Matrix Training

In order to run Markov Matrix Training, you need 2 input files - one for Major and one for Minor. These can be found as `notesMajor.txt` and `notesMinor.txt` in the `pre` folder.

There are 3 versions of matrix training. All versions use the `training.h` header file.

The parallel version consists of the files `main.cpp` and `cudaTraining.cu` In order to run the parallel version, you can use the `run.sh` file or run

```
> nvcc -std=c++11 main.cpp cudaTraining.cu -o main

> ./main notesMajor.txt notesMinor.txt
```

The sequential version consists of the file `mainSeq.cpp` In order to run this version, you can use the `seq.sh` file or run
```
> g++ -std=c++11 mainSeq.cpp -o mainSeq

> ./mainSeq notesMajor.txt notesMinor.txt
```

The testing version consists of the files `mainFileLoad.cpp` and `cudaTraining.cu.` It will run both a parallel and sequential version of the matrix training code, but should not be used to generate final matrices. In order to run this version, you can run

```
> nvcc -std=c++11 mainFileLoad.cpp cudaTraining.cu -o mainTest

> ./mainTest notesMajor.txt notesMinor.txt
```

In order to generate larger testing sets, `makeBig.py` can be used to copy information in text files many times to increase the overall number of lines.