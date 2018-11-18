#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define NOTES_PER_THREAD 10

typedef struct note {
	int noteNumber;
	int noteLength;
	int startTime;
}note_t;

//Transforms a (noteTone, noteLength) pair to a matrix index
//where length ranges from 1 - 16 (where x = x/16 of a measure)
//and tone ranges from 1 - 24 (2 octaves, start at C)
static inline int notePairToIndex(int tone, int length)
{
	return tone * length - 1;
}

//Transforms 2 (noteTone, noteLength) pairs to a matrix index
static inline int doubleLayerPairToIndex(int tone1, int length1, int tone2, int length2)
{
	return tone1 * length1 * tone2 * length2 - 1;
}

//Gets flattened matrix index
static inline int flatten(int i, int j){
	return i * 24 * 16 + j;
}

__global__ void CountSection(note_t* notes, int N, double** M)
{
	int start = threadIdx.x * NOTES_PER_THREAD + 2;
    int end = start + NOTES_PER_THREAD;

    int prevTone1 = notes[start-2].noteNumber;
    int prevLength1 = notes[start-2].noteLength;
    int curTone = notes[start-1].noteNumber;
    int curLength = notes[start-1].noteLength;
    
    for (int noteIndex = start; noteIndex < end; noteIndex++){
    	if (noteIndex > N)
    		return;
    	
    	int prevTone2 = prevTone1;
    	int prevLength2 = prevLength1;
    	prevTone1 = curTone;
        prevLength1 = curLength;
        curTone = notes[start].noteNumber;
        curLength = notes[start].noteLength;

        double* cell = &M[flatten(doubleLayerPairToIndex(prevTone1, prevLength1, prevTone2, prevLength2),notePairToIndex(curTone, curLength))];

        atomicAdd(cell, 1);
    }
}

//Takes in an array of notes (in order by startTime), the length of said array,
// and the transitionMatrix that corresponds to this
//array's theme and part
void countTransitionMatrix(note_t* notes, int length, double* transitionM){
	int numThreads = ((length - 2) + NOTES_PER_THREAD - 1)/NOTES_PER_THREAD;

    CountSection<<<numThreads, 1>>>(notes, length, transitionM);
}

__global__ void SumAndDivide(double* M)
{
	int i = threadIdx.x;
    double total = 0;

	for (int j = 0; j < 24 * 16; j ++){
        total = total + M[flatten(i, j)];
	}
	for (int j = 24*16 - 1; j >= 0; j --){
	    M[flatten(i,j)] = M[flatten(i,j)] / total;
	}
}

//Takes in the count for transition matrix, and turns it into
//a valid probability matrix instead
void createProbabilityMatrix(double* M){
	int numThreads = 24 * 16 * 24 * 16; //Number of rows

	SumAndDivide<<<numThreads, 1>>>(M);
}

void buildMatrixFromNotes(note_t* notes, int length, double* device_M)
{
    note_t* device_notes;

	cudaMalloc((void **)&device_notes, sizeof(note_t) * length);
	cudaMemcpy(device_notes, notes, sizeof(note_t) * length, cudaMemcpyHostToDevice);

	CountTransitionMatrix(device_notes, length, device_M);

	cudaThreadSynchronize();

	cudaFree(device_notes);
}

//Takes in a place to put the final matrix, a list of files
//to read, and 
//uses Cuda to actually compute it
void buildMatrix(double* finalMatrix, string* files, int lengthFiles){
	double* device_M;
	cudaMalloc((void**)&device_M, sizeof(double) * 24 * 16 * 24 * 16 * 24 * 16);

    cudaMemset(device_M, 0, sizeof(double) * 24 * 16 * 24 * 16 * 24 * 16);

    for (int i = 0; i < lengthFiles, i++){
        note_t* notes = readFile(files[i]);
        buildMatrixFromNotes(notes, length, device_M);
    }

    createProbabilityMatrix(device_M);
    cudaThreadSnchronize();
    cudaMemcpy(finalMatrix, device_M, sizeof(double) * 24 * 16 * 24 * 16 * 24 * 16, cudaMemcpyDeviceToHost);
    cudaFree(device_M);
}

//Given a filename, stores the notes in the file
note_t* readFile(string fileName){
	//Skip first line

	//Read other lines and start storing into an array
}

//Reads in a list of filenames, sorts them
//by theme and part, and calls BuildMatrix to build
//each matrix
void sortFiles(){
	//Read list of files from somewhere

	//Based on first line of each file, sort into 20 different arrays,
	//based on part and theme

	//For each of these 20 arrays, set up and call buildMatrix on them

    //Write the results to 20 different text files
}