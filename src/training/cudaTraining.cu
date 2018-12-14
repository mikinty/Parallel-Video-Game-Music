#include <string>
#include <algorithm>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "training.h"

// Define global variables
int* deviceMajorHighNotes[NUM_GPU_PER_MATRIX];
int* deviceMajorLowNotes[NUM_GPU_PER_MATRIX];
int* deviceMajorChords;
int* deviceMinorHighNotes[NUM_GPU_PER_MATRIX];
int* deviceMinorLowNotes[NUM_GPU_PER_MATRIX];
int* deviceMinorChords;

cudaStream_t majorHighStream[NUM_GPU_PER_MATRIX];
cudaStream_t majorLowStream[NUM_GPU_PER_MATRIX];
cudaStream_t majorChordSStream;
cudaStream_t majorChordBStream;
cudaStream_t minorHighStream[NUM_GPU_PER_MATRIX];
cudaStream_t minorLowStream[NUM_GPU_PER_MATRIX];
cudaStream_t minorChordSStream;
cudaStream_t minorChordBStream;

sound_t* deviceMajorSoprano[NUM_GPU_PER_MATRIX + 1];
sound_t* deviceMajorBass[NUM_GPU_PER_MATRIX + 1];
sound_t* deviceMinorSoprano[NUM_GPU_PER_MATRIX + 1];
sound_t* deviceMinorBass[NUM_GPU_PER_MATRIX + 1];

/**
 * @brief Transforms 2 (noteTone, noteLength) pairs to a matrix index for a melody matrix,
 * and returns -1 if the row is not in the correct device
 * 
 * @param curTone tone of the current note
 * @param curDur duration of the current note
 * @param prevTone1 tone of the previous note
 * @param prevDur1 duration of the previous note
 * @param prevTone2 tone of the note before previous note
 * @param prevDur2 duration of the note before previous note
 * @param deviceI index of the GPU device we are in
 */
__device__ 
inline int findNoteCell(int curTone, int curDur, int prevTone1, int prevDur1, int prevTone2, int prevDur2, int deviceI)
{
  // current note 
  int col = curTone * NUM_DUR + curDur;

  //If previous tones are chords, get top note and find closest
  if (prevTone1 >= CHORD_OFFSET){
      prevTone1 = (prevTone1 - CHORD_OFFSET) / 144; //get top chord note
      prevTone1 = curTone - (curTone % 12) + prevTone1; //find closest prevTone1 note
  }
  if (prevTone2 >= CHORD_OFFSET){
      prevTone2 = (prevTone2 - CHORD_OFFSET) / 144; //get top chord note
      prevTone2 = curTone - (curTone % 12) + prevTone2; //find closest note
  }

  int row = ((prevTone1 * NUM_DUR) + prevDur1) * NUM_NOTES
    + ((prevTone2 * NUM_DUR) + prevDur2);

  if (row < deviceI * MATRIX_BLOCK_ROWS || row >= (deviceI + 1) * MATRIX_BLOCK_ROWS)
    return -1;

  return (row - deviceI * MATRIX_BLOCK_ROWS) * NUM_NOTES + col;
}

/**
 * @brief Transforms 2 tones into a matrix index for the chord matrix
 * 
 * @param curTone tone of the current note
 * @param prevTone tone of the previous note
 */
__device__
inline int findChordCell(int curTone, int prevTone){
	
  if (prevTone >= CHORD_OFFSET) { //is a chord
    prevTone = prevTone - CHORD_OFFSET; //shift chord down
  }
  else { //is a note
    prevTone = prevTone % 12;
  }
  
  return prevTone * NUM_CHORDS + (curTone - CHORD_OFFSET);
}

/**
 * @brief Depending on the threadIndex and GPU device index, 
 * counts note transitions in a 
 * section of the given notes array and adds
 * to existing counts on the given matrices
 * 
 * @param part notes array to count
 * @param length notes array length
 * @param deviceMatrix matrix for inserting into
 * @param deviceI index of the current GPU device
 */
__global__ 
void CountNoteSection(sound_t* part, int length, int* deviceMatrix, int deviceI)
{
  //Bounds for section to read based on threadId
  int start;
  int end;

  //Determine proper start and end bounds
  start = threadIdx.x * (length / NUM_THREADS) + 2;
  if (threadIdx.x == NUM_THREADS - 1){ //If we are the last thread, make sure to grab any extra bits
    end = length;
  }
  else{
    end = start + length/NUM_THREADS;
  } 

  //Grab previous notes and durations
  int prevTone1 = part[start-2].tone;
  int prevDur1 = part[start-2].duration;
  int curTone = part[start-1].tone;
  int curDur = part[start-1].duration;
  //Counts notes for the assigned section
  for (int noteIndex = start; noteIndex < end; noteIndex++){
    //Grab correct tones and durations for current, previous, and note before previous
  	int prevTone2 = prevTone1;
  	int prevDur2 = prevDur1;
  	prevTone1 = curTone;
    prevDur1 = curDur;
    curTone = part[noteIndex].tone;
    curDur = part[noteIndex].duration;

    if (curTone < NUM_TONES) { //if not a chord, check device and insert
      int cell = findNoteCell(curTone, curDur, prevTone1, prevDur1, prevTone2, prevDur2, deviceI);
      if (cell != -1)
        atomicAdd(&deviceMatrix[cell], 1);   
    }
  } 
}

/**
 * @brief Depending on the threadIndex, counts chord transitions in a 
 * section of the given notes array and adds
 * to existing counts on the given matrices
 * 
 * @param part notes array to count
 * @param length notes array length
 * @param deviceMatrix matrix for chords
 */
__global__ 
void CountChordSection(sound_t* part, int length, int* deviceMatrix)
{
  //Bounds for section to read based on threadId
  int start;
  int end; 

  //Determine proper start and end bounds
  start = threadIdx.x * (length / NUM_THREADS) + 2;
  if (threadIdx.x == NUM_THREADS - 1){ //If we are the last thread, make sure to grab any extra bits
    end = length;
  }
  else {
    end = start + length/NUM_THREADS;
  }

  //Grab previous notes and durations
  int prevTone1 = part[start-2].tone;
  int curTone = part[start-1].tone;
  //Counts notes for the assigned section
  for (int noteIndex = start; noteIndex < end; noteIndex++){
    //Grab correct tones and durations for current, previous, and note before previous 	
  	prevTone1 = curTone;
    curTone = part[noteIndex].tone;

    int cell; //matrix index to insert into
    if (curTone >= CHORD_OFFSET) { //insert into chord matrix
      cell = findChordCell(curTone, prevTone1);
      atomicAdd(&deviceMatrix[cell], 1);
    }
  } 
}

/**
 * @brief Allocates device memory for the matrices and initializes them,
 * as well as sets up all streams
 */
void initCuda()
{
  //Allocate device memory for all melodic line matrices
  for (int offsetGPU = 0; offsetGPU < NUM_GPU_PER_MATRIX; offsetGPU ++){

    cudaSetDevice(offsetGPU + MAJOR_HIGH_DEVICE);
    cudaStreamCreate(&majorHighStream[offsetGPU]);
    cudaMalloc((void**)&deviceMajorSoprano[offsetGPU], sizeof(sound_t) * ARRAY_LENGTH);
    cudaMalloc((void**)&deviceMajorHighNotes[offsetGPU], sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES);
    cudaMemsetAsync(deviceMajorHighNotes[offsetGPU], 0, sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES, majorHighStream[offsetGPU]);

    cudaSetDevice(offsetGPU + MAJOR_LOW_DEVICE);
    cudaStreamCreate(&majorLowStream[offsetGPU]);
    cudaMalloc((void**)&deviceMajorBass[offsetGPU], sizeof(sound_t) * ARRAY_LENGTH);
    cudaMalloc((void**)&deviceMajorLowNotes[offsetGPU], sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES);
    cudaMemsetAsync(deviceMajorLowNotes[offsetGPU], 0, sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES, majorLowStream[offsetGPU]);

    cudaSetDevice(offsetGPU + MINOR_HIGH_DEVICE);
    cudaStreamCreate(&minorHighStream[offsetGPU]);
    cudaMalloc((void**)&deviceMinorSoprano[offsetGPU], sizeof(sound_t) * ARRAY_LENGTH);
    cudaMalloc((void**)&deviceMinorHighNotes[offsetGPU], sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES);
    cudaMemsetAsync(deviceMinorHighNotes[offsetGPU], 0, sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES, minorHighStream[offsetGPU]);

    cudaSetDevice(offsetGPU + MINOR_LOW_DEVICE);
    cudaStreamCreate(&minorLowStream[offsetGPU]);
    cudaMalloc((void**)&deviceMinorBass[offsetGPU], sizeof(sound_t) * ARRAY_LENGTH);
    cudaMalloc((void**)&deviceMinorLowNotes[offsetGPU], sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES);
    cudaMemsetAsync(deviceMinorLowNotes[offsetGPU], 0, sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES, minorLowStream[offsetGPU]);
  }

  //Allocate chord matrices

  cudaSetDevice(MAJOR_CHORD_DEVICE);
  cudaStreamCreate(&majorChordSStream);
  cudaStreamCreate(&majorChordBStream);
  cudaMalloc((void**)&deviceMajorSoprano[NUM_GPU_PER_MATRIX], sizeof(sound_t) * ARRAY_LENGTH);
  cudaMalloc((void**)&deviceMajorBass[NUM_GPU_PER_MATRIX], sizeof(sound_t) * ARRAY_LENGTH);
  cudaMalloc((void**)&deviceMajorChords, sizeof(int) * NUM_CHORDS * NUM_CHORDS); 
  cudaMemsetAsync(deviceMajorChords, 0, sizeof(int) * NUM_CHORDS * NUM_CHORDS, majorChordSStream);

  cudaSetDevice(MINOR_CHORD_DEVICE);
  cudaStreamCreate(&minorChordSStream);
  cudaStreamCreate(&minorChordBStream);
  cudaMalloc((void**)&deviceMinorSoprano[NUM_GPU_PER_MATRIX], sizeof(sound_t) * ARRAY_LENGTH);
  cudaMalloc((void**)&deviceMinorBass[NUM_GPU_PER_MATRIX], sizeof(sound_t) * ARRAY_LENGTH);
  cudaMalloc((void**)&deviceMinorChords, sizeof(int) * NUM_CHORDS * NUM_CHORDS); 
  cudaMemsetAsync(deviceMinorChords, 0, sizeof(int) * NUM_CHORDS * NUM_CHORDS, minorChordSStream);
}

/**
 * @brief Frees device memory and streams for all matrices
 * 
 */
void freeCuda()
{
  //Step through all GPUS working on the same matrix and free
  for (int offsetGPU = 0; offsetGPU < NUM_GPU_PER_MATRIX; offsetGPU ++){

    cudaSetDevice(offsetGPU + MAJOR_HIGH_DEVICE);
    cudaDeviceSynchronize();
    cudaStreamDestroy(majorHighStream[offsetGPU]);
    cudaFree(deviceMajorSoprano[offsetGPU]);
    cudaFree(deviceMajorHighNotes[offsetGPU]);

    cudaSetDevice(offsetGPU + MAJOR_LOW_DEVICE);
    cudaDeviceSynchronize();
    cudaStreamDestroy(majorLowStream[offsetGPU]);
    cudaFree(deviceMajorBass[offsetGPU]);
    cudaFree(deviceMajorLowNotes[offsetGPU]);

    cudaSetDevice(offsetGPU + MINOR_HIGH_DEVICE);
    cudaDeviceSynchronize();
    cudaStreamDestroy(minorHighStream[offsetGPU]);
    cudaFree(deviceMinorSoprano[offsetGPU]);
    cudaFree(deviceMinorHighNotes[offsetGPU]);

    cudaSetDevice(offsetGPU + MINOR_LOW_DEVICE);
    cudaDeviceSynchronize();
    cudaStreamDestroy(minorLowStream[offsetGPU]);
    cudaFree(deviceMinorBass[offsetGPU]);
    cudaFree(deviceMinorLowNotes[offsetGPU]);
  }

  //Free on the Chord GPUs

  cudaSetDevice(MAJOR_CHORD_DEVICE);
  cudaDeviceSynchronize();
  cudaStreamDestroy(majorChordSStream);
  cudaStreamDestroy(majorChordBStream);
  cudaFree(deviceMajorSoprano[NUM_GPU_PER_MATRIX]);
  cudaFree(deviceMajorBass[NUM_GPU_PER_MATRIX]);
  cudaFree(deviceMajorChords);
  
  cudaSetDevice(MINOR_CHORD_DEVICE);
  cudaDeviceSynchronize();
  cudaStreamDestroy(minorChordSStream);
  cudaStreamDestroy(minorChordBStream);
  cudaFree(deviceMinorSoprano[NUM_GPU_PER_MATRIX]);
  cudaFree(deviceMinorBass[NUM_GPU_PER_MATRIX]);
  cudaFree(deviceMinorChords);
}

/**
 * @brief Stalls until all stream of the given mood are clear
 * 
 * @param mood either major or minor, denoting which streams to check
 */
void cudaStreamSynch(int mood){
  //Check all copies of melodic matrices
  for (int offsetGPU = 0; offsetGPU < NUM_GPU_PER_MATRIX; offsetGPU ++){
    if (mood == 0) { //major
      cudaSetDevice(offsetGPU + MAJOR_HIGH_DEVICE);
      cudaStreamSynchronize(majorHighStream[offsetGPU]);
      cudaSetDevice(offsetGPU + MAJOR_LOW_DEVICE);
      cudaStreamSynchronize(majorLowStream[offsetGPU]);
    }
    else { //minor
      cudaSetDevice(offsetGPU + MINOR_HIGH_DEVICE);
      cudaStreamSynchronize(minorHighStream[offsetGPU]);
      cudaSetDevice(offsetGPU + MINOR_LOW_DEVICE);
      cudaStreamSynchronize(minorLowStream[offsetGPU]);
    }
  }

  //Check chord matrix
  if (mood == 0) { //major
    cudaSetDevice(MAJOR_CHORD_DEVICE);
    cudaStreamSynchronize(majorChordSStream);
    cudaStreamSynchronize(majorChordBStream);
  }
  else { //minor
    cudaSetDevice(MINOR_CHORD_DEVICE);
    cudaStreamSynchronize(minorChordSStream);
    cudaStreamSynchronize(minorChordBStream);
  }
}

/**
 * @brief Counts the note transitions in the given notes arrays in parallel
 * 
 * @param soprano notes array for soprano line
 * @param sLength length of soprano array
 * @param bass notes array for bass line
 * @param bLength length of bass array
 * @param mood marks if the soprano/bass arrays are in major or minor
 */
void countTransitionsCuda(sound_t* soprano, int sLength, sound_t* bass, int bLength, int mood){

  //Determine if we should use major or minor matrices, and call GPUS to count note transitions for melodic lines
  for (int offsetGPU = 0; offsetGPU < NUM_GPU_PER_MATRIX; offsetGPU ++) {
    if (mood == 0) { //major 
      cudaSetDevice(offsetGPU + MAJOR_HIGH_DEVICE);
      cudaMemcpyAsync(deviceMajorSoprano[offsetGPU], soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice, majorHighStream[offsetGPU]);
      CountNoteSection<<<1, NUM_THREADS, 0, majorHighStream[offsetGPU]>>>(deviceMajorSoprano[offsetGPU], sLength, deviceMajorHighNotes[offsetGPU], offsetGPU);
   
      cudaSetDevice(offsetGPU + MAJOR_LOW_DEVICE);
      cudaMemcpyAsync(deviceMajorBass[offsetGPU], bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice, majorLowStream[offsetGPU]);
      CountNoteSection<<<1, NUM_THREADS, 0, majorLowStream[offsetGPU]>>>(deviceMajorBass[offsetGPU], bLength, deviceMajorLowNotes[offsetGPU], offsetGPU);
    }
    else { //minor
      cudaSetDevice(offsetGPU + MINOR_HIGH_DEVICE);
      cudaMemcpyAsync(deviceMinorSoprano[offsetGPU], soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice, minorHighStream[offsetGPU]);
      CountNoteSection<<<1, NUM_THREADS, 0, minorHighStream[offsetGPU]>>>(deviceMinorSoprano[offsetGPU], sLength, deviceMinorHighNotes[offsetGPU], offsetGPU);

      cudaSetDevice(offsetGPU + MINOR_LOW_DEVICE);
      cudaMemcpyAsync(deviceMinorBass[offsetGPU], bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice, minorLowStream[offsetGPU]);
      CountNoteSection<<<1, NUM_THREADS, 0, minorLowStream[offsetGPU]>>>(deviceMinorBass[offsetGPU], bLength, deviceMinorLowNotes[offsetGPU], offsetGPU);
    }
  }

  //Call GPUS for chords
  if (mood == 0) { //major
    cudaSetDevice(MAJOR_CHORD_DEVICE);
    cudaMemcpyAsync(deviceMajorSoprano[NUM_GPU_PER_MATRIX], soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice, majorChordSStream);
    cudaMemcpyAsync(deviceMajorBass[NUM_GPU_PER_MATRIX], bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice, majorChordBStream);
    CountChordSection<<<1, NUM_THREADS, 0, majorChordSStream>>>(deviceMajorSoprano[NUM_GPU_PER_MATRIX], sLength, deviceMajorChords);
    CountChordSection<<<1, NUM_THREADS, 0, majorChordBStream>>>(deviceMajorBass[NUM_GPU_PER_MATRIX], bLength, deviceMajorChords);
  }
  else { //minor
    cudaSetDevice(MINOR_CHORD_DEVICE);
    cudaMemcpyAsync(deviceMinorSoprano[NUM_GPU_PER_MATRIX], soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice, minorChordSStream);
    cudaMemcpyAsync(deviceMinorBass[NUM_GPU_PER_MATRIX], bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice, minorChordBStream);
    CountChordSection<<<1, NUM_THREADS, 0, minorChordSStream>>>(deviceMinorSoprano[NUM_GPU_PER_MATRIX], sLength, deviceMinorChords);
    CountChordSection<<<1, NUM_THREADS, 0, minorChordBStream>>>(deviceMinorBass[NUM_GPU_PER_MATRIX], bLength, deviceMinorChords);
  }
}

/**
 * @brief Copies matrices in device memory to host
 * 
 */
void cudaToHost()
{
  //Copy all melodic matrices into host memory
  for (int offsetGPU = 0; offsetGPU < NUM_GPU_PER_MATRIX; offsetGPU++) { 
    int matrixIndex = NUM_NOTES * MATRIX_BLOCK_ROWS * offsetGPU; 

    cudaSetDevice(offsetGPU + MAJOR_HIGH_DEVICE);
    cudaMemcpyAsync(majorHighNotes + matrixIndex, deviceMajorHighNotes[offsetGPU], sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES, cudaMemcpyDeviceToHost, majorHighStream[offsetGPU]);

    cudaSetDevice(offsetGPU + MAJOR_LOW_DEVICE);
    cudaMemcpyAsync(majorLowNotes + matrixIndex, deviceMajorLowNotes[offsetGPU], sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES, cudaMemcpyDeviceToHost, majorLowStream[offsetGPU]);

    cudaSetDevice(offsetGPU + MINOR_HIGH_DEVICE);
    cudaMemcpyAsync(minorHighNotes + matrixIndex, deviceMinorHighNotes[offsetGPU], sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES, cudaMemcpyDeviceToHost, minorHighStream[offsetGPU]);

    cudaSetDevice(offsetGPU + MINOR_LOW_DEVICE);
    cudaMemcpyAsync(minorLowNotes + matrixIndex, deviceMinorLowNotes[offsetGPU], sizeof(int) * MATRIX_BLOCK_ROWS * NUM_NOTES, cudaMemcpyDeviceToHost, minorLowStream[offsetGPU]);

  }

  //Copy Chord matrices
  cudaSetDevice(MAJOR_CHORD_DEVICE);
  cudaMemcpy(majorChords, deviceMajorChords, sizeof(int) * NUM_CHORDS * NUM_CHORDS, cudaMemcpyDeviceToHost);

  cudaSetDevice(MINOR_CHORD_DEVICE);
  cudaMemcpy(minorChords, deviceMinorChords, sizeof(int) * NUM_CHORDS * NUM_CHORDS, cudaMemcpyDeviceToHost);
}
