#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "training.h"

#include <thrust/iterator/discard_iterator.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

// Define global variables
int* deviceMajorHighNotes;
int* deviceMajorLowNotes;
int* deviceMajorChords;
int* deviceMinorHighNotes;
int* deviceMinorLowNotes;
int* deviceMinorChords;

/**
 * @brief Transforms 2 (noteTone, noteLength) pairs to a matrix index for a melody matrix
 * 
 * @param curTone tone of the current note
 * @param curDur duration of the current note
 * @param prevTone1 tone of the previous note
 * @param prevDur1 duration of the previous note
 * @param prevTone2 tone of the note before previous note
 * @param prevDur2 duration of the note before previous note
 */
__device__ 
inline int findNoteCell(int curTone, int curDur, int prevTone1, int prevDur1, int prevTone2, int prevDur2)
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

  return row * NUM_NOTES + col;
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
 * @brief Depending on the threadIndex, counts note transitions in a section of the given notes array and adds
 * to existing counts on the given matrices
 * 
 * @param part notes array to count
 * @param length notes array length
 * @param deviceMatrix matrix for inserting into
 */
__global__ 
void CountSection(sound_t* part, int length, int* deviceMatrix)
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
    
    if (curTone < NUM_TONES) { //if not a chord, insert
      int cell = findNoteCell(curTone, curDur, prevTone1, prevDur1, prevTone2, prevDur2);
      atomicAdd(&deviceMatrix[cell], 1);   
    }
  } 
}

/**
 * @brief Depending on the threadIndex, counts chord transitions in a section of the given notes array and adds
 * to existing counts on the given matrices
 * 
 * @param deviceS soprano notes array to count
 * @param sLength soprano notes array length
 * @param deviceB bass notes array to count
 * @param bLength bass notes array length
 * @param deviceMatrix matrix for chords
 */
__global__ 
void CountChordSection(sound_t* deviceS, int sLength, sound_t* deviceB, int bLength, int* deviceMatrix)
{
  //Bounds for section to read based on threadId
  int start;
  int end;
  sound_t* part; //either soprano or bass line to read

  //Determine proper start and end bounds
  if (blockIdx.x == 0) { //Thread is in Soprano Line Block
      part = deviceS;
      start = threadIdx.x * (sLength / NUM_THREADS) + 2;
      if (threadIdx.x == NUM_THREADS - 1){ //If we are the last thread, make sure to grab any extra bits
          end = sLength;
      }
      else{
          end = start + sLength/NUM_THREADS;
      }
  }
  else{ //Bass line block
      part = deviceB;
      start = threadIdx.x * (bLength / NUM_THREADS) + 2;
      if (threadIdx.x == NUM_THREADS - 1){ //If we are the last thread, make sure to grab any extra bits
          end = bLength;
      }
      else{
          end = start + bLength/NUM_THREADS;
      }
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
 * @brief Allocates device memory for the matrices
 * 
 */
void initCuda()
{
  //Allocate device memory for all matrices
  cudaSetDevice(MAJOR_HIGH_DEVICE);
  cudaMalloc((void**)&deviceMajorHighNotes, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMemset(deviceMajorHighNotes, 0, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES);

  cudaSetDevice(MAJOR_LOW_DEVICE);
  cudaMalloc((void**)&deviceMajorLowNotes, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMemset(deviceMajorLowNotes, 0, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES);

  cudaSetDevice(MAJOR_CHORD_DEVICE);
  cudaMalloc((void**)&deviceMajorChords, sizeof(int) * NUM_CHORDS * NUM_CHORDS); 
  cudaMemset(deviceMajorChords, 0, sizeof(int) * NUM_CHORDS * NUM_CHORDS);

  cudaSetDevice(MINOR_HIGH_DEVICE);
  cudaMalloc((void**)&deviceMinorHighNotes, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMemset(deviceMinorHighNotes, 0, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES);

  cudaSetDevice(MINOR_LOW_DEVICE);
  cudaMalloc((void**)&deviceMinorLowNotes, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMemset(deviceMinorLowNotes, 0, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES);

  cudaSetDevice(MINOR_CHORD_DEVICE);
  cudaMalloc((void**)&deviceMinorChords, sizeof(int) * NUM_CHORDS * NUM_CHORDS); 
  cudaMemset(deviceMinorChords, 0, sizeof(int) * NUM_CHORDS * NUM_CHORDS);
}

/**
 * @brief Frees device memory for all matrices
 * 
 */
void freeCuda()
{
  //Frees all device matrices
  cudaSetDevice(MAJOR_HIGH_DEVICE);
  cudaFree(deviceMajorHighNotes);
  cudaSetDevice(MAJOR_LOW_DEVICE);
  cudaFree(deviceMajorLowNotes);
  cudaSetDevice(MAJOR_CHORD_DEVICE);
  cudaFree(deviceMajorChords); 
  cudaSetDevice(MINOR_HIGH_DEVICE);
  cudaFree(deviceMinorHighNotes);
  cudaSetDevice(MINOR_LOW_DEVICE);
  cudaFree(deviceMinorLowNotes);
  cudaSetDevice(MINOR_CHORD_DEVICE);
  cudaFree(deviceMinorChords); 
}

/**
 * @brief Synchs all devices/GPUS used
 */
void synchAllCuda()
{
  cudaSetDevice(MAJOR_HIGH_DEVICE);
  cudaDeviceSynchronize();
  cudaSetDevice(MAJOR_LOW_DEVICE);
  cudaDeviceSynchronize();
  cudaSetDevice(MAJOR_CHORD_DEVICE);
  cudaDeviceSynchronize();
  cudaSetDevice(MINOR_HIGH_DEVICE);
  cudaDeviceSynchronize();
  cudaSetDevice(MINOR_LOW_DEVICE);
  cudaDeviceSynchronize();
  cudaSetDevice(MINOR_CHORD_DEVICE);
  cudaDeviceSynchronize();
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
void countTransitionsCuda(sound_t* soprano, int sLength, sound_t* bass, int bLength, std::string mood){

  //Allocate and copy device memory for soprano, bass note lines
  sound_t* deviceS;
  sound_t* deviceB;
  sound_t* deviceChordS;
  sound_t* deviceChordB;

  //Determine if we should use major or minor matrices, and call GPUS to count note transitions
  if (mood.compare("major") == 0) { //major 
    cudaSetDevice(MAJOR_HIGH_DEVICE);
    cudaMalloc((void **)&deviceS, sizeof(sound_t) * sLength);
    cudaMemcpyAsync(deviceS, soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice);
    CountSection<<<1, NUM_THREADS>>>(deviceS, sLength, deviceMajorHighNotes);
   
    cudaSetDevice(MAJOR_LOW_DEVICE);
    cudaMalloc((void **)&deviceB, sizeof(sound_t) * bLength);
    cudaMemcpyAsync(deviceB, bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice);
    CountSection<<<1, NUM_THREADS>>>(deviceB, bLength, deviceMajorLowNotes);

    cudaSetDevice(MAJOR_CHORD_DEVICE);
    cudaMalloc((void **)&deviceChordS, sizeof(sound_t) * sLength);
    cudaMemcpyAsync(deviceChordS, soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&deviceChordB, sizeof(sound_t) * bLength);
    cudaMemcpyAsync(deviceChordB, bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice);
    CountChordSection<<<2, NUM_THREADS>>>(deviceChordS, sLength, deviceChordB, bLength, deviceMajorChords);

    //Free used memory
    cudaFree(deviceChordS);
    cudaFree(deviceChordB);
    cudaSetDevice(MAJOR_HIGH_DEVICE);
    cudaFree(deviceS);
    cudaSetDevice(MAJOR_LOW_DEVICE);
    cudaFree(deviceB);

  }
  else { //minor
     cudaSetDevice(MINOR_HIGH_DEVICE);
     cudaMalloc((void **)&deviceS, sizeof(sound_t) * sLength);
     cudaMemcpyAsync(deviceS, soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice);
     CountSection<<<1, NUM_THREADS>>>(deviceS, sLength, deviceMinorHighNotes);
     
     cudaSetDevice(MINOR_LOW_DEVICE);
     cudaMalloc((void **)&deviceB, sizeof(sound_t) * bLength);
     cudaMemcpyAsync(deviceB, bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice);
     CountSection<<<1, NUM_THREADS>>>(deviceB, bLength, deviceMinorLowNotes);
     
     cudaSetDevice(MINOR_CHORD_DEVICE);
     cudaMalloc((void **)&deviceChordS, sizeof(sound_t) * sLength);
     cudaMemcpyAsync(deviceChordS, soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice);
     cudaMalloc((void **)&deviceChordB, sizeof(sound_t) * bLength);
     cudaMemcpyAsync(deviceChordB, bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice);
     CountChordSection<<<2, NUM_THREADS>>>(deviceChordS, sLength, deviceChordB, bLength, deviceMinorChords);

     //Free used memory
     cudaFree(deviceChordS);
     cudaFree(deviceChordB);
     cudaSetDevice(MINOR_HIGH_DEVICE);
     cudaFree(deviceS);
     cudaSetDevice(MINOR_LOW_DEVICE);
     cudaFree(deviceB);
  }
}

//Template for matrix normalization in thrust
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {

    T Ncols; // --- Number of columns

    __host__ __device__ linear_index_to_row_index(T Ncols) : Ncols(Ncols) {}

    __host__ __device__ T operator()(T i) { return i / Ncols; }
};

/**
 * @brief Normalizes all 6 device matrices, and copies them into host
 */
/*void normalizeCuda(){

  //setup number of rows, cols for normalizing melodic matrices
  int Nrows = NUM_NOTES * NUM_NOTES;
  int Ncols = NUM_NOTES;

  cudaSetDevice(MAJOR_HIGH_DEVICE);
  
  thrust::device_ptr<float> thrust_majorHigh(deviceMajorHighNotes);
  // --- Allocate space for row sums and indices
  thrust::device_vector<float> d_row_sums(Nrows);
  thrust::device_vector<int> d_row_indices(Nrows);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_majorHigh,
                        d_row_indices.begin(),
                        d_row_sums.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_majorHigh,
                thrust::make_discard_iterator(),
                d_row_sums.begin());

  cudaDeviceSynchronize();

  cudaSetDevice(MAJOR_LOW_DEVICE);

  thrust::device_ptr<float> thrust_majorLow(deviceMajorLowNotes);
  // --- Allocate space for row sums and indices
  thrust::device_vector<float> drowsums(100);
  thrust::device_vector<int> d_row_indices(5);

  // --- Compute row sums by summing values with equal row indices
   thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_majorLow,
                        d_row_indices.begin(),
                        d_row_sums.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

   thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_majorLow,
                thrust::make_discard_iterator(),
                d_row_sums.begin());

  cudaSetDevice(MINOR_HIGH_DEVICE);

  thrust::device_ptr<float> thrust_minorHigh(deviceMinorHighNotes);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_minorHigh,
                        d_row_indices.begin(),
                        d_row_sums.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_minorHigh,
                thrust::make_discard_iterator(),
                d_row_sums.begin());

  thrust::device_ptr<float> thrust_minorLow(deviceMinorLowNotes);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_minorLow,
                        d_row_indices.begin(),
                        d_row_sums.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_minorLow,
                thrust::make_discard_iterator(),
                d_row_sums.begin());


  //Set up dimensions for chord matrices
  Nrows = NUM_CHORDS;
  Ncols = NUM_CHORDS;

  thrust::device_ptr<float> thrust_majorChord(deviceMajorChords);
  thrust::device_vector<float> d_row_sums_c(Nrows);
  thrust::device_vector<int> d_row_indices_c(Nrows);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_majorChord,
                        d_row_indices_c.begin(),
                        d_row_sums_c.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_majorChord,
                thrust::make_discard_iterator(),
                d_row_sums_c.begin());

  thrust::device_ptr<float> thrust_minorChord(deviceMinorChords);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_minorChord,
                        d_row_indices_c.begin(),
                        d_row_sums_c.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_minorChord,
                thrust::make_discard_iterator(),
                d_row_sums_c.begin());

}*/

void cudaToHost()
{
  //Copy all matrices into host memory
  cudaSetDevice(MAJOR_HIGH_DEVICE);
  cudaMemcpy(majorHighNotes, deviceMajorHighNotes, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
  cudaSetDevice(MAJOR_LOW_DEVICE);
  cudaMemcpy(majorLowNotes, deviceMajorLowNotes, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
  cudaSetDevice(MAJOR_CHORD_DEVICE);
  cudaMemcpy(majorChords, deviceMajorChords, sizeof(int) * NUM_CHORDS * NUM_CHORDS, cudaMemcpyDeviceToHost);
  cudaSetDevice(MINOR_HIGH_DEVICE);
  cudaMemcpy(minorHighNotes, deviceMinorHighNotes, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
  cudaSetDevice(MINOR_LOW_DEVICE);
  cudaMemcpy(minorHighNotes, deviceMinorHighNotes, sizeof(int) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
  cudaSetDevice(MINOR_CHORD_DEVICE);
  cudaMemcpy(minorChords, deviceMinorChords, sizeof(int) * NUM_CHORDS * NUM_CHORDS, cudaMemcpyDeviceToHost);
}
