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
  int col = curTone * curDur - 1;

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
	
  if (prevTone >= CHORD_OFFSET){
        prevTone = prevTone - CHORD_OFFSET; //shift chord down
  }
  
  return prevTone * NUM_CHORDS + (curTone - CHORD_OFFSET);
}

/**
 * @brief Depending on the threadIndex, counts note transitions in a section of the given notes array and adds
 * to existing counts on the given matrices
 * 
 * @param deviceS soprano notes array to count
 * @param sLength soprano notes array length
 * @param deviceB bass notes array to count
 * @param bLength bass notes array length
 * @param deviceHigh matrix for high melodic line
 * @param deviceLow matrix for low melodic line
 * @param deviceHigh matrix for chords
 */
__global__ 
void CountSection(sound_t* deviceS, int sLength, sound_t* deviceB, int bLength, float* deviceHigh, float* deviceLow, float* deviceChord)
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
     int cell; //matrix index to insert into
    if (curTone >= CHORD_OFFSET) { //insert into chord matrix
      cell = findChordCell(curTone, prevTone1);
      atomicAdd(deviceChord + cell, 1);
    }
    else { //insert into melody note matrix
      cell = findNoteCell(curTone, curDur, prevTone1, prevDur1, prevTone2, prevDur2);
      if (blockIdx.x == 0) { //insert into soprano matrix
        atomicAdd(deviceHigh + cell, 1);
      }
      else { //insert into bass matrix
        atomicAdd(deviceLow + cell, 1);
      }
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
  cudaMalloc((void**)&deviceMajorHighNotes, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMalloc((void**)&deviceMajorLowNotes, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMalloc((void**)&deviceMajorChords, sizeof(float) * NUM_CHORDS * NUM_CHORDS); 
  cudaMalloc((void**)&deviceMinorHighNotes, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMalloc((void**)&deviceMinorLowNotes, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMalloc((void**)&deviceMinorChords, sizeof(float) * NUM_CHORDS * NUM_CHORDS); 

  //Initialize memory for all matrices
  cudaMemset(deviceMajorHighNotes, 0, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMemset(deviceMajorLowNotes, 0, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMemset(deviceMajorChords, 0, sizeof(float) * NUM_CHORDS * NUM_CHORDS);
  cudaMemset(deviceMinorHighNotes, 0, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMemset(deviceMinorLowNotes, 0, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
  cudaMemset(deviceMinorChords, 0, sizeof(float) * NUM_CHORDS * NUM_CHORDS);

  __syncthreads();
}

/**
 * @brief Frees device memory for all matrices
 * 
 */
void freeCuda()
{
  //Allocate device memory for all matrices
  cudaFree(deviceMajorHighNotes);
  cudaFree(deviceMajorLowNotes);
  cudaFree(deviceMajorChords); 
  cudaFree(deviceMinorHighNotes);
  cudaFree(deviceMinorLowNotes);
  cudaFree(deviceMinorChords); 
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
  cudaMalloc((void **)&deviceS, sizeof(sound_t) * sLength);
  cudaMemcpy(deviceS, soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&deviceB, sizeof(sound_t) * bLength);
  cudaMemcpy(deviceB, bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice);

  //Determine if we should use major or minor matrices, and call GPUS to count note transitions
  if (mood.compare("maj") == 0) { //major
    CountSection<<<NUM_THREADS, 2>>>(deviceS, sLength, deviceB, bLength, deviceMajorHighNotes, deviceMajorLowNotes, deviceMajorChords);
  else { //minor
    CountSection<<<NUM_THREADS, 2>>>(deviceS, sLength, deviceB, bLength, deviceMinorHighNotes, deviceMinorLowNotes, deviceMinorChords);
  }
  cudaThreadSynchronize();

  //Free device memory file buffers
  cudaFree(deviceS);
  cudaFree(deviceB);
}

/**
 * @brief Non-thrust-using code to normalize a matrix, NOT USED FOR NOW
 *
 */
__global__ void normalizeRow(float* deviceHigh,float* deviceLow,float* deviceChord){
    int start;
    int end;
    if(blockIdx.x == 0){
        start = threadIdx.x * (NUM_NOTES * NUM_NOTES + NUM_THREADS - 1)/NUM_THREADS;
        end = start + (NUM_NOTES * NUM_NOTES + NUM_THREADS - 1)/NUM_THREADS;
        if (end > NUM_NOTES * NUM_NOTES){
            end = NUM_NOTES * NUM_NOTES;
        }
        for (int i = start; i < end; i++){
            int sum = 0;
            for(int j = 0; j < NUM_NOTES; j++){
                sum = sum + deviceHigh[i * NUM_NOTES + j];
            }
            for (int j = 0; j < NUM_NOTES; j++){
                deviceHigh[i * NUM_NOTES + j] = deviceHigh[i * NUM_NOTES + j] / sum;
            }
        }
    }
    else if (blockIdx.x == 1){
        start = threadIdx.x * (NUM_NOTES * NUM_NOTES + NUM_THREADS - 1)/NUM_THREADS;
        end = start + (NUM_NOTES * NUM_NOTES + NUM_THREADS - 1)/NUM_THREADS;
        if (end > NUM_NOTES * NUM_NOTES){
            end = NUM_NOTES * NUM_NOTES;
        }
        for (int i = start; i < end; i++){
            int sum = 0;
            for(int j = 0; j < NUM_NOTES; j++){
                sum = sum + deviceLow[i * NUM_NOTES + j];
            }
            for (int j = 0; j < NUM_NOTES; j++){
                deviceLow[i * NUM_NOTES + j] = deviceLow[i * NUM_NOTES + j] / sum;
            }
        }
    }
    else{
        start = threadIdx.x * (NUM_CHORDS + NUM_THREADS - 1)/NUM_THREADS;
        end = start + (NUM_NOTES * NUM_CHORDS + NUM_THREADS - 1)/NUM_THREADS;
        if (end > NUM_CHORDS){
            end = NUM_CHORDS;
        }
        for (int i = start; i < end; i++){
            int sum = 0;
            for(int j = 0; j < NUM_CHORDS; j++){
                sum = sum + deviceChord[i * NUM_CHORDS + j];
            }
            for (int j = 0; j < NUM_CHORDS; j++){
                deviceChord[i * NUM_CHORDS + j] = deviceChord[i * NUM_CHORDS + j] / sum;
            }
        }
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
void normalizeCuda(){
 
  //setup number of rows, cols for normalizing melodic matrices
  int Nrows = NUM_NOTES * NUM_NOTES;
  int Ncols = NUM_NOTES;

  thrust::device_ptr<float> thrust_high(deviceMajorHighNotes);
  // --- Allocate space for row sums and indices
  thrust::device_vector<float> d_row_sums(Nrows);
  thrust::device_vector<int> d_row_indices(Nrows);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_high,
                        d_row_indices.begin(),
                        d_row_sums.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_high,
                thrust::make_discard_iterator(),
                d_row_sums.begin());

  thrust::device_ptr<float> thrust_low(deviceMajorLowNotes);

  // --- Compute row sums by summing values with equal row indices
   thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_low,
                        d_row_indices.begin(),
                        d_row_sums.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

   thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_low,
                thrust::make_discard_iterator(),
                d_row_sums.begin());

  thrust::device_ptr<float> thrust_low(deviceMinorHighNotes);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_low,
                        d_row_indices.begin(),
                        d_row_sums.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_low,
                thrust::make_discard_iterator(),
                d_row_sums.begin());

  thrust::device_ptr<float> thrust_low(deviceMinorLowNotes);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_low,
                        d_row_indices.begin(),
                        d_row_sums.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_low,
                thrust::make_discard_iterator(),
                d_row_sums.begin());


  //Set up dimensions for chord matrices
  Nrows = NUM_CHORDS;
  Ncols = NUM_CHORDS;

  thrust::device_ptr<float> thrust_chord(deviceMajorChords);
  thrust::device_vector<float> d_row_sums_c(Nrows);
  thrust::device_vector<int> d_row_indices_c(Nrows);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_chord,
                        d_row_indices_c.begin(),
                        d_row_sums_c.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_chord,
                thrust::make_discard_iterator(),
                d_row_sums_c.begin());

  thrust::device_ptr<float> thrust_chord(deviceMinorChords);

  // --- Compute row sums by summing values with equal row indices
  thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)),
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                        thrust_chord,
                        d_row_indices_c.begin(),
                        d_row_sums_c.begin(),
                        thrust::equal_to<int>(),
                        thrust::plus<float>());

  thrust::reduce_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                thrust_chord,
                thrust::make_discard_iterator(),
                d_row_sums_c.begin());

  //Copy all matrices into host memory
  cudaMemcpy(majorHighNotes, deviceMajorHighNotes, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(majorLowNotes, deviceMajorLowNotes, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(majorChords, deviceMajorChords, sizeof(float) * NUM_CHORDS * NUM_CHORDS, cudaMemcpyDeviceToHost);
  cudaMemcpy(minorHighNotes, deviceMinorHighNotes, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(minorHighNotes, deviceMinorHighNotes, sizeof(float) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(minorChords, deviceMinorChords, sizeof(float) * NUM_CHORDS * NUM_CHORDS, cudaMemcpyDeviceToHost);

  __syncthreads();
}
