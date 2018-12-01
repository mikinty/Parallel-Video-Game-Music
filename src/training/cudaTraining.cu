#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "training.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

//Transforms 2 (noteTone, noteLength) pairs to a matrix index
static inline int findNoteCell(int tone, int length, int tone1, int length1, int tone2, int length2)
{
	int col = tone * length - 1;
    if (tone1 > NUM_TONES){
        tone1 = tone1 % 12 - NUM_TONES; //get bottom chord note
    }
    if (tone2 > NUM_TONES){
        tone2 = tone2 % 12 - NUM_TONES; //get bottom chord note
    }
    int row = tone1 * length1 * tone2 * length2 - 1;
    return row * NUM_NOTES + col;
}

//Gets flattened matrix index
static inline int findChordCell(int tone, int tone1){
	if (tone1 > NUM_TONES){
        tone1 = tone1 - NUM_TONES - 1; //shift chord down
    }
    return tone1 * NUM_CHORDS + tone1;
}

__global__ void CountSection(sound_t* deviceS,int sLength,sound_t* deviceB,int bLength,double* deviceHigh,double* deviceLow,double* deviceChord)
{
    sound_t* part;
    int start;
    int end;
    __shared__ double melodyM[NUM_NOTES * NUM_NOTES * NUM_NOTES];

    //initialize shared mem
    start = threadIdx.x * (NUM_NOTES * NUM_NOTES * NUM_NOTES + NUM_THREADS - 1)/NUM_THREADS;
    end = start + (NUM_NOTES * NUM_NOTES * NUM_NOTES + NUM_THREADS - 1)/NUM_THREADS;
    for (int mIndex = start; mIndex < end; mIndex++){
        if (mIndex > NUM_NOTES * NUM_NOTES * NUM_NOTES)
            break;
        melodyM[mIndex] = 0;
    }

    if (blockIdx.x == 0){
        part = deviceS;
        start = threadIdx.x * (sLength / NUM_THREADS) + 2;
        if (threadIdx.x == NUM_THREADS - 1){
            end = sLength;
        }
        else{
            end = start + sLength/NUM_THREADS;
        }
    }
    else{
        part = deviceB;
        start = threadIdx.x * (bLength / NUM_THREADS) + 2;
        if (threadIdx.x == NUM_THREADS - 1){
            end = bLength;
        }
        else{
            end = start + bLength/NUM_THREADS;
        }
    }

    int prevTone1 = part[start-2].tone;
    int prevLength1 = part[start-2].duration;
    int curTone = part[start-1].tone;
    int curLength = part[start-1].duration;

    __syncthreads();

    for (int noteIndex = start; noteIndex < end; noteIndex++){    	
    	int prevTone2 = prevTone1;
    	int prevLength2 = prevLength1;
    	prevTone1 = curTone;
        prevLength1 = curLength;
        curTone = part[noteIndex].tone;
        curLength = part[noteIndex].duration;

        int cell;
        if (curTone > NUM_TONES) { //insert into chord matrix
            cell = findChordCell(curTone, prevTone1);
            atomic_add(deviceChord[cell], 1);
        }
        else (blockIdx.x == 0) { //insert into soprano note matrix
            cell = findNoteCell(curTone, curLength, prevTone1, prevLength1, prevTone2, prevLength2);
            atomic_add(melodyM[cell], 1);
        }
    }

    __syncthreads();

    start = threadIdx.x * (NUM_NOTES * NUM_NOTES * NUM_NOTES + NUM_THREADS - 1)/NUM_THREADS;
    end = start + (NUM_NOTES * NUM_NOTES * NUM_NOTES + NUM_THREADS - 1)/NUM_THREADS;
    for (int mIndex = start; mIndex < end; mIndex++){
        if (mIndex > NUM_NOTES * NUM_NOTES * NUM_NOTES)
            break;
        if (blockIdx.x == 0)
            deviceHigh[mIndex] = melodyM[mIndex];
        else
            deviceLow[mIndex] = melodyM[mIndex];
    }
    
}

void countTransitionsCuda(sound_t* soprano, int sLength, sound_t* bass, int bLength, double* deviceHigh, double* deviceLow, double* deviceChord){
    cudaMalloc((void**)&deviceHigh, sizeof(double) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
    cudaMalloc((void**)&deviceLow, sizeof(double) * NUM_NOTES * NUM_NOTES * NUM_NOTES);
    cudaMalloc((void**)&deviceChord, sizeof(double) * NUM_CHORDS * NUM_CHORDS);

    cudaMemset(deviceChord, 0, sizeof(double) * NUM_CHORDS * NUM_CHORDS);

    sound_t* deviceS;
    sound_t* deviceB;
    cudaMalloc((void **)&deviceS, sizeof(sound_t) * sLength);
    cudaMemcpy(deviceS, soprano, sizeof(sound_t) * sLength, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&deviceB, sizeof(sound_t) * bLength);
    cudaMemcpy(deviceB, bass, sizeof(sound_t) * bLength, cudaMemcpyHostToDevice);

    CountSection<<<NUM_THREADS, 2>>>(deviceS, sLength, deviceB, bLength, deviceHigh, deviceLow, deviceChord);

    cudaThreadSynchronize();

    cudaFree(deviceS);
    cudaFree(deviceB);
}



template<typename T>
struct MulC: public thrust::unary_function<T, T>
{
    T C;
    __host__ __device__ MulC(T c) : C(c) { }
    __host__ __device__ T operator()(T x) { return x * C; }
};

__global__ void normalizeRow(double* deviceHigh,double* deviceLow,double* deviceChord){
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


void normalizeCuda(double* deviceHigh,double* deviceLow,double* deviceChord,double* highNotes,double* lowNotes,double* chords){
 
   /* thrust::device_vector<float> d_row_sums_3(Nrows, 0);
    thrust::device_vector<float> d_temp(Nrows * Ncols);
    thrust::inclusive_scan_by_key(
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)),
                thrust::make_transform_iterator(thrust::make_counting_iterator(0), linear_index_to_row_index<int>(Ncols)) + (Nrows*Ncols),
                d_matrix.begin(),
                d_temp.begin());
    thrust::copy(
                thrust::make_permutation_iterator(
                        d_temp.begin() + Ncols - 1,
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), MulC<int>(Ncols))),
    thrust::make_permutation_iterator(
                        d_temp.begin() + Ncols - 1,
                        thrust::make_transform_iterator(thrust::make_counting_iterator(0), MulC<int>(Ncols))) + Nrows,
                d_row_sums_3.begin()); */

    normalizeRow<<<NUM_THREADS, 3>>>(deviceHigh, deviceLow, deviceChord);

    cudaMemcpy(highNotes, deviceHigh, sizeof(double) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(lowNotes, deviceLow, sizeof(double) * NUM_NOTES * NUM_NOTES * NUM_NOTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(chords, deviceChord, sizeof(double) * NUM_CHORDS * NUM_CHORDS, cudaMemcpyDeviceToHost);

    cudaFree(deviceHigh);
    cudaFree(deviceLow);
    cudaFree(deviceChord);
}