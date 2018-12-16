#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "training.h"
#include <iostream>
#include <fstream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

//Define global variables
int* majorHighNotes;
int* majorLowNotes;
int* majorChords;
int* minorHighNotes;
int* minorLowNotes;
int* minorChords;

sound_t* majorSoprano;
sound_t* majorBass;
sound_t* minorSoprano;
sound_t* minorBass;

inline int findNoteCell(int curTone, int curDur, int prevTone1, int prevDur1, int prevTone2, int prevDur2, int part)
{
  // current note 
  int col = curTone * NUM_DUR + curDur;

  //If previous tones are chords, get top note and find closest
  if (prevTone1 >= CHORD_OFFSET){
      prevTone1 = (prevTone1 - CHORD_OFFSET) / 144; //get top chord note
      if (curTone == NUM_TONES - 1){ //is rest
        prevTone1 = prevTone1 + 12 * (2 * part);
      }
      else {
        prevTone1 = curTone - (curTone % 12) + prevTone1; //find closest prevTone1 note
      }
  }
  if (prevTone2 >= CHORD_OFFSET){
      prevTone2 = (prevTone2 - CHORD_OFFSET) / 144; //get top chord note
      if (curTone == NUM_TONES - 1) { //is rest
        prevTone2 = prevTone2 + 12 * (2 * part);
      }
      else { //is note 
        prevTone2 = curTone - (curTone % 12) + prevTone2; //find closest note
      }
  }

  int row = ((prevTone1 * NUM_DUR) + prevDur1) * NUM_NOTES
    + ((prevTone2 * NUM_DUR) + prevDur2); 

  return (row) * NUM_NOTES + col;
}

/**
 * @brief Transforms 2 tones into a matrix index for the chord matrix
 * 
 * @param curTone tone of the current note
 * @param prevTone tone of the previous note
 */
inline int findChordCell(int curTone, int prevTone){
  
  if (prevTone >= CHORD_OFFSET) { //is a chord
    prevTone = prevTone - CHORD_OFFSET; //shift chord down
  }
  else if (prevTone == NUM_TONES - 1) { //is a rest
    return -1;
  }
  else { //is a note
    prevTone = prevTone % 12 + 12 * (prevTone % 12) + 144 * (prevTone % 12);
  }

  return prevTone * NUM_CHORDS + (curTone - CHORD_OFFSET);
}

void countTransitions(sound_t* sop, int sLen, sound_t* bass, int bLen, int mood){
  int curTone = -1;
  int curDur = -1;
  int prevTone1 = -1;
  int prevDur1 = -1;
  int prevTone2 = -1;
  int prevDur2 = -1;
  if (mood == 0){
    for (int i = 0; i < sLen; i++){
      prevTone2 = prevTone1;
      prevDur2 = prevDur1;
      prevTone1 = curTone;
      prevDur1 = curDur;
      curTone = sop[i].tone;
      curDur = sop[i].duration;
      if (curTone < NUM_TONES && prevTone2 != -1) { //if not a chord, check device and insert
        int cell = findNoteCell(curTone, curDur, prevTone1, prevDur1, prevTone2, prevDur2,2);
        majorHighNotes[cell]++;   
      }
      else if (curTone >= CHORD_OFFSET && prevTone1 != -1) {
        int cell = findChordCell(curTone, prevTone1);
        if (cell != -1)
          majorChords[cell]++;
      }
    }
    prevTone2 = -1;
    prevTone1 = -1;
    for (int i = 0; i < bLen; i++){
      prevTone2 = prevTone1;
      prevDur2 = prevDur1;
      prevTone1 = curTone;
      prevDur1 = curDur;
      curTone = bass[i].tone;
      curDur = bass[i].duration;
      if (curTone < NUM_TONES && prevTone2 != -1) { //if not a chord, check device and insert
        int cell = findNoteCell(curTone, curDur, prevTone1, prevDur1, prevTone2, prevDur2,2);
        majorLowNotes[cell]++;
      }
      else if (curTone >= CHORD_OFFSET && prevTone1 != -1) {
        int cell = findChordCell(curTone, prevTone1);
        if (cell != -1)
          majorChords[cell]++;
      }
    }
  }
  else {
    for (int i = 0; i < sLen; i++){
      prevTone2 = prevTone1;
      prevDur2 = prevDur1;
      prevTone1 = curTone;
      prevDur1 = curDur;
      curTone = sop[i].tone;
      curDur = sop[i].duration;
      if (curTone < NUM_TONES && prevTone2 != -1) { //if not a chord, check device and insert
        int cell = findNoteCell(curTone, curDur, prevTone1, prevDur1, prevTone2, prevDur2,2);
        minorHighNotes[cell]++;
      }
      else if (curTone >= CHORD_OFFSET && prevTone1 != -1) {
        int cell = findChordCell(curTone, prevTone1);
        if (cell != -1)
          minorChords[cell]++;
        }
      }
      prevTone2 = -1;
      prevTone1 = -1;
      for (int i = 0; i < bLen; i++){
        prevTone2 = prevTone1;
        prevDur2 = prevDur1;
        prevTone1 = curTone;
        prevDur1 = curDur;
        curTone = bass[i].tone;
        curDur = bass[i].duration;
        if (curTone < NUM_TONES && prevTone2 != -1) { //if not a chord, check device and insert
          int cell = findNoteCell(curTone, curDur, prevTone1, prevDur1, prevTone2, prevDur2,2);
          minorLowNotes[cell]++;
        }
        else if (curTone >= CHORD_OFFSET && prevTone1 != -1) {
          int cell = findChordCell(curTone, prevTone1);
          if (cell != -1)
            minorChords[cell]++;
          }
       }
  }
}
/**
 * @brief Outputs matrices to files
 */
void outputMatrices() {
  
  //For each matrix, copy matrix into file
	std::ofstream outFile;
	outFile.open("majorHighMatrixNew.txt");
	for (int i = 0; i < NUM_NOTES * NUM_NOTES; i ++){
		for (int j = 0; j < NUM_NOTES; j++){
			outFile << majorHighNotes[i * NUM_NOTES + j] << " ";
		}
		outFile << "\n";
	}
	outFile.close();

  printf("Printing 1/6 complete\n");

	outFile.open("majorLowMatrixNew.txt");
	for (int i = 0; i < NUM_NOTES * NUM_NOTES; i ++){
		for (int j = 0; j < NUM_NOTES; j++){
      outFile << majorLowNotes[i * NUM_NOTES + j] << " ";
		}
		outFile << "\n";
	}
	outFile.close();

  printf("Printing 2/6 complete\n");

	outFile.open("majorChordMatrixNew.txt");
	for (int i = 0; i < NUM_CHORDS; i ++){
		for (int j = 0; j < NUM_CHORDS; j++){
      outFile << majorChords[i * NUM_CHORDS + j] << " ";
		}
		outFile << "\n";
	}
	outFile.close();

  printf("Printing 3/6 complete\n");

  outFile.open("minorHighMatrixNew.txt");
  for (int i = 0; i < NUM_NOTES * NUM_NOTES; i ++){
    for (int j = 0; j < NUM_NOTES; j++){
      outFile << minorHighNotes[i * NUM_NOTES + j] << " ";
    }
    outFile << "\n";
  }
  outFile.close();

  printf("Printing 4/6 complete\n");

  outFile.open("minorLowMatrixNew.txt");
  for (int i = 0; i < NUM_NOTES * NUM_NOTES; i ++){
    for (int j = 0; j < NUM_NOTES; j++){
      outFile << minorLowNotes[i * NUM_NOTES + j] << " ";
    }
    outFile << "\n";
  }
  outFile.close();

  printf("Printing 5/6 complete\n");

  outFile.open("minorChordMatrixNew.txt");
  for (int i = 0; i < NUM_CHORDS; i ++){
    for (int j = 0; j < NUM_CHORDS; j++){
      outFile << minorChords[i * NUM_CHORDS + j] << " ";
    }
    outFile << "\n";
  }
  outFile.close();

  printf("Printing 6/6 complete\n");

  //Remove all old files, and replace with the new matrix files
  remove("majorHighMatrix.txt");
  remove("majorHowMatrix.txt");
  remove("majorChordMatrix.txt");
  remove("minorHighMatrix.txt");
  remove("minorHowMatrix.txt");
  remove("minorChordMatrix.txt");
  std::rename("majorHighMatrixNew.txt", "majorHighMatrix.txt");
  std::rename("majorLowMatrixNew.txt", "majorLowMatrix.txt");
  std::rename("majorChordMatrixNew.txt", "majorChordMatrix.txt");
  std::rename("minorHighMatrixNew.txt", "minorHighMatrix.txt");
  std::rename("minorLowMatrixNew.txt", "minorLowMatrix.txt");
  std::rename("minorChordMatrixNew.txt", "minorChordMatrix.txt");

  printf("File renaming complete\n");
}

/**
 * @brief uses cudaHostAlloc to allocated pinned memory for global structures
 */
void allocHost() {
  //Allocate memory for all final host matrices
  /*
  cudaHostAlloc(&majorHighNotes, sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES), cudaHostAllocPortable);
  cudaHostAlloc(&majorLowNotes, sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES), cudaHostAllocPortable);
  cudaHostAlloc(&minorHighNotes, sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES), cudaHostAllocPortable);
  cudaHostAlloc(&minorLowNotes, sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES), cudaHostAllocPortable);
  */
  majorHighNotes = (int *) malloc(sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES));
  majorLowNotes = (int *) malloc(sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES));
  minorHighNotes = (int *) malloc(sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES));
  minorLowNotes = (int *) malloc(sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES));
   
  //Chord Matrices
  majorChords =  (int *) malloc(sizeof(int) * (NUM_CHORDS * NUM_CHORDS));
  minorChords =  (int *) malloc(sizeof(int) * (NUM_CHORDS * NUM_CHORDS));
 
  //Arrays of notes read from files, initialized to ARRAY_LENGTH (1000) in length
  cudaHostAlloc(&majorSoprano, sizeof(sound_t) * ARRAY_LENGTH, cudaHostAllocPortable);
  cudaHostAlloc(&majorBass, sizeof(sound_t) * ARRAY_LENGTH, cudaHostAllocPortable);
  cudaHostAlloc(&minorSoprano, sizeof(sound_t) * ARRAY_LENGTH, cudaHostAllocPortable);
  cudaHostAlloc(&minorBass, sizeof(sound_t) * ARRAY_LENGTH, cudaHostAllocPortable);
  /*
  majorSoprano = (sound_t*) malloc(sizeof(sound_t) * ARRAY_LENGTH);
  majorBass = (sound_t*) malloc(sizeof(sound_t) * ARRAY_LENGTH);
  minorSoprano = (sound_t*) malloc(sizeof(sound_t) * ARRAY_LENGTH);
  minorBass = (sound_t*) malloc(sizeof(sound_t) * ARRAY_LENGTH);
  */
}

/**
 * @brief Sets up and calls functions to create matrices from input files given in command line
 * 
 * @param argc nunmber of command line arguments
 * @param argv array of command line arguments, where the first is the major files, second is minor files
 */
int main(int argc, char** argv) {
  // We require you to input the major and minor files, in that order
  if (argc != 3) {
    printf("Improperly formatted command line input (give two file paths)\n");
    return 0;
  }

  printf("Beginning CUDA Initialization\n");

  auto start = high_resolution_clock::now();

  //Allocate memory for host and device
  allocHost();

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);

  std::cout << "Finished Host Initialization." 
            << duration.count() << "microS" << std::endl;

  start = high_resolution_clock::now();

  initCuda();

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);

  std::cout << "Finished CUDA Initialization." 
            << duration.count() << "microS" << std::endl;

  std::string fileLine;
  std::ifstream majorFile(argv[1]);
  std::ifstream minorFile(argv[2]);
  std::size_t found;

  int minorPart; //part the minor file is on
  int majorPart; //part the major file is on
  int mood = 0; // 0 = major, 1 = minor
  int endFile = 0;
  int sLenMaj = 0;
  int bLenMaj = 0;
  int sLenMin = 0;
  int bLenMin = 0;

  if (!majorFile || !minorFile){ //Throw error if file not found
    std::cerr << "Cannot open file " <<std::endl;
    return false;
  }

  printf("Begin parsing files \n");

  /*** TIME: Start parsing file and queueing CUDA ***/
  start = high_resolution_clock::now();

  //Loop through all given input files, parse file, and add count to device matrices
  while (endFile != 2) { //Keep looping until both files are finished
    if (mood == 0) { //major case
      if (!std::getline(majorFile, fileLine)) { //invalid line, switch to finish other side
        mood = 1;
        endFile++;
        continue;
      }
      found = fileLine.find(' ');
      if (fileLine.find('H') != std::string::npos) { //Set correct part, soprano or bass
        majorPart = 1;
      }
      else if (fileLine.find('L') != std::string::npos){
        majorPart = 0;
      }
      else if (found != std::string::npos){ //insert into correct notes line
        if (majorPart == 0){   
          majorBass[bLenMaj].tone = std::stoi(fileLine.substr(0, found));
          majorBass[bLenMaj].duration = std::stoi(fileLine.substr(found+1));
          bLenMaj ++;
        }
        else {
          majorSoprano[sLenMaj].tone = std::stoi(fileLine.substr(0, found));
          majorSoprano[sLenMaj].duration = std::stoi(fileLine.substr(found+1));
          sLenMaj ++;
        }
      }
    }
    else { //minor case
      if (!std::getline(minorFile, fileLine)) { //invalid line, switch to finish other side
        endFile++;
        //Wait for major stream to open up before overwriting
        continue;
      }
      found = fileLine.find(' ');
      if (fileLine.find('H') != std::string::npos) { //Set correct part, soprano or bass
        minorPart = 1;
      }
      else if (fileLine.find('L') != std::string::npos){
        minorPart = 0;
      }
      else if (found != std::string::npos){ //insert into correct notes line
        if (minorPart == 0) {
          minorBass[bLenMin].tone = std::stoi(fileLine.substr(0, found));
          minorBass[bLenMin].duration = std::stoi(fileLine.substr(found+1));
          bLenMin ++;
        }
        else {
          minorSoprano[sLenMin].tone = std::stoi(fileLine.substr(0, found));
          minorSoprano[sLenMin].duration = std::stoi(fileLine.substr(found+1));
          sLenMin ++;
        }
      }
    }   
  }

  start = high_resolution_clock::now();

  countTransitionsCuda(majorSoprano, sLenMaj, majorBass, bLenMaj, 0);
  countTransitionsCuda(minorSoprano, sLenMin, minorBass, bLenMin, 1);

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);

  std::cout << "Finished Cuda Counting\n " 
            << duration.count() << "microS" << std::endl;

  /*** TIME: copying to host ***/
  start = high_resolution_clock::now();

  cudaToHost();
  printf("Finished copying to host \n");
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);

  std::cout << "Finished copying to host. " 
            << duration.count() << "microS" << std::endl;

  /*** TIME: freeing up memory ***/
  start = high_resolution_clock::now();

  //Free all device memory
  freeCuda();

  start = high_resolution_clock::now();

  countTransitions(majorSoprano, sLenMaj, majorBass, bLenMaj, 0);
  countTransitions(minorSoprano, sLenMin, minorBass, bLenMin, 1);

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  std::cout << "FInished Seq Counting" << duration.count() << "microS" << std::endl;

//Free all host matrices
  free(majorHighNotes);
  free(majorLowNotes);
  free(majorChords);
  free(minorHighNotes);
  free(minorLowNotes);
  free(minorChords);
  cudaFree(majorSoprano);
  cudaFree(majorBass);
  cudaFree(minorSoprano);
  cudaFree(minorBass);

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);

  std::cout << "Finished freeing up data structures. " 
            << duration.count() << "microS" << std::endl;

  return 0;
}
