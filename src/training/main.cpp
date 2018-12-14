#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "training.h"
#include <iostream>
#include <fstream>

#include <cuda.h>
#include<cuda_runtime.h>

using namespace std;

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

  //Free all host matrices
  cudaFreeHost(majorHighNotes);
  cudaFreeHost(majorLowNotes);
  free(majorChords);
  cudaFreeHost(minorHighNotes);
  cudaFreeHost(minorLowNotes);
  free(minorChords);
}

/**
 * @brief uses cudaHostAlloc to allocated pinned memory for global structures
 */
void allocHost(){

  //Allocate memory for all final host matrices
  cudaHostAlloc(&majorHighNotes, sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES), cudaHostAllocPortable);
  cudaHostAlloc(&majorLowNotes, sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES), cudaHostAllocPortable);
  cudaHostAlloc(&minorHighNotes, sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES), cudaHostAllocPortable);
  cudaHostAlloc(&minorLowNotes, sizeof(int) * (MATRIX_BLOCK_ROWS * NUM_GPU_PER_MATRIX * NUM_NOTES), cudaHostAllocPortable);
   
  //Chord Matrices
  majorChords =  (int *) malloc(sizeof(int) * (NUM_CHORDS * NUM_CHORDS));
  minorChords =  (int *) malloc(sizeof(int) * (NUM_CHORDS * NUM_CHORDS));
 
  //Arrays of notes read from files, initialized to ARRAY_LENGTH (1000) in length
  cudaHostAlloc(&majorSoprano, sizeof(sound_t) * ARRAY_LENGTH, cudaHostAllocPortable);
  cudaHostAlloc(&majorBass, sizeof(sound_t) * ARRAY_LENGTH, cudaHostAllocPortable);
  cudaHostAlloc(&minorSoprano, sizeof(sound_t) * ARRAY_LENGTH, cudaHostAllocPortable);
  cudaHostAlloc(&minorBass, sizeof(sound_t) * ARRAY_LENGTH, cudaHostAllocPortable);
}

/**
 * @brief Sets up and calls functions to create matrices from input files given in command line
 * 
 * @param argc nunmber of command line arguments
 * @param argv array of command line arguments, where the first is the major files, second is minor files
 */
int main(int argc, char** argv) {

  //If there is not a directory to look at, stop
  if (argc != 3) {
    printf("Improperly formatted command line input (give two file paths)\n");
    return 0;
  }

  //Allocate memory for host and device
  allocHost();
  initCuda();

  printf("Finished Initialization \n");

  std::string fileLine;
  std::ifstream majorFile(argv[1]);
  std::ifstream minorFile(argv[2]);
  std::size_t found;

  int minorPart; //part the minor file is on
  int majorPart; //part the major file is on
  int mood = 0; // 0 = major, 1 = minor
  int endFile = 0;
  int sLen = 0;
  int bLen = 0;

  if (!majorFile || !minorFile){ //Throw error if file not found
    std::cerr << "Cannot open file " <<std::endl;
    return false;
  }

  printf("Begin parsing files \n");

  //Loop through all given input files, parse file, and add count to device matrices
  while (endFile != 2) { //Keep looping until both files are finished
    if (mood == 0) { //major case
      if (!std::getline(majorFile, fileLine)) { //invalid line, switch to finish other side
        countTransitionsCuda(majorSoprano, sLen, majorBass, bLen, mood);
        sLen = 0;
        bLen = 0;
        mood = 1;
        endFile++;
        //Wait for minor stream to open up before overwriting
        cudaStreamSynch(mood);
        break;
      }
      if (fileLine.find('H') != std::string::npos) { //Set correct part, soprano or bass
        majorPart = 1;
      }
      else if (fileLine.find('L') != std::string::npos){
        majorPart = 0;
      }
      else if (found = fileLine.find(' ') != std::string::npos){ //insert into correct notes line
        if (majorPart == 0){
          majorBass[bLen].tone = std::stoi(fileLine.substr(0, found));
          majorBass[bLen].duration = std::stoi(fileLine.substr(found+1));
          bLen ++;
        }
        else {
          majorSoprano[sLen].tone = std::stoi(fileLine.substr(0, found));
          majorSoprano[sLen].duration = std::stoi(fileLine.substr(found+1));
          sLen ++;
        }
      }
      //If the notes run past the array length, send array over and switch maj/min
      if (bLen >= ARRAY_LENGTH || sLen >= ARRAY_LENGTH) {
        countTransitionsCuda(majorSoprano, sLen, majorBass, bLen, mood);
        sLen = 0;
        bLen = 0;
        mood = 1;
        //Wait for minor stream to open up before overwriting
        cudaStreamSynch(mood);
        break;
      }
    }
    else { //minor case
      if (!std::getline(minorFile, fileLine)) { //invalid line, switch to finish other side
        countTransitionsCuda(minorSoprano, sLen, minorBass, bLen, mood);
        sLen = 0;
        bLen = 0;
        mood = 0;
        endFile++;
        //Wait for major stream to open up before overwriting
        cudaStreamSynch(mood);
        break;
      }
      if (fileLine.find('H') != std::string::npos) { //Set correct part, soprano or bass
        minorPart = 1;
      }
      else if (fileLine.find('L') != std::string::npos){
        minorPart = 0;
      }
      else if (found = fileLine.find(' ') != std::string::npos){ //insert into correct notes line
        if (minorPart == 0) {
          minorBass[bLen].tone = std::stoi(fileLine.substr(0, found));
          minorBass[bLen].duration = std::stoi(fileLine.substr(found+1));
          bLen ++;
        }
        else {
          minorSoprano[sLen].tone = std::stoi(fileLine.substr(0, found));
          minorSoprano[sLen].duration = std::stoi(fileLine.substr(found+1));
          sLen ++;
        }
      }
      //If the notes run past the array length, send array over and switch maj/min
      if (bLen >= ARRAY_LENGTH || sLen >= ARRAY_LENGTH){
        countTransitionsCuda(minorSoprano, sLen, minorBass, bLen, mood);
        sLen = 0;
        bLen = 0;
        mood = 0;
        //Wait for major stream to open up before overwriting
        cudaStreamSynch(mood);
        break;
      }
    }   
  }
 
  printf("Finished queuing transitions \n");

  cudaToHost();
  printf("Finished copying to host \n");

  //Free all device memory
  freeCuda();

  printf("Start outputting matrices \n");
  //output matrices to files
  outputMatrices();

  return 0;
}
