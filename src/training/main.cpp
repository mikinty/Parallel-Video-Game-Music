#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "training.h"
#include <iostream>
#include <fstream>

using namespace std;

//Define global variables
float* majorHighNotes;
float* majorLowNotes;
float* majorChords;
float* minorHighNotes;
float* minorLowNotes;
float* minorChords;

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

	outFile.open("majorLowMatrixNew.txt");
	for (int i = 0; i < NUM_NOTES * NUM_NOTES; i ++){
		for (int j = 0; j < NUM_NOTES; j++){
			outFile << majorLowNotes[i * NUM_NOTES + j] << " ";
		}
		outFile << "\n";
	}
	outFile.close();

	outFile.open("majorChordMatrixNew.txt");
	for (int i = 0; i < NUM_CHORDS; i ++){
		for (int j = 0; j < NUM_CHORDS; j++){
			outFile << majorChords[i * NUM_CHORDS + j] << " ";
		}
		outFile << "\n";
	}
	outFile.close();

  outFile.open("minorHighMatrixNew.txt");
  for (int i = 0; i < NUM_NOTES * NUM_NOTES; i ++){
    for (int j = 0; j < NUM_NOTES; j++){
      outFile << minorHighNotes[i * NUM_NOTES + j] << " ";
    }
    outFile << "\n";
  }
  outFile.close();

  outFile.open("minorLowMatrixNew.txt");
  for (int i = 0; i < NUM_NOTES * NUM_NOTES; i ++){
    for (int j = 0; j < NUM_NOTES; j++){
      outFile << minorLowNotes[i * NUM_NOTES + j] << " ";
    }
    outFile << "\n";
  }
  outFile.close();

  outFile.open("minorChordMatrixNew.txt");
  for (int i = 0; i < NUM_CHORDS; i ++){
    for (int j = 0; j < NUM_CHORDS; j++){
      outFile << minorChords[i * NUM_CHORDS + j] << " ";
    }
    outFile << "\n";
  }
  outFile.close();

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

  //Free all host matrices
  free(majorHighNotes);
  free(majorLowNotes);
  free(majorChords);
  free(minorHighNotes);
  free(minorLowNotes);
  free(minorChords);
}

/**
 * @brief Sets up and calls functions to create matrices from input files given in command line
 * 
 * @param argc nunmber of command line arguments
 * @param argv array of command line arguments, where every argument is a path to an input file
 */
int main(int argc, char** argv) {

  //Allocate memory for all final host matrices
  majorHighNotes = (float *) malloc(sizeof(float) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
  majorLowNotes =  (float *) malloc(sizeof(float) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
  majorChords =  (float *) malloc(sizeof(float) * (NUM_CHORDS * NUM_CHORDS));
  minorHighNotes = (float *) malloc(sizeof(float) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
  minorLowNotes =  (float *) malloc(sizeof(float) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
  minorChords =  (float *) malloc(sizeof(float) * (NUM_CHORDS * NUM_CHORDS));

  initCuda();

  //Arrays of notes read from files, initialized to INIT_ARRAY_LENGTH (1000) in length
  sound_t* soprano = (sound_t*) malloc(sizeof(sound_t) * INIT_ARRAY_LENGTH);
	sound_t* bass = (sound_t *) malloc(sizeof(sound_t) * INIT_ARRAY_LENGTH);
  int maxLen = INIT_ARRAY_LENGTH;

  //Loop through all given input files, parse file, and add count to device matrices
	for(int fileIndex = 0; fileIndex < argc; fileIndex++){
    int sLen = 0;
    int bLen = 0;
    std::string mood;
    int currentPart;

		std::string fileLine;
    std::ifstream file(argv[fileIndex]);
    std::size_t found;

    if (!file){ //Throw error if file not found
    	std::cerr << "Cannot open file : " <<argv[fileIndex]<<std::endl;
    	return false;
    }

    std::getline(file, mood); //Grab major or minor mark

    while(std::getline(file, fileLine)){ //run through every line (note or chord) in the file
      if (fileLine.find('H') != std::string::npos) { //Set correct part, soprano or bass
        currentPart = 1;
      }
      else if (fileLine.find('L') != std::string::npos){
      	currentPart = 0;
      }
      else if (found = fileLine.find(' ') != std::string::npos){ //insert into correct notes line
      	if (currentPart == 0){
      		bass[bLen].tone = std::stoi(fileLine.substr(0, found));
      		bass[bLen].duration = std::stoi(fileLine.substr(found+1));
      		bLen ++;
      	}
      	else{
      		soprano[sLen].tone = std::stoi(fileLine.substr(0, found));
      		soprano[sLen].duration = std::stoi(fileLine.substr(found+1));
      		sLen ++;
    		}

        //If the notes run past the array length, re-allocate for more space
        if (bLen >= maxLen || sLen >= maxLen){
      		maxLen = maxLen * 2;
    				soprano = (sound_t *) realloc(soprano, sizeof(sound_t) * maxLen);
          	bass = (sound_t *) realloc(bass, sizeof(sound_t) * maxLen);
        }
    	}
    }

    countTransitionsCuda(soprano, sLen, bass, bLen, mood);
  }

  //Free the arrays used to parse input files
  free(soprano);
  free(bass);

  //Normalize the transition matrices and move to host
  normalizeCuda();

  //Free all device memory
  freeCuda();

  //output matrices to files
  outputMatrices();

  return 0;
}
