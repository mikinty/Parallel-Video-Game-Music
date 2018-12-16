#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "training.h"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

//Define global variables
int* majorHighNotes;
int* majorLowNotes;
int* majorChords;
int* minorHighNotes;
int* minorLowNotes;
int* minorChords;

/**
 * @brief Outputs matrices to files
 */
void outputMatrices() {
  
  //For each matrix, copy matrix into file
	std::ofstream outFile;
	/*outFile.open("majorHighMatrixNew.txt");
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
*/

  /*
  outFile.open("minorChordMatrixSeq.txt");
  for (int i = 0; i < NUM_CHORDS; i ++){
    for (int j = 0; j < NUM_CHORDS; j++){
      outFile << minorChords[i * NUM_CHORDS + j] << " ";
    }
    outFile << "\n";
  }
  outFile.close();

  printf("Printing 6/6 complete\n");
  */

  //Remove all old files, and replace with the new matrix files
  /*remove("majorHighMatrix.txt");
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

  printf("File renaming complete\n");*/
}

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

/**
 * @brief uses cudaHostAlloc to allocated pinned memory for global structures
 */
void allocHost(){

  //Allocate memory for all final host matrices
  majorHighNotes = (int *) malloc(sizeof(int) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
  majorLowNotes = (int *) malloc(sizeof(int) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
  minorHighNotes = (int *) malloc(sizeof(int) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
  minorLowNotes = (int *) malloc(sizeof(int) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
   
  //Chord Matrices
  majorChords =  (int *) malloc(sizeof(int) * (NUM_CHORDS * NUM_CHORDS));
  minorChords =  (int *) malloc(sizeof(int) * (NUM_CHORDS * NUM_CHORDS));

  majorSoprano = (sound_t*) malloc(sizeof(sound_t) * ARRAY_LENGTH);
  majorBass = (sound_t*) malloc(sizeof(sound_t) * ARRAY_LENGTH);
  minorSoprano = (sound_t*) malloc(sizeof(sound_t) * ARRAY_LENGTH);
  minorBass = (sound_t*) malloc(sizeof(sound_t) * ARRAY_LENGTH);
}

void countTransitions(sop, sLen, bass, bLen, mood){
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

  printf("Begin host malloc\n");
  auto start = high_resolution_clock::now();

  // Allocate memory for host and device
  allocHost();

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);

  std::cout << "Finished Initialization." 
            << duration.count() << "microS" << std::endl;

  std::string fileLine;
  std::ifstream majorFile(argv[1]);
  std::ifstream minorFile(argv[2]);
  std::size_t found;

  int minorPart; //part the minor file is on
  int majorPart; //part the major file is on
  int mood = 0; // 0 = major, 1 = minor
  int curTone = -1;
  int curDur = -1;
  int prevTone1 = -1;
  int prevDur1 = -1;
  int prevTone2 = -1;
  int prevDur2 = -1;

  if (!majorFile || !minorFile){ //Throw error if file not found
    std::cerr << "Cannot open file " <<std::endl;
    return false;
  }

  printf("Begin parsing files \n");
  start = high_resolution_clock::now();
  
  //Loop through all given input files, parse file, and add count to device matrices
  while (std::getline(majorFile, fileLine)) { //Keep looping until both files are finished
    found = fileLine.find(' ');
    if (fileLine.find('H') != std::string::npos) { //Set correct part, soprano or bass
       majorPart = 1;
    }
    else if (fileLine.find('L') != std::string::npos){
      majorPart = 0;
    }
    else if (found != std::string::npos){ //insert into correct notes line
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
  }
  while(std::getline(minorFile, fileLine)) { //invalid line, switch to finish other side
    found = fileLine.find(' ');
    if (fileLine.find('H') != std::string::npos) { //Set correct part, soprano or bass
      minorPart = 1;
    }
    else if (fileLine.find('L') != std::string::npos){
      minorPart = 0;
    }
    else if (found != std::string::npos){ //insert into correct notes line
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
  }

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);

  std::cout << "Finished Parsing." 
            << duration.count() << "microS" << std::endl;
  
  start = high_resolution_clock::now();
  countTransitions(minorSoprano, minorBass, mood);
  countTransitions(majorSoprano, majorBass, mood);
  duration = duration_cast<microseconds>(stop - start);

  std::cout << "Finished counting." << duration.count() << "microS" << std::endl;
  

  // printf("Start outputting matrices \n");

  // output matrices to files
  // outputMatrices();

  std::cout << "Freeing data structures." << std::endl;
  start = high_resolution_clock::now();

  // Free all host matrices
  free(majorHighNotes);
  free(majorLowNotes);
  free(majorChords);
  free(minorHighNotes);
  free(minorLowNotes);
  free(minorChords);

  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);

  std::cout << "Finished free data structures." 
            << duration.count() << "microS" << std::endl;

  return 0;
}
