#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "training.h"
#include <iostream>
#include <fstream>

using namespace std;

void outputMatrices(double* highNotes, double* lowNotes, double* chords){
	std::ofstream outFile;
	outFile.open("highMatrix.txt");
	for (int i = 0; i < NUM_NOTES * NUM_NOTES; i ++){
		for (int j = 0; j < NUM_NOTES; j++){
			outFile << highNotes[i * NUM_NOTES + j] << " ";
		}
		outFile << "\n";
	}
	outFile.close();

	outFile.open("lowMatrix.txt");
	for (int i = 0; i < NUM_NOTES * NUM_NOTES; i ++){
		for (int j = 0; j < NUM_NOTES; j++){
			outFile << lowNotes[i * NUM_NOTES + j] << " ";
		}
		outFile << "\n";
	}
	outFile.close();

	outFile.open("chordMatrix.txt");
	for (int i = 0; i < NUM_CHORDS; i ++){
		for (int j = 0; j < NUM_CHORDS; j++){
			outFile << chords[i * NUM_CHORDS + j] << " ";
		}
		outFile << "\n";
	}
	outFile.close();
}

int main(int argc, char** argv)
{
    //Set up all final matrices
    double* highNotes = malloc(sizeof(double) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
    double* lowNotes =  malloc(sizeof(double) * (NUM_NOTES * NUM_NOTES * NUM_NOTES));
    double* chords =  malloc(sizeof(double) * (NUM_CHORDS * NUM_CHORDS));

    double* deviceHigh;
    double* deviceLow;
    double* deviceChord;

	sound_t* soprano = malloc(sizeof(sound_t) * INIT_ARRAY_LENGTH); //set up notes array
	sound_t* bass = malloc(sizeof(sound_t) * INIT_ARRAY_LENGTH);

	int maxLen = INIT_ARRAY_LENGTH;
    int sLen = 0;
    int bLen = 0;

	//Read given files
	for(int fileIndex = 0; fileIndex < argc; fileIndex++){
		std::string fileLine;
		int currentPart;
    	std::ifstream file(argv[fileIndex]);
    	if (!file){
    		std::cerr << "Cannot open file : " <<argv[fileIndex]<<std::endl;
    		return false;
    	}
    	std::size_t found;
    	while(std::getline(file, fileLine)){
    		if (fileLine.find('H') != std::string::npos) { //Set correct part
                currentPart = 1;
    		}
    		else if (fileLine.find('L') != std::string::npos){
    			currentPart = 0;
    		}
    		else if (found = fileLine.find(' ') != std::string::npos){ //insert into correct notes line
    			if (currentPart == 0){
    				bass[bLen].tone = stoi(fileLine.substr(0, found));
    				bass[bLen].duration = stoi(fileLine.substr(found+1));
    				bLen ++;
    			}
    			else{
    				soprano[sLen].tone = stoi(fileLine.substr(0, found));
    				soprano[sLen].duration = stoi(fileLine.substr(found+1));
    				sLen ++;
    			}
    			if (bLen >= maxLen || sLen >= maxLen){
    				maxLen = maxLen * 2;
    				soprano = realloc(soprano, sizeof(sound_t) * maxLen);
    				bass = realloc(bass, sizeof(sound_t) * maxLen);
    			}
    		}
    	}
    	countTransitionsCuda(soprano, bass, deviceHigh, deviceLow, deviceChord);
    }

    normalizeCuda(deviceHigh, deviceLow, deviceChord, highNotes, lowNotes, chords);

    outputMatrices(highNotes, lowNotes, chords);

    return 0;
}