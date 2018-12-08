#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "musicGen.h"
#include <iostream>
#include <fstream>

using namespace std;

//Sets up and establishes link between server and client
void initServerLink() {

}

//Waits for the client to send over choice of major or minor
int receiveMajorMinor() {

}

//Sends message to client noting an error has occured, and to close link + retry
void sendErrorMessage(){

}

//Closes link between server and client
void closeServerLink(){

}

int main(int argc, char** argv) {

	//Establish Server-Client link
	initServerLink();

	//Wait to receive major/minor choice from client
	int mood = receiveMajorMinor(); //0 = minor, 1 = major

	//Reads in chord, high melody, and low melody matrices of the proper mood
	if (mood == 0) { //minor

	}
	else if (mood == 1) { //major

	}
	else { //error - not a valid mood
		sendErrorMessage();
		closeServerLink();
		return false;
	}

}