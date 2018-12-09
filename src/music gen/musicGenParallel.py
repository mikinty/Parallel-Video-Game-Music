# Module: musicGenParallel.py
import numpy as np
from pyculib import rand as curand

#Music settings
NUMMEASURES = 4
BEATSPERMEASURE = 4
NUMPARTS = 10
NUMTONES = 101 #TODO: FIX THIS NUMBER
NUMDUR = 15
CHORDOFFSET = 101

#(note, duration) datatype
note = np.dtype([('tone', int), ('duration', int)])

#Random number generator
prng = rand.PRNG(rndtype=rand.PRNG.XORWOW)

#Generates next note based on previous notes
#Returns a (tone, duration) pair
@cuda.jit(device = True)
def nextNote(noteIndex, partName, matrix, deviceMusic):
	probSum = 0
	i = 0
	tone = None
	duration = None

	if noteIndex == 0 : #Randomly generate first note


  else if noteIndex == 1 && partName != 0 : #Randomly generate second note

	else if partName == 0 : #chord
		#Get matrix line
		matrixIndex = (deviceMusic[noteIndex - 1][['tone']] - CHORDOFFSET) * deviceMusic[noteIndex - 1][['duration']] - 1
		probLine = matrix[matrixIndex]

		#Get random probablity from 0 to 1, and find corresponding note
		prob = prng.
		while(probSum < prob)
			probSum = probSum + probLine[i]
			i = i + 1
		duration = (i + 1) % NUMDUR + 1
		tone = 

  else : #melodic line

	return 

#Creates and stores a total of NUMMEASURES of music in parallel,
#split by parts
@cuda.jit(device = True)
def generatePart(deviceHigh, deviceLow, deviceChords, deviceParts, deviceMusic):
	partIndex = cuda.threadIdx.x
	numBeatsFilled = 0
	matrix = None
	noteIndex = 0

	#Determine which matrix to use
	if deviceParts[partIndex] == 0 : #chord
		matrix = deviceChords
	else if deviceParts[partIndex] == 1 : #bass
		matrix = deviceLow
	else if deviceParts[partIndex] == 2 : #soprano
		matrix = deviceHigh
	else : #silent
		return

	#Generating notes
	while numBeatsFilled < NUMMEAURES * BEATSPERMEASURE : 
		#Get next note
		newNote = nextNote(noteIndex, deviceParts[partIndex], matrix, deviceMusic, tonic)

		numBeatsFilled = numBeatsFilled + newNote[['duration']]

		#If too long, chop note off at end of measure
		if (numBeatsFilled > NUMMEAURES * BEATSPERMEASURE) :
			newNote[['duration']] = NUMMEAURES * BEATSPERMEASURE - numBeatsFilled + newNote[['duration']] 
		
		#Transpose new note to correct key
		newNote[['tone']] = newNote[['tone']] + tonic

		#Add note to music array
		deviceMusic[partIndex] = np.append(deviceMusic[partIndex], newNote)
		noteIndex = noteIndex + 1
	return

#Calls GPU to generate NUMMEASURES total measures of music
#Returns a 2D array containing the (tone, duration) pairs of the
#music generated, split by part (as assigned by the parts variable)
def generateMusic(highNotes, lowNotes, chords, parts, tonic):
	#Pass matrices to device
	deviceHigh = cuda.to_device(highNotes)
	deviceLow = cuda.to_device(lowNotes)
	deviceChords = cuda.to_device(chords)

	#Pass part assignment to device
	deviceParts = cuda.to_device(parts)

	#Create device music array
	deviceMusic = cuda.device_array((10, NUMMEASURES * BEATSPERMEASURE * 4), dtype = note)

	#Generate parts in parallel, and copy to host
	generatePart[1, 10](deviceHigh, deviceLow, deviceChords, deviceParts, deviceMusic, tonic)
	music = numpy.empty(shape = deviceMusic.shape, dtype = note)
	deviceMusic.copy_to_host(music)

	return music