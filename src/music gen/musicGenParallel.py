# Module: musicGenParallel.py
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

#Music settings
NUMMEASURES = 4
BEATSPERMEASURE = 4
NUMPARTS = 10
NUMTONES = 74 #TODO: FIX THIS NUMBER
NUMDUR = 15
CHORDOFFSET = 101
NUMOCTAVES = 6

#(note, duration) datatype
note = np.dtype([('tone', np.int32), ('duration', np.int32)])

#Generates next note based on previous notes
#Returns a (tone, duration) pair
@cuda.jit(device = True)
def nextNote(noteIndex, partName, matrix, deviceMusic, tonic, mood, rng_states):
	#Used for probablity calculations
	probSum = 0
	probIndex = 0

	#Note to be picked
	tone = 0
	duration = 0

	if partName == 0 : #chord
		if noteIndex == 0 : #End on base chord in some inversion
			#Get mid of chord
			mid = None
			if mood == 0 : #Minor chord
				mid = tonic + 3
			else: #Major chord
				mid = tonic + 4

			#Get random inversion
			prob = xoroshiro128p_uniform_float32(rng_states, cuda.threadIdx.x)
			if prob < 0.33 : 
				tone = (tonic + 7) + tonic * 12 + mid * 144
			else if prob < 0.66 :
				tone = mid + (tonic + 7) * 12 + tonic * 144
			else :
				tone = tonic + mid * 12 + (tonic + 7) * 144

		else : #Get chord based on previous chord
			#Get matrix line
			matrixIndex = ((deviceMusic[noteIndex - 1]['tone'] % 12 - tonic) + 
				(deviceMusic[noteIndex - 1]['tone'] / 12 % 12 - tonic) * 12 + 
				(deviceMusic[noteIndex - 1]['tone'] / 144 - tonic) * 144) - CHORDOFFSET
			probLine = matrix[matrixIndex]

			#Get random probablity from 0 to 1, and find corresponding note
			prob = xoroshiro128p_uniform_float32(rng_states, cuda.threadIdx.x)
			while(probSum < prob)
				probSum = probSum + probLine[probIndex]
				probIndex = probIndex + 1
			tone = ((probIndex % 12 + tonic) + (probIndex / 12 % 12 + tonic) * 12 + 
				(probIndex / 144 + tonic) * 144) + CHORDOFFSET

  else : #melodic line
  	if noteIndex < 2 : #Pick random notes
  		prob = xoroshiro128p_uniform_float32(rng_states, cuda.threadIdx.x)
  		if prob < 0.5 : #Return tonic most of the time
  			tone = tonic + 12 * (prob * NUMOCTAVES)
  		else if prob < 0.75 : #Return 5th a large portion of the time
  			tone = tonic + 7 + 12 * (prob * NUMOCTAVES)
  		else : #Random note
  			tone = int(prob * NUMTONES)

  		duration = int(prob * 16)

  	else: #Get note based on previous notes
  		#Get matrix line
			matrixIndex = (((deviceMusic[noteIndex - 1]['tone'] - tonic)* NUMDUR + 
				deviceMusic[noteIndex - 1]['duration'])
				* NUMNOTES) + ((deviceMusic[noteIndex - 2]['tone'] - tonic)* NUMDUR + 
				deviceMusic[noteIndex - 2]['duration'])
			probLine = matrix[matrixIndex]

			#Get random probablity from 0 to 1, and find corresponding note
			prob = xoroshiro128p_uniform_float32(rng_states, cuda.threadIdx.x)
			while(probSum < prob)
				probSum = probSum + probLine[probIndex]
				probIndex = probIndex + 1
			tone = probIndex / NUMDUR + tonic
			duration = probIndex % NUMDUR

	return (tone, duration)

#Creates and stores a total of NUMMEASURES of music in parallel,
#split by parts
@cuda.jit(device = True)
def generatePart(deviceHigh, deviceLow, deviceChords, deviceParts, deviceMusic, tonic, mood, rng_states):
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
		newNote = nextNote(noteIndex, deviceParts[partIndex], matrix, deviceMusic, tonic, mood, rng_states)

		numBeatsFilled = numBeatsFilled + newNote['duration']

		#If too long, chop note off at end of measure
		if (numBeatsFilled > NUMMEAURES * BEATSPERMEASURE) :
			newNote['duration'] = NUMMEAURES * BEATSPERMEASURE - numBeatsFilled + newNote[['duration']] 

		#Add note to music array
		deviceMusic[partIndex][noteIndex] = newNote
		noteIndex = noteIndex + 1
	return

#Calls GPU to generate NUMMEASURES total measures of music
#Returns a 2D array containing the (tone, duration) pairs of the
#music generated, split by part (as assigned by the parts variable)
def generateMusic(highNotes, lowNotes, chords, parts, tonic, mood):
	#Pass matrices to device
	deviceHigh = cuda.to_device(highNotes)
	deviceLow = cuda.to_device(lowNotes)
	deviceChords = cuda.to_device(chords)

	#Pass part assignment to device
	deviceParts = cuda.to_device(parts)

	#Create device music array
	deviceMusic = cuda.device_array((10, NUMMEASURES * BEATSPERMEASURE * 4), dtype = note)

	#Generate parts in parallel, and copy to host
	rng_states = create_xoroshiro128p_states(10, seed=1)
	generatePart[1, 10](deviceHigh, deviceLow, deviceChords, deviceParts, deviceMusic, tonic, mood, rng_states)
	music = numpy.empty(shape = deviceMusic.shape, dtype = note)
	deviceMusic.copy_to_host(music)

	return music