# Module: musicGenParallel.py
import numpy as np
import time
from CONST_SERVER import *

#Generates next note based on previous notes
#Returns a (tone, duration) pair
def nextNote(prev1, prev2, partMarker, matrix, mood):
	#Note to be picked
  tone = None 
  duration = None

  if partMarker == 0 : #chord
    if prev1 == None : #End on base chord in some inversion
      #Get mid of chord
      mid = None
      if mood == 0 : #Minor chord
        mid = 3
      else: #Major chord
        mid = 4

			#Get random inversion
      tone = np.random.choice([7 + mid * 144, mid + 7*12, mid * 12 + 7 * 144]) + CHORDOFFSET

    else : #Get chord based on previous chord
			#Get normalized matrix line
      matrixIndex = prev1[0]  - CHORDOFFSET
      lineSum = np.sum(matrix[matrixIndex])
      if lineSum != 0: #Use normalized row distribution
        probLine = matrix[matrixIndex] / lineSum
        tone = np.random.choice(1728, p = probLine) + CHORDOFFSET
      else: #Use uniform distribution
        tone = np.random.choice(1728) + CHORDOFFSET				

		#Get random chord duration from 4 to 15 (at least quarter note)
    duration = np.random.randint(4, 16)

  else : #melodic line
    if prev2 == None or prev1 == None : #Pick random notes weighted by music theoretic ideas
      tone = np.random.choice(12, p = [0.5, 0, 0.1, 0.1, 0.1, 0.2, 0, 0, 0, 0, 0, 0])
      tone = tone + np.random.randint(NUMOCTAVES) * 12 #Get random octave
      duration = np.random.randint(0, 16)

    else: #Get note based on previous notes
  		#Get matrix line
      matrixIndex = ((prev1[0] * NUMDUR + 
        prev1[1]) * NUMNOTES) + (prev2[0] * NUMDUR + prev2[1])
      lineSum = np.sum(matrix[matrixIndex])
      if (lineSum != 0): #Use normalized row distribution
        probLine = matrix[matrixIndex] / lineSum
        noteIndex = np.random.choice(NUMNOTES, p = probLine)
        tone = noteIndex / NUMDUR
        duration = noteIndex % NUMDUR
      else: #Use uniform distribution
        noteIndex = np.random.choice(NUMNOTES)
        tone = noteIndex / NUMDUR
        duration = noteIndex % NUMDUR
 
  return int(tone), int(duration)

#Creates and stores a total of NUMMEASURES of music in parallel,
#split by parts
def generatePart(matrix, partMarker, mood, partMusic, numMeasures):
  numBeatsFilled = 0

  prev1 = None
  prev2 = None

	#Generating notes
  while numBeatsFilled < numMeasures * BEATSPERMEASURE: 
		#Get next note
    tone, duration = nextNote(prev1, prev2, partMarker, matrix, mood)

    numBeatsFilled = numBeatsFilled + (duration + 1)
    #If too long, chop note off at end of measure
    if (numBeatsFilled > numMeasures * BEATSPERMEASURE) :
      duration = duration - (numBeatsFilled - numMeasures * BEATSPERMEASURE)

		#Add note to music array 
    partMusic.append([tone, duration])

    prev2 = prev1
    prev1 = [tone, duration]

  return

#Calls GPU to generate NUMMEASURES total measures of music
#Returns a 2D array containing the (tone, duration) pairs of the
#music generated, split by part (as assigned by the parts variable)
def generateMusic(highNotes, lowNotes, chords, parts, mood, numMeasures):

  music = [[] for i in range(10)]

  for index, partMarker in enumerate(parts):
    t = None
    if (partMarker == -1): #silent
      continue
    elif (partMarker == 0): #chord
      generatePart(chords, partMarker, mood, music[index], numMeasures)
    elif (partMarker == 1): #bass
      generatePart(lowNotes, partMarker, mood, music[index], numMeasures)
    elif (partMarker == 2): #soprano
      generatePart(highNotes, partMarker, mood, music[index], numMeasures)
    else: #Error
      print ('Error: Part assignment not allowed')

  return music
