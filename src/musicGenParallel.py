'''
musicGenParallel.py - parallel implementation for music generation.

Uses Python multiprocessing to map multiple jobs to multiple processors
to work on in parallel.
'''
import numpy as np
import threading
import time
from CONST_SERVER import *
import multiprocessing as mp
from functools import partial
from pickle import load

def nextNote(prev1, prev2, partMarker, matrix, mood):
'''
Generates next note based on previous notes
returns: a (tone, duration) pair
'''
	# Note to be picked
  tone = None 
  duration = None

  if partMarker == 0 : # chord
    if prev1 == None : # End on base chord in some inversion
      # Get mid of chord
      mid = None
      if mood == 0 : # Minor chord
        mid = 3
      else: # Major chord
        mid = 4

			# Get random inversion
      tone = np.random.choice([7 + mid * 144, mid + 7*12, mid * 12 + 7 * 144]) + CHORDOFFSET

    else : # Get chord based on previous chord
			# Get normalized matrix line
      matrixIndex = prev1[0]  - CHORDOFFSET
      lineSum = np.sum(matrix[matrixIndex])
      if lineSum != 0: # Use normalized row distribution
        probLine = matrix[matrixIndex] / lineSum
        tone = np.random.choice(1728, p = probLine) + CHORDOFFSET
      else: # Use uniform distribution
        tone = np.random.choice(1728) + CHORDOFFSET				

		# Get random chord duration from 4 to 15 (at least quarter note)
    duration = np.random.randint(4, 16)

  else : #melodic line
    if prev2 == None or prev1 == None : #Pick random notes weighted by music theoretic ideas
      tone = np.random.choice(12, p = [0.5, 0, 0.1, 0.1, 0.1, 0.2, 0, 0, 0, 0, 0, 0])
      tone = tone + np.random.randint(NUMOCTAVES) * 12 #Get random octave
      duration = np.random.randint(0, 16)

    # Get note based on previous notes
    else: 
  		# Get matrix line
      matrixIndex = ((prev1[0] * NUMDUR + 
        prev1[1]) * NUMNOTES) + (prev2[0] * NUMDUR + prev2[1])
      lineSum = np.sum(matrix[matrixIndex])
      if (lineSum != 0): # Use normalized row distribution
        probLine = matrix[matrixIndex] / lineSum
        noteIndex = np.random.choice(NUMNOTES, p = probLine)
        tone = noteIndex / NUMDUR
        duration = noteIndex % NUMDUR
      else: # Use uniform distribution
        noteIndex = np.random.choice(NUMNOTES)
        tone = noteIndex / NUMDUR
        duration = noteIndex % NUMDUR
 
  return int(tone), int(duration)


def generatePart(matrixType, partMarker, mood):
'''
Creates and stores a total of NUMMEASURES of music in parallel, split by parts.
'''
  if matrixType == 2:
    matrix = MAJORHIGH = load(open(MAJOR_HIGH_FILE, 'rb'))
  elif matrixType == 1:
    matrix = MAJORLOW = load(open(MAJOR_LOW_FILE, 'rb'))
  elif matrixType == 0:
    matrix = MAJORCHORD = load(open(MAJOR_CHORD_FILE, 'rb'))
  else:
    return []

  numBeatsFilled = 0
  partMusic = []
  prev1 = None
  prev2 = None

	#Generating notes
  while numBeatsFilled < NUMMEASURES * BEATSPERMEASURE : 
		#Get next note
    tone, duration = nextNote(prev1, prev2, partMarker, matrix, mood)

    numBeatsFilled = numBeatsFilled + (duration + 1)
    #If too long, chop note off at end of measure
    if (numBeatsFilled > NUMMEASURES * BEATSPERMEASURE) :
      duration = duration - (numBeatsFilled - NUMMEASURES * BEATSPERMEASURE)

		#Add note to music array 
    partMusic.append([tone, duration])

    prev2 = prev1
    prev1 = [tone, duration]

  return partMusic

def f (mood, partMarker):
'''
Lambda function used to map on our set of work to do in parallel.

returns: melodic line for the specified partMarker
'''
  return generatePart(partMarker, partMarker, mood)

def generateMusic(highNotes, lowNotes, chords, parts, mood):
'''
Calls GPU to generate NUMMEASURES total measures of music
Returns a 2D array containing the (tone, duration) pairs of the
music generated, split by part (as assigned by the parts variable)

returns: list list [note, duration], containing all 10 voices
'''
  pool = mp.Pool(processes=10)

  fPart = partial(f, mood)

  music = pool.map(fPart, parts)

  pool.close()

  return music
