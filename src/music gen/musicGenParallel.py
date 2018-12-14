# Module: musicGenParallel.py
import numpy as np
import threading
import time

#Music settings
NUMMEASURES = 4
BEATSPERMEASURE = 4
NUMPARTS = 10
NUMTONES = 73
NUMDUR = 15
CHORDOFFSET = 101
NUMOCTAVES = 6
NUMCHORDS = 1728
NUMNOTES = 1095

#(note, duration) datatype
note = np.dtype([('tone', np.int32), ('duration', np.int32)])

#Generates next note based on previous notes
#Returns a (tone, duration) pair
def nextNote(noteIndex, partMusic, partMarker, matrix, mood):
	#Note to be picked
  tone = 0
  duration = 0

  if partMarker == 0 : #chord
    if noteIndex == 0 : #End on base chord in some inversion
      #Get mid of chord
      mid = None
      if mood == 0 : #Minor chord
        mid = 3
      else: #Major chord
        mid = 4

			#Get random inversion
      tone = np.random.choice([7 + mid * 144, mid + 7*12, mid * 12 + 7 * 144])

    else : #Get chord based on previous chord
			#Get normalized matrix line
      matrixIndex = partMusic[noteIndex - 1]['tone']  - CHORDOFFSET
      lineSum = np.sum(matrix[matrixIndex])
      if lineSum != 0: #Use normalized row distribution
        probLine = matrix[matrixIndex] / lineSum
        tone = np.random.choice(1728, p = probLine) + CHORDOFFSET
      else: #Use uniform distribution
        tone = np.random.choice(1728) + CHORDOFFSET				

		#Get random chord duration from 4 to 15 (at least quarter note)
    duration = np.random.randint(4, 16)

  else : #melodic line
    if noteIndex < 2 : #Pick random notes weighted by music theoretic ideas
      tone = np.random.choice(12, [0.5, 0, 0.1, 0.1, 0.1, 0.2, 0, 0, 0, 0, 0, 0])
      duration = np.random.randint(0, 16)

    else: #Get note based on previous notes
  		#Get matrix line
      matrixIndex = ((deviceMusic[noteIndex - 1]['tone'] * NUMDUR + 
        deviceMusic[noteIndex - 1]['duration']) * NUMNOTES) + (
        deviceMusic[noteIndex - 2]['tone'] * NUMDUR + deviceMusic[noteIndex - 2]['duration'])
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

  return (tone, duration)

#Creates and stores a total of NUMMEASURES of music in parallel,
#split by parts
def generatePart(matrix, partMarker, mood, partMusic):
  numBeatsFilled = 0
  noteIndex = 0

	#Generating notes
  while numBeatsFilled < NUMMEAURES * BEATSPERMEASURE : 
		#Get next note
    newNote = nextNote(noteIndex, partMusic, partMarker, matrix, mood)

    numBeatsFilled = numBeatsFilled + newNote['duration']
    #If too long, chop note off at end of measure
    if (numBeatsFilled > NUMMEAURES * BEATSPERMEASURE) :
      newNote['duration'] = NUMMEAURES * BEATSPERMEASURE - numBeatsFilled + newNote[['duration']] 

		#Add note to music array
    partMusic[noteIndex] = newNote;
    noteIndex = noteIndex + 1
  return

#Calls GPU to generate NUMMEASURES total measures of music
#Returns a 2D array containing the (tone, duration) pairs of the
#music generated, split by part (as assigned by the parts variable)
def generateMusic(highNotes, lowNotes, chords, parts, mood):

  music = np.ndarray((10, NUMMEASURES * BEATSPERMEASURE * 4), dtype = note)

  try:
    for index, partMarker in enumerate(parts):
      t = None
      if (partMarker == -1): #silent
        continue
      elif (partMarker == 0): #chord
        t = threading.Thread(target = generatePart, args = (chords, mood, music[index],))
      elif (partMarker == 1): #bass
        t = threading.Thread(target = generatePart, arg = (lowNotes, mood, music[index],))
      elif (partMark == 2): #soprano
        t = threading.Thread(target = generatePart, arg = (highNotes, mood, music[index],))
      else: #Error
        print ('Error: Part assignment not allowed')
      t.start()
  except:
    print ('Error: unable to start thread')

  for t in threading.enumerate():
    t.join()

  return music
