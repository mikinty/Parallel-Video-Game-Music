import numpy as np
import musicGenParallel as mgp

"""Settings Client Picks:
Note that certain settings do not affect (note, duration) choices,
and so do not need to be known by server. The client, after
receiving measures of music, can then implement tempo and instrumentation
on its side
	Mood (Major/Minor) - Picked at start screen, needed by server
	Parts Split - Picked at mixing screen, used by server if not default
	Tempo - Picked at mixing screen, only used by client
	Instrumentation - Picked at mixing screen, only used by client
	Tonic? - Currently random, could be picked at start screen and sent to server
	Time Signature - Currently 4/4, could be picked at start screen and sent to server
"""

#File names for matrices
MAJORHIGH = 'majorHighMatrix.txt'
MAJORLOW = 'majorLowMatrix.txt'
MAJORCHORD = 'majorChordMatrix.txt'
MINORHIGH = 'minorHighMatrix.txt'
MINORLOW = 'minorLowMatrix.txt'
MINORCHORD = 'minorChordMatrix.txt'

#Music settings
NUMMEASURES = 4
BEATSPERMEASURE = 4

#Sets up and establishes link between server and client
def initServerLink():
	return

#Waits for the client to send over choice of major or minor
#Returns either 0 (minor) or 1 (major)
def receiveMajorMinor():
	return 1

#Sends message to client noting an error has occured, and to close link + retry
def sendErrorMessage():
	return

#Closes link between server and client
def closeServerLink():
	return

#Determines if the client wants to stop music
#Returns either true or false
def endMusic():
	return true

#Receives information about the number of soprano, bass, and chord
#voices wanted and sets that up
#Returns array representing the parts wanted
def receivePartsSplit():
	return parts

#Send NUMMEASURES total measures of music to client, stored in music
def sendMusic():
	return

#Array of music generated, assuming 4/4 time signature
#2D array of (note, duration) pairs split by parts (up to 10)
music = None

#Global Variables - Matrices
highNotes = None
lowNotes = None
chords = None

#Array of parts, where 0 = chord, 1 = bass, 2 = soprano, -1 = silent
parts = np.array([0, 0, 1, 1, 2, 2, -1, -1, -1, -1])

#Pick a random tonic note 0 = C, 11 = B
tonic = random.randint(0, 11)

#Establish Server-Client link
initServerLink()

#Wait to receive major/minor choice from client
mood = receiveMajorMinor() #0 = minor, 1 = major

#Reads in chord, high melody, and low melody matrices of the proper mood
if mood == 0: #minor
	highNotes = np.loadtxt(minorHighMatrix, dtype = np.float64, delminter = ' ')
	lowNotes = np.loadtxt(minorLowMatrix, dtype = np.float64, delminter = ' ')
	chords = np.loadtxt(minorChordMatrix, dtype = np.float64, delminter = ' ')
else if mood == 1: #major
	highNotes = np.loadtxt(majorHighMatrix, dtype = np.float64, delminter = ' ')
	lowNotes = np.loadtxt(majorLowMatrix, dtype = np.float64, delminter = ' ')
	chords = np.loadtxt(majorChordMatrix, dtype = np.float64, delminter = ' ')
else: #error - not a valid mood
	sendErrorMessage()
	closeServerLink()
	return

while(true)
	#If client says to stop music, we stop and close link
	if endMusic():
		closeServerLink()
		return

	#If client says to change parts split, modify parts array
	receivePartsSplit()

	#Calls GPUs to generate measures of music
	music = mgp.generateMusic(highNotes, lowNotes, chords, parts, tonic, mood)

	#Sends music messages
	sendMusic()

