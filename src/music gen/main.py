import numpy as np
import musicGenParallel as mgp
import asyncio
import websockets
import json

"""Settings Client Picks:
Note that certain settings do not affect (note, duration) choices,
and so do not need to be known by server. The client, after
receiving measures of music, can then implement tempo and instrumentation
on its side
	Mood (Major/Minor) - Picked at start screen, needed by server
	Parts Split - Picked at mixing screen, used by server if not default
	Tempo - Picked at mixing screen, only used by client
	Instrumentation - Picked at mixing screen, only used by client
	Tonic - Picked at mixing screen, only used by client
	Time Signature - Currently 4/4, could be picked at start screen and sent to server
"""

#File names for matrices
MAJORHIGH = 'majorHighMatrix.txt'
MAJORLOW = 'majorLowMatrix.txt'
MAJORCHORD = 'majorChordMatrix.txt'
MINORHIGH = 'minorHighMatrix.txt'
MINORLOW = 'minorLowMatrix.txt'
MINORCHORD = 'minorChordMatrix.txt'

# remember what number requests are
transactionID = 0

#Global Variables - Matrices
highNotes = None
lowNotes = None
chords = None

#Array of parts, where 0 = chord, 1 = bass, 2 = soprano, -1 = silent
parts = np.array([0, 0, 1, 1, 2, 2, -1, -1, -1, -1])

#Keeps track of major (0) /minor (1)
mood = 0 

#Random tonic note to start on

#Receives information about the number of soprano, bass, and chord
#voices wanted and sets that up
#Returns array representing the parts wanted
def receivePartsSplit():
	return

def getNotes():
  '''
  Returns the generated notes.
  returns: json of notes and corresponding transaction ID
  '''
  #Array of music generated, assuming 4/4 time signature
	#2D array of (note, duration) pairs split by parts (up to 10)
	#Calls GPUs to generate measures of music
	music = mgp.generateMusic(highNotes, lowNotes, chords, parts, mood)

	return json.dumps({'id': transactionID, 'notes': music})

async def main(websocket, path):
  '''
  Client-facing function that:
    1. Handles client music requests
    2. Updates Transaction ID
    3. Triggers music generation
    4. Sends back generated music to client
  '''
  
  global transactionID

  # TODO: use try catches for more robust server, but for
  # now we need to seee what the actual errors are...
  # try:
  async for message in websocket:
    data = json.loads(message)

    if data['request'] == 'START_MUSIC':
      print('Received new music request', transactionID)
      transactionID += 1
      
      try:
        await websocket.send(getNotes())
      except:
        print('Error getting notes')
    else if data['request'] == 'SET_MAJOR':
    	mood = 0
    	highNotes = np.loadtxt(majorHighMatrix, dtype = np.int32, delminter = ' ')
			lowNotes = np.loadtxt(majorLowMatrix, dtype = np.int32, delminter = ' ')
			chords = np.loadtxt(majorChordMatrix, dtype = np.int32, delminter = ' ')
    else if data['request'] == 'SET_MINOR':
    	mood = 1
    	highNotes = np.loadtxt(minorHighMatrix, dtype = np.int32, delminter = ' ')
			lowNotes = np.loadtxt(minorLowMatrix, dtype = np.int32, delminter = ' ')
			chords = np.loadtxt(minorChordMatrix, dtype = np.int32, delminter = ' ')
    else if data['request'] == 'SET_PARTS':
    	#Set the parts array to the array given by client
    	parts = np.array(data['info']); 
    else:
      print('Unknown request')

  #except:
  #  print('SERVER ERROR')

start_server = websockets.serve(main, '0.0.0.0', 80)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()