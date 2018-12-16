import numpy as np
import pandas as pd
import musicGenParallel as mgp
import asyncio
import websockets
import json
from pickle import load, dump
from time import time
import math

from multiprocessing import Process, Queue

from CONST_SERVER import *

'''
Settings Client Picks:
Note that certain settings do not affect (note, duration) choices,
and so do not need to be known by server. The client, after
receiving measures of music, can then implement tempo and instrumentation
on its side:

	- Mood (Major/Minor): Picked at start screen, needed by server
	- Parts Split: Picked at mixing screen, used by server if not default
	- Tempo: Picked at mixing screen, only used by client
	- Instrumentation: Picked at mixing screen, only used by client
	- Tonic: Picked at mixing screen, only used by client
	- Time Signature: Currently 4/4, could be picked at start screen and sent to server
'''


#Loaded matrices
MAJORHIGH = None 
MAJORLOW = None 
MAJORCHORD = None 
MINORHIGH = None 
MINORLOW = None 
MINORCHORD = None 

# remember what number requests are
transactionID = 0

#Global Variables - Matrices
highNotes = None 
lowNotes = None
chords = None

#Array of parts, where 0 = chord, 1 = bass, 2 = soprano, -1 = silent
parts = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 2])

#Keeps track of major (0) /minor (1)
mood = 0 

# Runs functions in parallel using Process
# Adapted from https://stackoverflow.com/questions/7207309/python-how-can-i-run-python-functions-in-parallel
def loadMatrices():
  global MAJORHIGH  
  global MAJORLOW 
  global MAJORCHORD 
  global MINORHIGH 
  global MINORLOW 
  global MINORCHORD
  global highNotes
  global lowNotes
  global chords

  MAJORHIGH = load(open(MAJOR_HIGH_FILE, 'rb'))
  MAJORLOW = load(open(MAJOR_LOW_FILE, 'rb'))
  MAJORCHORD = load(open(MAJOR_CHORD_FILE, 'rb'))
  '''
  MINORHIGH = load(open(MINOR_HIGH_FILE, 'rb'))
  MINORLOW = load(open(MINOR_LOW_FILE, 'rb'))
  MINORCHORD = load(open(MINOR_CHORD_FILE, 'rb'))
  '''

  # default safety
  highNotes = MAJORHIGH
  lowNotes = MAJORLOW
  chords = MAJORCHORD

  print('Done loading all matrices')

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

  async for message in websocket:
    data = json.loads(message)

    if data['request'] == 'START_MUSIC':
      print('Received new music request', transactionID)
      transactionID += 1
      
      try:
        await websocket.send(getNotes())
      except:
        print('Error getting notes')
    elif data['request'] == 'SET_MAJOR':
      mood = 0
      highNotes = MAJORHIGH
      lowNotes = MAJORLOW
      chords = MAJORCHORD
    elif data['request'] == 'SET_MINOR':
      mood = 1
      highNotes = MINORHIGH
      lowNotes = MINORLOW
      chords = MINORCHORD
    elif data['request'] == 'SET_PARTS':
    	#Set the parts array to the array given by client
    	parts = np.array(data['info']); 
    else:
      print('Unknown request')


a = time()
getNotes()
print('Time:', time() - a, 'to get notes for', NUMMEASURES, 'measures')

#start_server = websockets.serve(main, '0.0.0.0', 80)

#asyncio.get_event_loop().run_until_complete(start_server)
#asyncio.get_event_loop().run_forever()
