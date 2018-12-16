import numpy as np
import musicGenSeq as mgp
import asyncio
import websockets
import json
from pickle import load, dump
from time import time
import math

from CONST_SERVER import *

'''
Settings Client Picks:
	- Mood (Major/Minor): Picked at start screen, needed by server
	- Parts Split: Picked at mixing screen, used by server if not default

Note that certain settings do not affect (note, duration) choices
during generation, and so do not need to be sent to the server. 
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

def loadMatrices():
  global MAJORHIGH  
  global MAJORLOW 
  global MAJORCHORD 
  global MINORHIGH 
  global MINORLOW 
  global MINORCHORD

  MAJORHIGH = load(open(MAJOR_HIGH_FILE, 'rb'))
  MAJORLOW = load(open(MAJOR_LOW_FILE, 'rb'))
  MAJORCHORD = load(open(MAJOR_CHORD_FILE, 'rb'))
  MINORHIGH = load(open(MINOR_HIGH_FILE, 'rb'))
  MINORLOW = load(open(MINOR_LOW_FILE, 'rb'))
  MINORCHORD = load(open(MINOR_CHORD_FILE, 'rb'))

  print('Done loading all matrices')

def getNotes(genre, mood, voices, numMeasures):
  '''
  Returns the generated notes.
  returns: json of notes and corresponding transaction ID
  '''

  # Array of music generated, assuming 4/4 time signature
	# 2D array of (note, duration) pairs split by parts (up to 10)
  music = mgp.generateMusic(genre, mood, voices, numMeasures, MAJORHIGH, MAJORLOW, MAJORCHORD, MINORHIGH, MINORLOW, MINORCHORD)

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

    # parse request
    genre = data['name']
    mood = data['mood']
    voices = data['voices']
    numMeasures = data['measures']

    try:
      await websocket.send(getNotes(genre, mood, voices, numMeasures))

    # except:
    # print('SERVER ERROR')


print('Initializing matrices, will take about 30 seconds')
a = time()
loadMatrices()
print('Time:', time() - a, 'to initialize matrices')


### SERVER INIT ###
start_server = websockets.serve(main, '0.0.0.0', 80)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
