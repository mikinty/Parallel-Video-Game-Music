import numpy as np
import pandas as pd
import musicGenParallel as mgp
import asyncio
import websockets
import json
from time import time

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

MATRICES = {}

#Loaded matrices
MAJORHIGH = None # np.loadtxt('majorHighMatrix.txt', dtype = np.int32)
MAJORLOW = None # np.loadtxt('majorLowMatrix.txt', dtype = np.int32)
MAJORCHORD = None # np.loadtxt('majorChordMatrix.txt', dtype = np.int32)
MINORHIGH = None #np.loadtxt('minorHighMatrix.txt', dtype = np.int32)
MINORLOW = None #np.loadtxt('minorLowMatrix.txt', dtype = np.int32)
MINORCHORD = None #np.loadtxt('minorChordMatrix.txt', dtype = np.int32)

# remember what number requests are
transactionID = 0

#Global Variables - Matrices
highNotes = MAJORHIGH
lowNotes = MAJORLOW
chords = MAJORCHORD

#Array of parts, where 0 = chord, 1 = bass, 2 = soprano, -1 = silent
parts = np.array([0, 0, 1, 1, 2, 2, -1, -1, -1, -1])

#Keeps track of major (0) /minor (1)
mood = 0 

def loadMajH(q):
  q.put([MAJOR_HIGH_FILE, pd.read_csv(MAJOR_HIGH_FILE, sep =' ', header=None)])
  #q.put(['a', 1])
  print(1)
def loadMajL(q):
  q.put([MAJOR_LOW_FILE, pd.read_csv(MAJOR_LOW_FILE, sep =' ', header=None)])
  #q.put(['b', 2])
  print(2)
def loadMajK(q):
  q.put([MAJOR_CHORD_FILE, pd.read_csv(MAJOR_CHORD_FILE, sep =' ', header=None)])
  print(3)
def loadMinH(q):
  q.put([MINOR_HIGH_FILE, pd.read_csv(MINOR_HIGH_FILE, sep =' ', header=None)])
  #q.put(['c', 3])
  print(4)
def loadMinL(q):
  q.put([MINOR_LOW_FILE, pd.read_csv(MINOR_LOW_FILE, sep =' ', header=None)])
  #q.put(['d', 4])
  print(5)
def loadMinK(q):
  q.put([MINOR_CHORD_FILE, pd.read_csv(MINOR_CHORD_FILE, sep =' ', header=None)])
  print(6)

# Runs functions in parallel using Process
# Adapted from https://stackoverflow.com/questions/7207309/python-how-can-i-run-python-functions-in-parallel
def loadInParallel(*fns):
  global MATRICES 

  proc = []
  q = Queue()
  for fn in fns:
    p = Process(target=fn, args=(q,))
    p.start()
    proc.append(p)

  for x in range(NUM_MATRICES):
    temp = q.get()
    MATRICES[temp[0]] = temp[1]
    print('done', x)

  for p in proc:
    p.join()

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

  #except:
  #  print('SERVER ERROR')

def check():
  global MAJORHIGH
  print(MAJORHIGH)

a = time()
print('Initializing matrices', a)
loadInParallel(loadMajH, loadMajL, loadMajK, loadMinH, loadMinL, loadMinK)
print('Time:', time() - a)
check()

#start_server = websockets.serve(main, '0.0.0.0', 80)

#asyncio.get_event_loop().run_until_complete(start_server)
#asyncio.get_event_loop().run_forever()
