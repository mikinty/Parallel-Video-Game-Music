import numpy as np
import asyncio
import websockets
import json
from pickle import load, dump
from time import time
import math

from multiprocessing import Process, Queue

from CONST_SERVER import *

#Loaded matrices
MAJORHIGH = None 
MAJORLOW = None 
MAJORCHORD = None 
MINORHIGH = None 
MINORLOW = None 
MINORCHORD = None 

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
  '''
  MINORHIGH = load(open(MINOR_HIGH_FILE, 'rb'))
  MINORLOW = load(open(MINOR_LOW_FILE, 'rb'))
  MINORCHORD = load(open(MINOR_CHORD_FILE, 'rb'))
  '''

  print('Done loading all matrices')

print('Initializing matrices, will take about 30 seconds')
a = time()
loadMatrices()
print('Time:', time() - a, 'to initialize matrices')
