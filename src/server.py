'''
This file serves as the server endpoint for the music generation.

It handles requests from the client and sends back generated music
using the Markov Model created during the training process.
'''

import asyncio
import websockets
import json

# remember what number requests are
transactionID = 0

def getNotes():
  '''
  Returns the generated notes.

  returns: json of notes and corresponding transaction ID
  '''
  
  # TODO: use REAL generated notes
  testNotes = [
    [32, 4],
    [24, 4],
    [40, 5],
    [44, 5],
    [36, 4]
  ]

  return json.dumps({'id': transactionID, 'notes': testNotes})


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
    else:
      print('Unknown request')

  #except:
  #  print('SERVER ERROR')

start_server = websockets.serve(main, '0.0.0.0', 80)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
