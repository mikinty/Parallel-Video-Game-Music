'''
This file serves as the server endpoint for the music generation.

It handles requests from the client and sends back generated music
using the Markov Model created during the training process.
'''

import asyncio
import datetime
import random
import websockets

# @asyncio.coroutine
async def time(websocket, path):
  while True:
    now = datetime.datetime.utcnow().isoformat() + 'Z'
    await websocket.send(now)
    await asyncio.sleep(random.random() * 3)

start_server = websockets.serve(time, '0.0.0.0', 80)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
