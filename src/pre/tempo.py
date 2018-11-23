# Adapted from https://gist.github.com/aldous-rey/68c6c43450517aa47474
# Converts everything to have a quarter note of X time
# We want to normalize note transitions so that our learning algorithm
# can have easier data to work with. 

import glob
import os
import music21

# Where MIDI files are located
PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MIDI')
os.chdir(PATH)

# Constant
QUARTERNOTE = 1

f='Basic/ff1menu.mid'

score = music21.converter.parse(f)

