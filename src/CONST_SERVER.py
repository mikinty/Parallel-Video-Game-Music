'''
Constants for the server code
'''

# Music settings
NUMMEASURES = 8 
BEATSPERMEASURE = 16 #Using 16th notes
NUMPARTS = 10
NUMTONES = 73
NUMDUR = 15
CHORDOFFSET = 101
NUMOCTAVES = 6
NUMCHORDS = 1728
NUMNOTES = 1095

# Labels
SETTINGS_BASIC = 'basic'
SETTINGS_JOURNEY = 'journey'
SETTINGS_SORROW = 'sorrow'
SETTINGS_BATTLE = 'battle'

# Load 
FOLDER = ''
MAJOR_HIGH_FILE = FOLDER +'majorHighMatrix.pkl'
MAJOR_LOW_FILE = FOLDER + 'majorLowMatrix.pkl'
MAJOR_CHORD_FILE = FOLDER + 'majorChordMatrix.pkl'
MINOR_HIGH_FILE = FOLDER +'minorHighMatrix.pkl'
MINOR_LOW_FILE = FOLDER + 'minorLowMatrix.pkl'
MINOR_CHORD_FILE = FOLDER + 'minorChordMatrix.pkl'

NUM_MATRICES = 6

# Note durations that we support
NOTE_DURATIONS = [
  0.083, 
  0.167, 
  0.250, 
  0.333, 
  0.500, 
  0.667, 
  0.750, 
  1.000, 
  1.333, 
  1.500, 
  1.750, 
  2.000, 
  3.000, 
  4.000, 
  8.000
]
