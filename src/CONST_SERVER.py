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
