# Constants for process.py

START_CODE = 'S'
END_CODE = 'E'
ENDFILE_CODE = 'X'

LOWEST_NOTE_OFFSET = 24
NOTE_RANGE = 72 # number of distinct notes defined
OCTAVE_RANGE = 12 # number of half-steps in an octave
HIGHEST_C_RAW = 84 # C6
HIGHEST_C = HIGHEST_C_RAW - LOWEST_NOTE_OFFSET

SOPRANO_MARK = 'H' # high
BASS_MARK = 'L' # low

CHORD_OFFSET = 101
REST_NUM = 72

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

# Key conversions from C
majors = dict([
  ("A-", 4),
  ("A", 3),
  ("A#", 2),
  ("B-", 2),
  ("B", 1),
  ("B#", 0),
  ("C-", 1),
  ("C", 0),
  ("C#", -1),
  ("D-", -1),
  ("D", -2),
  ("D#", -3),
  ("E-", -3),
  ("E", -4),
  ("E#", -5),
  ("F-", -4),
  ("F", -5),
  ("F#", 6),
  ("G-", 6),
  ("G", 5),
  ("G#", 4)
])

# Key conversions from A
minors = dict([
  ("A-", 1),
  ("A", 0),
  ("A#", -1),
  ("B-", -1),
  ("B", -2),
  ("B#", -3),
  ("C-", -2),
  ("C", -3),
  ("C#", -4),
  ("D-", -4),
  ("D", -5),
  ("D#", 6),
  ("E-", 6),
  ("E", 5),
  ("E#", 4),
  ("F-", 5),
  ("F", 4),
  ("F#", 3),
  ("G-", 3),
  ("G", 2),
  ("G#", 1)
])

# Soprano, Treble, Tenor, Bass cutoffs
# Currrently only dividing between Soprano and Bass
MIDDLENOTE = 'C4'