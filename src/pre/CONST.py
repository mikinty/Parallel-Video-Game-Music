# Constants for process.py

STARTCODE = 'S'
ENDCODE = 'S'

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
# TODO