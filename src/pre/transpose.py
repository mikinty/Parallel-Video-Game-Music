# Adapted from https://gist.github.com/aldous-rey/68c6c43450517aa47474
# Converts everything into the key of C major/A minor
# Converts all MIDI files in the directory PATH
# Notice that this script WILL overwrite all the original files

import glob
import os
import music21

# Where MIDI files are located
PATH = dir_path = os.path.dirname(os.path.realpath(__file__)) + '\MIDI'
os.chdir(PATH)

# Key conversions
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

# Convert files
for f in glob.glob("**\*.mid", recursive=True):
  print('Converting', f)

  score = music21.converter.parse(f)
  key = score.analyze('key')

  if key.mode == "major":
    halfSteps = majors[key.tonic.name]
      
  elif key.mode == "minor":
     halfSteps = minors[key.tonic.name]
  
  newscore = score.transpose(halfSteps)
  key = newscore.analyze('key')

  newFileName = f
  newscore.write('midi', newFileName)

print("Done converting files")