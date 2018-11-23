# Adapted from:
# - https://gist.github.com/aldous-rey/68c6c43450517aa47474
#
# Process:
# - Converts MIDI files in the key of C major/A minor
# - Prints out the text file representation of the MIDI, containing:
#   - Header with song info
#   - a music note with its duration on each line

from CONST import *
import music21
import glob

# Convert files
for f in glob.glob("TEST\*.mid", recursive=True):
  print('Converting', f)

  # first, transpose the entire score into C major or A minor
  score = music21.converter.parse(f)
  key = score.analyze('key')
  mode = key.mode

  if mode == "major":
    halfSteps = majors[key.tonic.name]
      
  elif mode == "minor":
     halfSteps = minors[key.tonic.name]
  
  newscore = score.transpose(halfSteps)

  ### Now, print out score in our text format ###

  # Major/Minor
  print(mode)

  # Print parts
  for part in score.parts:
    print(STARTCODE)
    print(str(part.getInstrument()))

    # Find most common note
    l = [e.pitch for e in part.notes]
    m = max(set(l), key=l.count)

    # case on the most common pitch
    music21.pitch.Pitch('C#5') < m

    print('Soprano')

  # prints out the notes
  for e in part.notesAndRests:
    if e.isNote:
        print(e.pitch, end=', ')
    else:
        print('R', end=', ')
    print("{:0.3f}".format(float(e.duration.quarterLength)))

  print(ENDCODE)
  
  
  newFileName = f[:-4] + '.txt'
  print('Saving in', newFileName)

print("Done converting files")