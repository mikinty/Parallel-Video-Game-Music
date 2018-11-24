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

# Convert files in directory
for f in glob.glob("TEST\*.mid", recursive=True):
  print('Converting', f)

  # output file
  newFileName = f[:-4] + '.txt'

  fo = open(newFileName, 'w')  

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
  print(mode, file=fo)

  # Print parts
  for part in score.parts:
    print(STARTCODE, file=fo)
    print(str(part.getInstrument()), file=fo)

    # Find most common note
    l = [e.pitch for e in part.notes if e.isNote]
    m = max(set(l), key=l.count)

    # case on the most common pitch
    if m < music21.pitch.Pitch(MIDDLENOTE):
      print(BASSMARK, file=fo)
    else:
      print(SOPRANOMARK, file=fo)

    # prints out the notes
    for e in part.notesAndRests:
      d = "{:0.3f}".format(float(e.duration.quarterLength))
      if e.isChord:
        for p in e.pitches:
          print(CHORDMARK, str(p), d, file=fo)
      if e.isNote:
          print(e.pitch, d, file=fo)
      else:
          print(RESTMARK, d, file=fo)

    print(ENDCODE, file=fo)
  
  
  print('Saved', newFileName)

print("Done converting files")