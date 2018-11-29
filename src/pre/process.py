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
for f in glob.glob("TEST\**\*.mid", recursive=True):
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
    # Find most common note
    l = [e.pitch for e in part.notes if e.isNote]
    c = [e.pitches[0] for e in part.notes if e.isChord]
    L = l+c

    # somehow have no elements in this part...continue
    if len(L) == 0:
      continue

    m = max(set(L), key=L.count)

    print(START_CODE, file=fo)

    # case on the most common pitch
    if m < music21.pitch.Pitch(MIDDLENOTE):
      print(BASS_MARK, file=fo)
    else:
      print(SOPRANO_MARK, file=fo)

    # prints out the notes
    for e in part.notesAndRests:
      d = float(e.duration.quarterLength)
      
      # identify the closest note length according to our defined durations
      d = NOTE_DURATIONS.index(min(NOTE_DURATIONS, key=lambda x:abs(x-d)))
      
      # CHORD. Encoded in base 12.
      if e.isChord:
        tempLen = min(3, len(e.pitches))
        
        n = 0
        
        notes = []
        
        for i in range(0, tempLen):
          notes.append(int(e.pitches[i].ps))
            
        # sort from lowest to highest
        notes.sort()
        
        for i in range(0, tempLen):
          n += (notes[i] % OCTAVE_RANGE) * pow(OCTAVE_RANGE, i)
    
        # we MUST offset in order to account for notes and rests
        n += CHORD_OFFSET
          
      # NOTE. Encoded from 0 (c0) to 95 (b7)
      elif e.isNote:
        n = int(e.pitch.ps) - LOWEST_NOTE_OFFSET
        
        if n < 0:
          n = n % NOTE_RANGE
        elif n >= NOTE_RANGE:
          n = n % OCTAVE_RANGE + HIGHEST_C
          
      # REST. Just a special rest marker
      else:
        n = REST_NUM
          
      # Always print out in "%d %d" format
      print("{0} {1}".format(n, d), file=fo)
    
    # end of part
    print(END_CODE, file=fo)

  # end of file
  print(ENDFILE_CODE, file=fo)
  fo.close()

  print('Saved', newFileName)

print("Done converting files")