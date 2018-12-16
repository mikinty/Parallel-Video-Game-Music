# Combine all the text files together into two matrices

from glob import glob
from CONST import OUTPUT_FILE_MAJOR, OUTPUT_FILE_MINOR

files = glob('MIDI/*.txt', recursive=True)
totalFiles = len(files)
currIter = 1
errorNum = 0

majorData = []
minorData = []

# Convert all pre-processed MIDIs
with open(OUTPUT_FILE_MAJOR, 'w') as outputMajor, open(OUTPUT_FILE_MINOR, 'w') as outputMinor:

  for currFile in files:
    print('Merging', currIter, 'out of', totalFiles)

    currIter += 1

    mood = ''

    try:
      fRead = open(currFile, 'r')
      mood = fRead.readline()

      if mood == 'major\n':
        for l in fRead:
          outputMajor.write(l)
      elif mood == 'minor\n':
        for l in fRead:
          outputMinor.write(l)
      else:
        errorNum += 1
        print('Skipped', currFile, 'because incomplete')
        continue
      
      fRead.close()
    except:
      errorNum += 1
      print('Error reading', currFile)
      continue


print('Done combining all', totalFiles, 'files, with', errorNum, 'errors')
