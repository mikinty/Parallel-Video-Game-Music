

NUM = 1
num = str(NUM)


total = 5000000 

def replicateMinor():
  with open('notesMinor.txt', 'r') as inFile, open('notesMinor'+num+'.txt', 'w') as outFile:
    j = 0
    for x in range(NUM):
      for f in inFile:
        j+=1
        outFile.write(f)

        if j > total:
          break

      inFile.seek(0)

      print(x)

def replicateMajor():
  with open('notesMajor.txt', 'r') as inFile, open('notesMajor'+num+'.txt', 'w') as outFile:
    j = 0
    for x in range(NUM):
      for f in inFile:
        j+=1
        outFile.write(f)
        if j > total:
          break
      inFile.seek(0)

      print(x)

replicateMinor()
replicateMajor()
