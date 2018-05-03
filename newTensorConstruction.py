import numpy
import os
np = numpy.zeros(shape=(10, 10))
globalDict = {}
globalCounter = 0
dataDir = '/Users/Phaniteja/desktop/RA/NewTensor/'
text_files = [f for f in os.listdir(dataDir) if f.endswith('.txt')]
for filename in text_files:
    if str(filename)[0]=='m':
        flag=1
    else:
        flag=-1
    with open(dataDir+filename) as f:
        for line in f:
            st = str(line)
            li = st.split("-")
            if len(li[1])!=1:
                li[1]=li[1][0]
            if li[0] not in globalDict:
                globalDict[li[0]]=globalCounter
                globalCounter=globalCounter+1
                np[globalCounter-1][int(li[1])]=np[globalCounter-1][int(li[1])]+flag
                print(li[0]+"---i="+str(globalCounter-1)+" j="+str(li[1])+" Value="+str(flag))
            else:
                np[globalDict[li[0]]][int(li[1])]=np[globalCounter-1][int(li[1])]+flag
                print(li[0]+"---i="+str(globalCounter-1)+" j="+str(li[1])+" Value="+str(flag))


print(np)
