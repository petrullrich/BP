# author: Petr Ullrich
# VUT FIT - BP
import  ChallengeClass
import ChannelDataClass

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

fPath = 'data/json/'
fFilename = '1548110507_features.json'

# nastaveni elektrod, ze kterych se zpracuji data
# musi byt list i v pripade jednoho channelu
channels = [1]

EEGdata = ChannelDataClass.channelData(channels, fFilename, fPath)
featuresTimestamps = EEGdata.timestamps

EEGdata.removeDcOffset() #filtr
EEGdata.removeMainInterference() #filtr

lPath = 'data/labels/'
lFilename = '1548110507_labels.json'

EEGlabels = ChallengeClass.challange(lFilename, lPath)

# vybrani, kterou sadu challengi chceme trenovat:
# prvni: delani cinnosti (do_it)
# druha: mysleni s otevrenyma ocima (think_open)
# treti: mysleni se zavrenyma ocima (think_closed)
challengeSet = 'think_open'

if(challengeSet == 'think_closed'):
    challengeNumbers = ['3', '6'] # v challengeNumbers jsou cisla, ktera odpovidaji prave a leve ruce pro danou sadu

elif(challengeSet == 'think_open'):
    challengeNumbers = ['2', '5']
elif(challengeSet == 'do_it'):
    challengeNumbers = ['1', '4']

challengeTypeBinary = [0,0,0]

for ch in challengeNumbers:

    if ch == challengeNumbers[0]:
        challengeTypeBinary[0] = 0
    else:
        challengeTypeBinary[0] = 1

    for i in range(1,4):

        if i == 1:
            challengeTypeBinary[1] = 0
            challengeTypeBinary[2] = 0
        elif i == 2:
            challengeTypeBinary[1] = 0
            challengeTypeBinary[2] = 1
        else:
            challengeTypeBinary[1] = 1
            challengeTypeBinary[2] = 0

        challengeType = ch+str(i)
        print('____________________________________________')
        print('ChallengeType: ', challengeType)
        print('ChallengeTypeBinary: ', challengeTypeBinary)
        challengesAttributes = EEGlabels.get_challange(challengeType, featuresTimestamps)  # ziskani ofsetu a delky dane challenge

        EEGdata.processData(challengesAttributes, challengeTypeBinary, len(channels))  # zpracovani dat pro danou challenge

#print("labelsForNN(before numpy array): ", EEGdata.labelsForNN)
#print('len in labelsForNN: ', len(EEGdata.labelsForNN))
EEGdata.dataForNN = np.array(EEGdata.dataForNN)
#for arr in EEGdata.dataForNN:
    #print(len(arr))
EEGdata.labelsForNN = np.array(EEGdata.labelsForNN)
print("dataForNN: ", EEGdata.dataForNN)
print("labelsForNN: ", EEGdata.labelsForNN)
#keras model
model = Sequential()
model.add(Dense(256, activation='sigmoid', input_shape=(64*len(channels),)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(EEGdata.dataForNN, EEGdata.labelsForNN, epochs=20, batch_size=1024, shuffle=True)
