# author: Petr Ullrich
# VUT FIT - BP
import  ChallengeClass
import ChannelDataClass

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation


def set_all_data(dataType, challengeSet):

    if (challengeSet == 'think_closed'):
        challengeNumbers = ['3',
                            '6']  # v challengeNumbers jsou cisla, ktera odpovidaji prave a leve ruce pro danou sadu

    elif (challengeSet == 'think_open'):
        challengeNumbers = ['2', '5']
    elif (challengeSet == 'do_it'):
        challengeNumbers = ['1', '4']

    challengeTypeBinary = [0, 0, 0]

    # format labelu pouzitych jako vstup do neuronove site
    # [x,y,z], x == 0 -> leva ruka, x == 1 -> prava ruka
    # yz == 00 -> stage 1 (priprava)
    # yz == 01 -> stage 2 (provadeni dane cinnosti)
    # yz == 10 -> stage 3 (odpocinek)

    # format labelu pouzitych jako vstup do neuronove site
    # [x,y,z]
    # x == 1 && y==0 && z==0 -> leva ruka
    # x == 0 && y==1 && z==0 -> prava ruka
    # x == 0 && y==0 && z==1 -> nic nedelani (priprava + odpocinek)

    left = False
    right = False

    for ch in challengeNumbers:
        # leva ruka
        if ch == challengeNumbers[0]:
            left = True
            right = False

        # prava ruka
        else:
            left = False
            right = True


        for i in range(1, 4):
            # stage 1 - priprava
            if i == 1:
                challengeTypeBinary[0] = 0
                challengeTypeBinary[1] = 0
                challengeTypeBinary[2] = 1
            # stage 2 - samotna challenge
            elif i == 2:

                challengeTypeBinary[2] = 0

                if left:
                    challengeTypeBinary[0] = 1
                    challengeTypeBinary[1] = 0
                elif right:
                    challengeTypeBinary[0] = 0
                    challengeTypeBinary[1] = 1
            # stage 3 - pauza
            else:
                challengeTypeBinary[0] = 0
                challengeTypeBinary[1] = 0
                challengeTypeBinary[2] = 1

            challengeType = ch + str(i)
            print('____________________________________________')
            print('ChallengeType: ', challengeType)
            print('ChallengeTypeBinary: ', challengeTypeBinary)
            challengesAttributes = EEGlabels.get_challange(challengeType,
                                                           featuresTimestamps)  # ziskani ofsetu a delky dane challenge

            EEGdata.processData(challengesAttributes, challengeTypeBinary,
                                len(channels), dataType)  # zpracovani dat pro danou challenge

#-------------------------------------------------------------------------------
# MAIN

# nastaveni elektrod, ze kterych se zpracuji data
# musi byt list i v pripade jednoho channelu
channels = [1,2,3,4,5,6,7,8]

# cesta k features
fPath = 'data/dataZdenek/test/'
# nazev souboru s features
fFilenames = []
fFilenames.append('0feat')
fFilenames.append('1feat')
fFilenames.append('2feat')
EEGdata =[]
featuresTimestamps = []

# instance tridy channelDataClass
for index, fFilename in enumerate(fFilenames):

    EEGdata[index] = ChannelDataClass.channelData(channels, fFilenames[index], fPath)
# upraveni souboru s features od Zdenka
    EEGdata[index].repairFeatures()
# nacteni souboru s features a dpolneni tridni promenne data
    EEGdata[index].load_json_features()

    featuresTimestamps[index] = EEGdata[index].timestamps

    EEGdata[index].removeDcOffset()  # filtr
    EEGdata[index].removeMainInterference()  # filtr

#----------------------------------------------------
#                     LABELS
#----------------------------------------------------


lPath = 'data/dataZdenek/test/'
lFilename = '0lab'

# instance tridy challengeClass
EEGlabels = ChallengeClass.challange(lFilename, lPath)
# upraveni souboru s labely od Zdenka
#EEGlabels.repairLabels()
# nacteni souboru s labely a doplneni tridni promenne chalenges
EEGlabels.load_json_labels()

# vybrani, kterou sadu challengi chceme trenovat:
# prvni: delani cinnosti (do_it)
# druha: mysleni s otevrenyma ocima (think_open)
# treti: mysleni se zavrenyma ocima (think_closed)
challengeSet = 'think_closed'
# arg1: typ dat -
#   training - trenovaci
#   testing - testovaci

set_all_data('training', challengeSet)
#set_all_data('testing', challengeSet)


#print("labelsForNN(before numpy array): ", EEGdata.labelsForNN)
#print('len in labelsForNN: ', len(EEGdata.labelsForNN))
EEGdata.dataForNN = np.array(EEGdata.dataForNN)
EEGdata.testingDataForNN = np.array(EEGdata.testingDataForNN)
#for arr in EEGdata.dataForNN:
    #print(len(arr))
EEGdata.labelsForNN = np.array(EEGdata.labelsForNN)
EEGdata.testingLabelsForNN = np.array(EEGdata.testingLabelsForNN)
print("dataForNN: ", EEGdata.dataForNN)
#print("first position in dataForNN LEN: ", len(EEGdata.dataForNN[0]))
#print("Testing data: ", EEGdata.testingDataForNN)
#print("labelsForNN: ", EEGdata.labelsForNN)

# _______________________________________
# Zkusebni NN



# _______________________________________________________
# keras model

#np.random.seed(7)

model = Sequential()
model.add(Dense(64*len(channels), activation='sigmoid', input_shape=(64*len(channels),)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(EEGdata.dataForNN, EEGdata.labelsForNN, epochs=50, batch_size=10)
acc = model.evaluate(EEGdata.dataForNN, EEGdata.labelsForNN)
pred = model.predict(EEGdata.dataForNN)

# ________________________________
# vyhodnoceni predikci
print("predictions: ", pred)
print("len of predict", len(pred))
print(model.metrics_names[1], acc[1])
print(model.metrics_names[0], acc[0])
#print("loss", loss)
