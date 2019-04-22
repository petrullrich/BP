# author: Petr Ullrich
# VUT FIT - BP
import  ChallengeClass
import ChannelDataClass

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation


def set_all_data(dataType, challengeSet, index):

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
            challengesAttributes = EEGlabels[index].get_challange(challengeType,
                                                           featuresTimestamps[index])  # ziskani ofsetu a delky dane challenge

            EEGdata[index].processData(challengesAttributes, challengeTypeBinary,
                                len(channels), dataType)  # zpracovani dat pro danou challenge

#-------------------------------------------------------------------------------
# MAIN

# data spojene z vice nahravani
allDataForNN = []
allLabelsForNN = []
# nastaveni elektrod, ze kterych se zpracuji data
# musi byt list i v pripade jednoho channelu
channels = [1,2,3,4,5,6,7,8]

#----------------------------------------------------
#                    FEATURES
#----------------------------------------------------

# cesta k features
fPath = 'data/dataZdenek/train/'
# nazev souboru s features
fFilenames = []
fFilenames.append('0feat')
fFilenames.append('1feat')
fFilenames.append('2feat')
fFilenames.append('3feat')
fFilenames.append('4feat')
fFilenames.append('5feat')
EEGdata =[]
featuresTimestamps = []

# instance tridy channelDataClass
for index, fFilename in enumerate(fFilenames):
    print("index: ", index)
    print("fFilename: ", fFilename)
    EEGdata.append(ChannelDataClass.channelData(channels, fFilename, fPath))
    # upraveni souboru s features od Zdenka
    #if index == 2:
    #EEGdata[index].repairFeatures()
    # nacteni souboru s features a dpolneni tridni promenne data
    EEGdata[index].load_json_features()

    featuresTimestamps.append(EEGdata[index].timestamps)

    EEGdata[index].removeDcOffset()  # filtr
    EEGdata[index].removeMainInterference()  # filtr


#----------------------------------------------------
#                     LABELS
#----------------------------------------------------

# vybrani, kterou sadu challengi chceme trenovat:
# prvni: delani cinnosti (do_it)
# druha: mysleni s otevrenyma ocima (think_open)
# treti: mysleni se zavrenyma ocima (think_closed)
challengeSet = 'think_closed'
# arg1: typ dat -
#   training - trenovaci
#   testing - testovaci


lPath = 'data/dataZdenek/train/'

lFilenames = []
lFilenames.append('0lab')
lFilenames.append('1lab')
lFilenames.append('2lab')
lFilenames.append('3lab')
lFilenames.append('4lab')
lFilenames.append('5lab')
EEGlabels = []

for index, lFilename in enumerate(lFilenames):
    print("lFilename: ", lFilename)
    # instance tridy challengeClass
    EEGlabels.append(ChallengeClass.challange(lFilename, lPath))
    # upraveni souboru s labely od Zdenka
    #if index == 2:
    #EEGlabels[index].repairLabels()
    # nacteni souboru s labely a doplneni tridni promenne chalenges
    EEGlabels[index].load_json_labels()


    # parametr index urcuje ktera instance tridy se prave zpracovava
    set_all_data('training', challengeSet, index)
    #set_all_data('testing', challengeSet)

    #print("dataForNN: ", EEGdata[index].dataForNN)
    allDataForNN = allDataForNN + EEGdata[index].dataForNN
    allLabelsForNN = allLabelsForNN + EEGdata[index].labelsForNN

print("Len of allDataForNN: ", len(allDataForNN))
allLabelsForNN = np.array(allLabelsForNN)
# print("labelsForNN(before numpy array): ", EEGdata.labelsForNN)
# print('len in labelsForNN: ', len(EEGdata.labelsForNN))
allDataForNN = np.array(allDataForNN)
print("Len of allDataForNN (np array): ", len(allDataForNN))
print("All data for NN: ",allDataForNN)
print("All labels for NN", allLabelsForNN)

allTestDataForNN = allDataForNN
allTestLabelsForNN = allLabelsForNN
# _______________________________________
# Zkusebni NN



# __________________________________________________________________________________________
# keras model

#np.random.seed(7)

model = Sequential()
model.add(Dense(64*len(channels), activation='sigmoid', input_shape=(64*len(channels),)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(allDataForNN, allLabelsForNN, epochs=300, batch_size=1000)
score = model.evaluate(allTestDataForNN, allTestLabelsForNN)
predictions = model.predict(allTestDataForNN)

# ----------------------------------------------
# vyhodnoceni presnosti
print(model.metrics_names[1], score[1])
print(model.metrics_names[0], score[0])

# ----------------------------------------------
# vyhodnoceni predikci

print("predictions: ", predictions)
print("len of predict", len(predictions))

left = 0
right = 0
pause = 0

for label in allTestLabelsForNN:
    if label[0] == 1:
        left += 1
    elif label[1] == 1:
        right += 1
    if label[2] == 1:
        pause += 1

leftCorrect = 0
rightCorrect = 0
pauseCorrect = 0
wrong = 0

for i, pr in enumerate(predictions):
    if pr[0] > pr[1] and pr[0] > pr[2] and allTestLabelsForNN[i][0] == 1:
        # leva spravne
        leftCorrect += 1
    elif pr[1] > pr[0] and pr[1] > pr[2] and allTestLabelsForNN[i][1] == 1:
        # prava spravne
        rightCorrect += 1
    elif pr[2] > pr[1] and pr[2] > pr[1] and allTestLabelsForNN[i][2] == 1:
        # nic nedelani spravne
        pauseCorrect += 1
    else:
        wrong += 1

print("leva: ", leftCorrect, "z", left)
print("prava: ", rightCorrect, "z", right)
print("pause: ", pauseCorrect, "z", pause)
print("wrong: ", wrong)