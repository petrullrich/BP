# author: Petr Ullrich
# VUT FIT - BP
import  ChallengeClass
import ChannelDataClass

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation


def set_all_data(dataType, challengeSet, index):

    challengeNumbers = ['3','6'] # defaultní hodnota

    if (challengeSet == 'think_closed'):
        challengeNumbers = ['3',
                            '6']  # v challengeNumbers jsou cisla, ktera odpovidaji prave a leve ruce pro danou sadu

    elif (challengeSet == 'think_open'):
        challengeNumbers = ['2', '5']
    elif (challengeSet == 'do_it'):
        challengeNumbers = ['1', '4']

    challengeTypeBinary = [0, 0, 0]

    # format labelu pouzitych jako vstup do neuronove site
    # [x,y,z]
    # x == 1 && y==0 && z==0 -> leva ruka
    # x == 0 && y==1 && z==0 -> prava ruka
    # x == 0 && y==0 && z==1 -> nic nedelani (priprava + odpocinek)

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
            if dataType == 'training':
                challengesAttributes = EEGlabels[index].get_challange(challengeType,
                                                           featuresTimestamps[index])  # ziskani ofsetu a delky dane challenge

                EEGdata[index].processData(challengesAttributes, challengeTypeBinary,
                                           len(channels))  # zpracovani dat pro danou challenge
            elif dataType == 'testing':
                challengesAttributes = EEGTestlabels[index].get_challange(challengeType,
                                                                      featuresTimestamps[
                                                                          index])  # ziskani ofsetu a delky dane challenge

                EEGtestData[index].processData(challengesAttributes, challengeTypeBinary,
                                           len(channels))  # zpracovani dat pro danou challenge
            else:
                print('Zadejte správný typ dat (training/testing)')



#___________________________________________________________________________________________________________
#                                                    MAIN
#___________________________________________________________________________________________________________
# data spojene z vice nahravani
allDataForNN = []
allLabelsForNN = []
allTestDataForNN = []
allTestLabelsForNN = []
# nastaveni elektrod, ze kterych se zpracuji data
# musi byt list i v pripade jednoho channelu
channels = [1,2,3,4,5,6,7,8]

#_________________________________________________________________
# TRAINING

#----------------------------------------------------
#                 FEATURES - TRAINING
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
EEGdata = []
featuresTimestamps = []

# instance tridy channelDataClass
for index, fFilename in enumerate(fFilenames):
    print("index: ", index)
    print("fFilename: ", fFilename)
    EEGdata.append(ChannelDataClass.channelData(channels, fFilename, fPath))
    # upraveni souboru s features od Zdenka
    #EEGdata[index].repairFeatures()
    # nacteni souboru s features a dpolneni tridni promenne data
    EEGdata[index].load_json_features()

    featuresTimestamps.append(EEGdata[index].timestamps)

    EEGdata[index].removeDcOffset()  # filtr
    EEGdata[index].removeMainInterference()  # filtr

#----------------------------------------------------
#                 LABELS - TRAINING
#----------------------------------------------------

# vybrani, kterou sadu challengi chceme trenovat:
# prvni: delani cinnosti (do_it)
# druha: mysleni s otevrenyma ocima (think_open)
# treti: mysleni se zavrenyma ocima (think_closed)
challengeSet = 'do_it'

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
    #EEGlabels[index].repairLabels()
    # nacteni souboru s labely a doplneni tridni promenne chalenges
    EEGlabels[index].load_json_labels()

    # parametr index urcuje ktera instance tridy se prave zpracovava
    set_all_data('training', challengeSet, index)

    #print("dataForNN: ", EEGdata[index].dataForNN)
    allDataForNN = allDataForNN + EEGdata[index].dataForNN
    allLabelsForNN = allLabelsForNN + EEGdata[index].labelsForNN

print("Len of allDataForNN: ", len(allDataForNN))

#_________________________________________________________________
# TESTING

#----------------------------------------------------
#                 FEATURES - TESTING
#----------------------------------------------------

# cesta k features
fTestPath = 'data/dataZdenek/test/'
# nazev souboru s features
fTestFilenames = []
fTestFilenames.append('0feat')
fTestFilenames.append('1feat')
fTestFilenames.append('2feat')
EEGtestData = []
featuresTimestamps = []

# instance tridy channelDataClass
for index, fFilename in enumerate(fTestFilenames):
    print("index: ", index)
    print("fTestFilename: ", fFilename)
    EEGtestData.append(ChannelDataClass.channelData(channels, fFilename, fTestPath))
    # upraveni souboru s features od Zdenka
    # EEGtestData[index].repairFeatures()
    # nacteni souboru s features a dpolneni tridni promenne data
    EEGtestData[index].load_json_features()

    featuresTimestamps.append(EEGtestData[index].timestamps) # pole s timestampy z features

    EEGtestData[index].removeDcOffset()  # filtr
    EEGtestData[index].removeMainInterference()  # filtr

#----------------------------------------------------
#                 LABELS - TESTING
#----------------------------------------------------

lTestPath = 'data/dataZdenek/test/'

lTestFilenames = []
lTestFilenames.append('0lab')
lTestFilenames.append('1lab')
lTestFilenames.append('2lab')
EEGTestlabels = []

for index, lFilename in enumerate(lTestFilenames):
    print("lTestFilename: ", lFilename)
    # instance tridy challengeClass
    EEGTestlabels.append(ChallengeClass.challange(lFilename, lTestPath))
    # upraveni souboru s labely od Zdenka
    #EEGTestlabels[index].repairLabels()
    # nacteni souboru s labely a doplneni tridni promenne chalenges
    EEGTestlabels[index].load_json_labels()

    # parametr index urcuje ktera instance tridy se prave zpracovava
    set_all_data('testing', challengeSet, index)

    #print("dataForNN: ", EEGdata[index].dataForNN)
    allTestDataForNN = allTestDataForNN + EEGtestData[index].dataForNN
    allTestLabelsForNN = allTestLabelsForNN + EEGtestData[index].labelsForNN

print("Len of allTestDataForNN: ", len(allTestDataForNN))

# __________________________________________________________________________________________
# pro vstupni data ->
# odstraneni prebytecnych ramcu - tzn. pro kazdou tridu stejny pocet ramcu

leftTraining = 0
rightTraining = 0
pauseTraining = 0

for label in allLabelsForNN:
    if label[0] == 1:
        leftTraining += 1
    elif label[1] == 1:
        rightTraining += 1
    elif label[2] == 1:
        pauseTraining += 1

print("Before reduce: left: ", leftTraining)
print("Before reduce: right: ", rightTraining)
print("Before reduce: pause: ", pauseTraining)

# odebrani right a pause
if leftTraining < rightTraining and leftTraining < pauseTraining:
    print("nejmeně je leftTraining")
    reduceRight = rightTraining - leftTraining
    reducePause = pauseTraining - leftTraining

    i = 0
    while i < len(allLabelsForNN):
        if allLabelsForNN[i][1] == 1 and reduceRight != 0:
            del allLabelsForNN[i]
            del allDataForNN[i]
            reduceRight -= 1
            i -= 1
        elif allLabelsForNN[i][2] == 1 and reducePause != 0:
            del allLabelsForNN[i]
            del allDataForNN[i]
            reducePause -= 1
            i -= 1
        i += 1
# odebrani left a pause
elif rightTraining < leftTraining and rightTraining < pauseTraining:

    reduceLeft = leftTraining - rightTraining
    reducePause = pauseTraining - rightTraining
    print("nejmeně je rightTraining")
    print("reduceLeft: ", reduceLeft)
    print("reducePause", reducePause)
    i = 0
    while i < len(allLabelsForNN):
        #print( "i: ", i)
        #print("allLabelsForNN[i] in while: ",allLabelsForNN[i])
        if allLabelsForNN[i][0] == 1 and reduceLeft != 0:
            #print("delete left")
            del allLabelsForNN[i]
            del allDataForNN[i]
            reduceLeft -= 1
            i -= 1
        elif allLabelsForNN[i][2] == 1 and reducePause != 0:
            #print("delete pause")
            del allLabelsForNN[i]
            del allDataForNN[i]
            reducePause -= 1
            i -= 1
        i += 1
# odebrani left a right
elif pauseTraining < leftTraining and pauseTraining < rightTraining:

    print("nejmeně je pauseTraining")
    reduceLeft = leftTraining - pauseTraining
    reduceRight = rightTraining - pauseTraining

    i = 0
    while i < len(allLabelsForNN):
        if allLabelsForNN[i][1] == 1 and reduceLeft != 0:
            del allLabelsForNN[i]
            del allDataForNN[i]
            reduceLeft -= 1
            i -= 1
        elif allLabelsForNN[i][2] == 1 and reduceRight != 0:
            del allLabelsForNN[i]
            del allDataForNN[i]
            reduceLeft -= 1
            i -= 1
        i += 1

leftTraining = 0
rightTraining = 0
pauseTraining = 0

for label in allLabelsForNN:
    if label[0] == 1:
        leftTraining += 1
    elif label[1] == 1:
        rightTraining += 1
    elif label[2] == 1:
        pauseTraining += 1


print("After reduce: left: ", leftTraining)
print("After reduce: right: ", rightTraining)
print("After reduce: pause: ", pauseTraining)


# prevedeni na numpy array, kvuli NN
allLabelsForNN = np.array(allLabelsForNN)
allDataForNN = np.array(allDataForNN)
print("Len of allDataForNN (np array): ", len(allDataForNN))
print("All data for NN: ",allDataForNN)
print("All labels for NN", allLabelsForNN)

allTestLabelsForNN = np.array(allTestLabelsForNN)
allTestDataForNN = np.array(allTestDataForNN)
print("Len of allTestingDataForNN (np array): ", len(allTestDataForNN))
print("All testing data for NN: ",allTestDataForNN)
print("All testing labels for NN", allTestLabelsForNN)
# _______________________________________
# Zkusebni NN



# __________________________________________________________________________________________
# keras model



#np.random.seed(7)

model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(64*len(channels),)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adamax', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(allDataForNN, allLabelsForNN, epochs=100, batch_size=1108)
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
    elif label[2] == 1:
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