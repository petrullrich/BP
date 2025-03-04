# author: Petr Ullrich
# VUT FIT - BP

import numpy as np
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json
import copy
import ast

class channelData:

    def __init__(self, channels, filename, path):

        self.channels = channels
        self.filename = filename
        self.path = path

        self.fs = 250.0 # sampling frequency (vzorkovaci frekvence) (Hz)
        self.frameLength = 128 # velikost ramce (počet vzorku)
        self.frameShift = 45 # posun ramce (o kolik vzorku)

        # list listu s daty z fetures, z kanalu, ktere chceme pouzivat
        # v pripade jednoho kanalu pouze list
        self.data = []
        # list s fetaures timestampy
        self.timestamps = []
        # list listu - vstup do neuronove site (features)
        self.dataForNN = []
        # list listu - vstup do neur. s. (labels)
        self.labelsForNN = []

        # list listu - vstup do neuronove site (features) - TESTOVACI
        self.testingDataForNN = []
        # list listu - vstup do neur. s. (labels) - TESTOVACI
        self.testingLabelsForNN = []
        # list v kterem jsou indexy koncu jednotlivych challengi
        self.challengeEnd = []

    # upraveni souboru s features od Zdenka
    def repairFeatures(self):

        temporaryDic = {}
        repairedFeautures = '{'
        with open(self.path+self.filename, "r") as read_file:
            for line in read_file.readlines():
                if repairedFeautures[-1] != '}':
                    repairedFeautures = repairedFeautures + line[1:-2]

                # pridani carky
                repairedFeautures = repairedFeautures + ', '

        repairedFeautures = repairedFeautures[0:-2]
        repairedFeautures = repairedFeautures + '}'
        #print("repairedFeautures: ", repairedFeautures)
        print(type(repairedFeautures))
        repairedFeautures = ast.literal_eval(repairedFeautures)
        print(type(repairedFeautures))

        for key in sorted(repairedFeautures.keys()):
            temporaryDic.update({key : repairedFeautures[key]})
        repairedFeautures = "["+str(temporaryDic)+"]"
        repairedFeautures = repairedFeautures.replace("\'", "\"")

        #with open("data/dataZdenek/Repairedfeatures", "w") as writeRepaired:
        #    writeRepaired.write(str(repairedFeautures))
        with open(self.path+self.filename, "w") as write_file:
            write_file.write(repairedFeautures)

    # doplneni tridni promenne data
    def load_json_features(self):
        print("__________________")
        print("Načítání souboru ", self.path+self.filename)
        firstTimestamp = None
        samplescount = [0]  # poctet vzorku - tzn. [1,2,3,4, ... ,n]
        seconds = []  # sekundy v kterych prichazi jednotlive vzorky (pouzito pro sanitycheck)
        channelsCount = 0

        with open(self.path+self.filename, "r") as read_file:
            JSONdata = json.load(read_file)

        # pro kazdy kanal, ze ktereho chceme cist data -> list v self.data
        for eachChannel in self.channels:
            channelsCount = channelsCount + 1
            data = []
            for blok in JSONdata:
                for timestamp, values in blok.items():
                    # doplneni timestampu do pole timestampu z features
                    if channelsCount == 1:
                        self.timestamps.append(timestamp)

                    # doplneni pole s poctem vzorku - tzn. [1,2,3,4, ... ,n]
                    samplescount.insert(len(samplescount), samplescount[-1] + 1)

                    if firstTimestamp is None:
                        firstTimestamp = timestamp

                    # doplneni sekundy, ve kterem prichazi konkretni vzorek
                    seconds.insert(len(seconds), round(float(timestamp) - float(firstTimestamp), 3))

                    for dataType, EEGdata in values.items():
                        if (dataType == "e"):
                            # print("e data: ", EEGdata)
                            # print("Timestamp: ", timestamp, "|Sensor number", str(eachChannel), ": ", EEGdata[eachChannel-1])

                            data.insert(len(data), EEGdata[eachChannel - 1])

            self.data.append(data)


    # horni propust - filtrovani spodniho 1 Hz
    def removeDcOffset(self):
        hzCutOff = 3.0
        b, a = signal.butter(2, (hzCutOff/(self.fs/2)), 'highpass')
        # axis=1   -> provede se pro vsechna pole (channely)
        self.data = signal.lfilter(b, a, self.data, axis=1)
        #print("Data after highpass: ", self.data)
        print("Na data byl aplikován filtr horní propust (",hzCutOff, " Hz )")

    # pasmova zadrz - filtrovani sitoveho brumu
    def removeMainInterference(self):
        hzRange = np.array([50.0, 100.0]) # hlavni + harmonicke frekvence

        for eachHz in np.nditer(hzRange):
            bandstopHz = eachHz + 3.0 * np.array([-1, 1]) # nastaveni pasmove zadrze
            b, a = signal.butter(3, (bandstopHz/(self.fs/2.0)), 'bandstop')
            # axis=1   -> provede se pro vsechna pole (channely)
            self.data = signal.lfilter(b, a, self.data, axis=1)
        print("Na data byl aplikován filtr pasmová zádrž (",hzRange, " Hz )")

    # challengesAttributes - list listu -> prvni hodnota je offset od zacatku dat, druha hodnota je delka challenge
    # challengeType - typ challenge, pro kterou funkci volame -
    def processData(self, challengesAttributes, challengeType, channelsCount):

        setOfThreeFrames = []
        framesCounter = 0
        # pro kazdou challenge
        for challengeAttr in challengesAttributes:
            if(len(challengeAttr) == 0):
                break
            currentPosition = challengeAttr[0] # dosazeni offsetu dane challenge do aktualni pozice

            while(currentPosition + self.frameLength < (challengeAttr[0] + challengeAttr[1])):

                allChannelsDataInFrame = [] # promenna pro list s jiz upravenym ramcem,v kterem jsou vsechny kanaly poskladane za sebou

                for channelNumber in range(channelsCount):
                    # inicializace konkretniho ramce

                    dataInFrame = self.data[channelNumber][currentPosition : (currentPosition+self.frameLength)]

                    # zpracovani konkretniho ramce na konkretnim kanalu
                    dataInFrame = channelData.processFrame(self, dataInFrame)
                    allChannelsDataInFrame = allChannelsDataInFrame + dataInFrame


                # doplneni vstupnich dat pro NN
                self.dataForNN.append(copy.deepcopy(allChannelsDataInFrame))

                self.labelsForNN.append(copy.deepcopy(challengeType))

                currentPosition += self.frameShift  # posunuti framu o shift


        self.challengeEnd.append(len(self.dataForNN))

    # zpracovani daneho ramce:
    # fourierova transformace -> zahozeni druhe poloviny dat -> absolutni hodnota
    # vraci jiz upraveny ramec
    def processFrame(self, dataInFrame):
        dataInFrame = np.fft.fft(dataInFrame) # FFT
        dataInFrame = dataInFrame[0 : int(self.frameLength/2)] # potrebujeme pouze prvni polovinu hodnot - argumenty zahodime
        dataInFrame = list(map(abs, dataInFrame)) # kazde cislo prevedeno na absolutni hodnotu
        return dataInFrame