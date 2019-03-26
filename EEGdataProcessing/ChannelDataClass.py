# author: Petr Ullrich
# VUT FIT - BP

import numpy as np
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json

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

        channelData.load_json_features(self)

    def load_json_features(self):

        firstTimestamp = None
        samplescount = [0]  # poctet vzorku - tzn. [1,2,3,4, ... ,n]
        seconds = []  # sekundy v kterych prichazi jednotlive vzorky (pouzito pro sanitycheck)
        channelsCount = 0
        print(self.path+self.filename)
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

        print(self.timestamps)
        print(*self.data, sep = "\n")


    # horni propust - filtrovani spodnich 5 Hz
    def removeDcOffset(self):
        hzCutOff = 5.0
        b, a = signal.butter(2, (hzCutOff/(self.fs/2)), 'highpass')
        self.data = signal.lfilter(b, a, self.data, 0)

    # pasmova zadrz - filtrovani sitoveho brumu
    def removeMainInterference(self):
        hzRange = np.array([50.0, 100.0]) #hlavni + harmonicke frekvence
        for eachHz in np.nditer(hzRange):
            bandstopHz = eachHz + 3.0 * np.array([-1, 1]) # nastaveni pasmove zadrze
            b, a = signal.butter(3, (bandstopHz/(self.fs/2.0)), 'bandstop')
            self.data = signal.lfilter(b, a, self.data, 0)

    # challengesAttributes - list listu -> prvni hodnota je offset od zacatku dat, druha hodnota je delka challenge
    # challengeType - typ challenge, pro kterou funkci volame -
    # TODO - challengeType vyuzit pro prirazeni typu challenge do druheho pole -- vytvorit toto pole
    # TODO - pro kazdy index tohoto pole sedi index framu z dane challenge
    def processData(self, challengesAttributes, challengeType):

        # pro kazdou challenge
        for challengeAttr in challengesAttributes:
            currentPosition = challengeAttr[0]
            dataInFrame = []
            while(currentPosition < (challengeAttr[0] + challengeAttr[1])):
                # TODO - dokud nejsme na konci challenge, brat vzdy jeden frame ->
                # TODO -> s nim pracovat (s daty v current framu)
                for i in range(currentPosition, currentPosition+self.frameLength):
                    dataInFrame.append(self.data[i])

                # TODO - zkontrolovat dataInFrame - dale udelat potrebne operace s daty -> data dat do pole poli
                # TODO - k nim prislusny challengeType (do druheho pole poli resp. list listu)

                currentPosition += self.frameShift  # posunuti framu o shift

    def getFrame(self, offset, length):
        print()