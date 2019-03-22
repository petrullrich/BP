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

