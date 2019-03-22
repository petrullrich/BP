# author: Petr Ullrich
# VUT FIT - BP

import numpy as np
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json

class challange:

    def __init__(self, filename, path):

        self.filename = filename
        self.path = path


        challange.load_json_labels(self)

    def load_json_labels(self):

        challenges = {}
        timestampChange = None
        challengeChange = None
        stageChange = None

        stateChange = 0

        with open(self.path+self.filename, "r") as read_file:
            LABELSdata = json.load(read_file)

        for blok in LABELSdata:
            for timestamp, values in blok.items():
                # print("TIMESTAMP: ", timestamp)
                # print("VALUES: ", values)

                # kontrola, zda se zmenila vyzva, nebo stage
                # pokud ne, pokracuje se na dalsi timestamp
                if challengeChange == values[0]:
                    if stageChange == values[1]:
                        break
                    else:
                        stageChange = values[1]
                else:
                    challengeChange = values[0]
                    stageChange = values[1]

                # prvni inicializace
                if timestampChange is None:
                    timestampChange = timestamp
                if challengeChange is None:
                    challengeChange = values[0]
                if stageChange is None:
                    stageChange = values[1]

                # zjisteni stage
                if values[1] == 1:
                    stage = 1
                elif values[1] == 2:
                    stage = 2
                else:
                    stage = 3

                print("Doba trvani tohoto useku: ", round(float(timestamp) - float(timestampChange)), " sekund")

                challenges[str(values[0])+str(values[1])] = timestamp
                # zjisteni challenge (vyzvy)
                if values[0] == 1:
                    print("TIMESTAMP: ", timestamp, " | Zvedni levou ruku, otevrene oci. Stage: ", stage)
                elif values[0] == 2:
                    print("TIMESTAMP: ", timestamp, " | Mysli na zvedani leve ruky, oci otevrene. Stage: ", stage)
                elif values[0] == 3:
                    print("TIMESTAMP: ", timestamp, " | Mysli na zvedani leve ruky oci zavrene. Stage: ", stage)
                elif values[0] == 4:
                    print("TIMESTAMP: ", timestamp, " | Zvedni pravou ruku, otevrene oci. Stage: ", stage)
                elif values[0] == 5:
                    print("TIMESTAMP: ", timestamp, " | Mysli na zvedani prave ruky, oci otevrene. Stage: ", stage)
                else:
                    print("TIMESTAMP: ", timestamp, " | Mysli na zvedani prave ruky oci zavrene. Stage: ", stage)

                timestampChange = timestamp

                print(challenges)

        list = [['first']]
        list2 = 'sec'
        dic = {}
        dic[1] = list
        dic[1][0].append(list2)
        print("DIC: ", dic)