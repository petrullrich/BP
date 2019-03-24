# author: Petr Ullrich
# VUT FIT - BP

import numpy as np
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import copy
import json

class challange:

    def __init__(self, filename, path):

        self.filename = filename
        self.path = path
        self.challenges = {}


        challange.load_json_labels(self)

    def load_json_labels(self):

        challenges = {}
        timestampChange = None
        challengeChange = None
        stageChange = None
        lastKey = None
        challengeRange = [[]] # prvni a posledni timestamp pro danou challenge (vyzvu)

        with open(self.path+self.filename, "r") as read_file:
            LABELSdata = json.load(read_file)

        for blok in LABELSdata:
            for timestamp, values in blok.items():

                lastTimestamp = timestamp

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
                if lastKey is None:
                    lastKey = str(values[0]) + str(values[1])
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

                # print("Doba trvani tohoto useku: ", round(float(timestamp) - float(timestampChange)), " sekund")
                # ----------------------------------------------------------------
                # vytvoreni dict s id challange + zacatky a konce danych challengi


                challengeRange[0].append(timestamp)

                if len(challengeRange[0]) != 2:
                    break

                print("Ch1: ", challenges)

                # pokud klic neexistuje, vytvori se list pro dany klic
                if not lastKey in challenges:
                    challenges[lastKey] = []

                print("ChallengeRange: ", challengeRange)
                challenges[lastKey][len(challenges[lastKey]):] = copy.deepcopy(challengeRange)
                print("Ch2: ", challenges)
                lastKey = str(values[0]) + str(values[1])
                print('Lastkey: ', lastKey)
                challengeRange[0].pop(0)
                print("ChallengeRange2: ", challengeRange)
                print("Ch3: ", challenges)
                #print(challenges)
                # ----------------------------------------------------------------

                # zjisteni challenge (vyzvy)
               # if values[0] == 1:
                  #  print("TIMESTAMP: ", timestamp, " | Zvedni levou ruku, otevrene oci. Stage: ", stage)
              #  elif values[0] == 2:
                  #  print("TIMESTAMP: ", timestamp, " | Mysli na zvedani leve ruky, oci otevrene. Stage: ", stage)
               # elif values[0] == 3:
                   # print("TIMESTAMP: ", timestamp, " | Mysli na zvedani leve ruky oci zavrene. Stage: ", stage)
              #  elif values[0] == 4:
                   # print("TIMESTAMP: ", timestamp, " | Zvedni pravou ruku, otevrene oci. Stage: ", stage)
              #  elif values[0] == 5:
                   # print("TIMESTAMP: ", timestamp, " | Mysli na zvedani prave ruky, oci otevrene. Stage: ", stage)
              #  else:
                   # print("TIMESTAMP: ", timestamp, " | Mysli na zvedani prave ruky oci zavrene. Stage: ", stage)

                timestampChange = timestamp

        # na konci je nutne doplnit rozsah posledni challenge
        challengeRange.append(lastTimestamp)
        challenges[lastKey] = []
        challenges[lastKey][len(challenges[lastKey]):] = challengeRange
        self.challenges = challenges
        print(self.challenges)
        print(challenges)

        list = [[]]
        list[0][len(list[0]):] = 'f'
        list[0][len(list[0]):] = 's'
        print(list[0])
        dic = {}
        dic[0] = []
        #dic[0].append(list)
       # dic[0].append(list)

        dic[0][len(dic[0]):] = copy.deepcopy(list)
        dic[0][len(dic[0]):] = list[:]
        #list[0][0] = list[0][1]
        #del list[0][1]
        list[0].pop(0)
        print(dic)