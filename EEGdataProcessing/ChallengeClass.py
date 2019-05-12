# author: Petr Ullrich
# VUT FIT - BP

import numpy as np
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import copy
import json

class challenge:

    def __init__(self, filename, path):

        self.filename = filename
        self.path = path
        self.challenges = {}

    # upraveni souboru s labely od Zdenka
    def repairLabels(self):
        repairedLabels = '['
        with open(self.path+self.filename, "r") as read_file:
            for line in read_file.readlines():
                repairedLabels = repairedLabels + line
                #odstraneni newline (pokud to neni posledni radek - tam newline neni)
                if repairedLabels[-1] != '}':
                    repairedLabels = repairedLabels[0:-1]
                # pridani carky
                repairedLabels = repairedLabels + ','

        repairedLabels = repairedLabels[0:-1]
        repairedLabels = repairedLabels +']'
        #print("repairedLabels: ", repairedLabels)

        with open(self.path+self.filename, "w") as write_file:
            write_file.write(repairedLabels)


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
                # vytvoreni dict s id challenge + zacatky a konce danych challengi

                challengeRange[0].append(timestamp)

                if len(challengeRange[0]) != 2:
                    break

                #print("Ch1: ", challenges)

                # pokud klic neexistuje, vytvori se list pro dany klic
                if not lastKey in challenges:
                    challenges[lastKey] = []

                # print("ChallengeRange: ", challengeRange)
                challenges[lastKey][len(challenges[lastKey]):] = copy.deepcopy(challengeRange)
                # print("Ch2: ", challenges)
                lastKey = str(values[0]) + str(values[1])
                # print('Lastkey: ', lastKey)
                challengeRange[0].pop(0)
                # print("ChallengeRange2: ", challengeRange)
                # print("Ch3: ", challenges)
                #print(challenges)
                # ----------------------------------------------------------------

                timestampChange = timestamp

        # na konci je nutne doplnit rozsah posledni challenge
        challengeRange[0].append(lastTimestamp)
        #print("CHall range: ",challengeRange)
       # challenges[lastKey] = []
        challenges[lastKey][len(challenges[lastKey]):] = copy.deepcopy(challengeRange)
        self.challenges = challenges
        print('Challenges: ', self.challenges)

    # vrati offset a delku pro kazdou konkretni challenge v ramci sady challengi
    def get_challenge(self, challengeType, featuresTimestamps):

        challengeNumber = 0
        challengesAttributes = []
        try:
            # pokud je s danym klicem vice nez jedna challenge -> provedeme pro kazdou
            for currentChallenge in self.challenges[challengeType]:

                offsetBool = True
                challengeAttributes = []
                offset = 1

                # kontrolujeme kazdy timestamp v poli timestampu
                for featuresTimestamp in featuresTimestamps:

                    if(bool(offsetBool)):
                        # pokud je pocatecni timestamp dane challenge mensi/roven aktualne prochazenemu timestampu (z features) ->
                        # zapiseme si cislo prochazeneho timestampu (promenna offset) (pocet vzorku od zacatku souboru)

                        # offsetBool prepneme na False a dale se zjistuje delka dane challenge
                        if float(currentChallenge[0]) <= float(featuresTimestamp):
                            challengeAttributes.append(offset-1)
                            offsetBool = False
                    else:
                        # zjistovani delky challenge
                        # ve chvili, kdy je koncovy timestamp mensi/roven aktualne prochazenemu timestampu (z features) ->
                        # zapiseme delku challenge
                        if float(currentChallenge[1]) <= float(featuresTimestamp):
                            challengeAttributes.append(offset - challengeAttributes[0])
                            break

                    offset += 1

                # pokud existuje pocatecni timestamp pro danou challenge, ale koncovy ne, doplni se jako koncovy posledni timestamp
                if len(challengeAttributes) == 1:
                    #print('Celkovy pocet timestampu (features): ', len(featuresTimestamps))
                    #print('Offset: ', challengeAttributes[0])
                    challengeAttributes.append(len(featuresTimestamps) - challengeAttributes[0])

                challengesAttributes.append(copy.deepcopy(challengeAttributes))
                # print("OFFSET AND LENGTH: ",challengeAttributes)
                challengeNumber += 1

            #print('timestamp 0: ', featuresTimestamps[0])
            #print('timestamp 0+824: ', featuresTimestamps[0 + 824])
            #print('timestamp 3813: ', featuresTimestamps[3813])
            #print('timestamp 3813+1109: ', featuresTimestamps[3813+1109])
            print("ChallengesAttributes: ", challengesAttributes)

            return challengesAttributes

        except KeyError:
            print('Pro tuto sadu challengi neexistuji data')
            exit(0)
