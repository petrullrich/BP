# author: Petr Ullrich
# VUT FIT - BP
import  ChallengeClass
import ChannelDataClass

import json
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import categorical_accuracy

#_______________________________________________________________________________________________________________________
#                                                    MAIN
#_______________________________________________________________________________________________________________________

class NNdata:

    def __init__(self):
        # listy instanci trid
        self.EEGdata = []
        self.EEGlabels = []

        self.EEGtestData = []
        self.EEGTestlabels = []
        # list timestamu z features
        self.featuresTimestamps = []

        # data spojene z vice nahravani
        self.allDataForNN = []
        self.allLabelsForNN = []
        self.allTestDataForNN = []
        self.allTestLabelsForNN = []

        # pole nazvu sfeeatures a labels
        self.fFilenames = []
        self.lFilenames = []

        self.fTestFilenames = []
        self.lTestFilenames = []

        # nastaveni elektrod, ze kterych se zpracuji data
        # musi byt list i v pripade jednoho channelu
        self.channels = [1,2,3,4,5,6,7,8]

        # vybrani, kterou sadu challengi chceme trenovat:
        # prvni: delani cinnosti (do_it)
        # druha: mysleni s otevrenyma ocima (think_open)
        # treti: mysleni se zavrenyma ocima (think_closed)
        self.challengeSet = 'think_closed'

        # pokud jsou vstupni data do NN ze souboru -> loadDataForNN == True
        self.loadDataForNN = False
        #_________________________________________________
        # TASKS
        #_________________________________________________

        self.fPath = None
        self.lPath = None

        self.fTestPath = None
        self.lTestPath = None

        self.challengeEnd = []

        # typ rozdeleni trid
        # 1 - tri tridy: leva, prava, nic nedelani
        # 2 - dve tridy: cinnost (leva+prava), nic nedelani
        self.typeOfClasses = 1


    # TASK 1
    # vzit veskera data ze vsech souboru, provest filtrovani, fft, rozdelit do ramcu
    # pred predanim NN rozdelit v pomeru 85% trenovaci 15% testovaci
    def task_1(self):

        print("Experiment 1 - rozdělení zamíchaných dat na 85% a 15%")

        if self.loadDataForNN == False:

            print("Vstupní pole do NN budou nově vytvořena")

            self.fPath = 'data/tasks/src/'
            self.fFilenames.extend(['0feat','1feat','2feat','3feat','4feat','5feat', '6feat', '7feat', '8feat'])
            print("Zpracovávané features (názvy souborů): ", self.fFilenames)

            self.lPath = 'data/tasks/src/'
            self.lFilenames.extend(['0lab', '1lab', '2lab', '3lab', '4lab', '5lab', '6lab', '7lab', '8lab'])
            print("Zpracovávané labels (názvy souborů): ", self.lFilenames)


            NNdata.get_NN_data(self)



            # nahodne zamichani vsech dat (PYTHON LIST)
            mixed = list(zip(self.allDataForNN, self.allLabelsForNN))
            random.shuffle(mixed)
            self.allDataForNN, self.allLabelsForNN = zip(*mixed)
            self.allDataForNN = list(self.allDataForNN)
            self.allLabelsForNN = list(self.allLabelsForNN)
            print("___________")
            print("Proběhlo zamíchání vstupních dat!")




            # rozdeleni dat na trenovaci a testovaci
            print("Délka dat pro testovaní před doplněním 15%: ", len(self.allTestDataForNN))
            testingDataLen = int((len(self.allDataForNN)/100)*15)


            # inicializace testovacich dat
            self.allTestDataForNN = self.allDataForNN[-testingDataLen:] # zkopirovani poslednich 15% dat
            self.allDataForNN = self.allDataForNN[0:-testingDataLen] #odstraneni posledich 15%

            # inicializace testovacich dat
            self.allTestLabelsForNN = self.allLabelsForNN[-testingDataLen:] # zkopirovani poslednich 15% dat
            self.allLabelsForNN = self.allLabelsForNN[0:-testingDataLen] #odstraneni posledich 15%



            self.balance_classes(self.allDataForNN, self.allLabelsForNN)
            self.balance_classes(self.allTestDataForNN, self.allTestLabelsForNN)
            self.lists_to_numpy_arrays()

            # ________________________________________
            # Ulozeni vstupnich poli do NN do numpy array souboru
            np.save("data/tasks/task1/forNN/"+self.challengeSet+"/featuresForNN", self.allDataForNN)
            np.save("data/tasks/task1/forNN/"+self.challengeSet+"/labelsForNN", self.allLabelsForNN)
            np.save("data/tasks/task1/forNN/"+self.challengeSet+"/testFeaturesForNN", self.allTestDataForNN)
            np.save("data/tasks/task1/forNN/"+self.challengeSet+"/testLabelsForNN", self.allTestLabelsForNN)

        else:
            print("Načínání vstupních polí do NN z připravených souborů...")

            self.allDataForNN = np.load("data/tasks/task1/forNN/"+self.challengeSet+"/featuresForNN.npy")
            self.allLabelsForNN = np.load("data/tasks/task1/forNN/"+self.challengeSet+"/labelsForNN.npy")
            self.allTestDataForNN = np.load("data/tasks/task1/forNN/"+self.challengeSet+"/testFeaturesForNN.npy")
            self.allTestLabelsForNN = np.load("data/tasks/task1/forNN/"+self.challengeSet+"/testLabelsForNN.npy")

            print("Data úspěšně načtena!")


    # TASK 2
    # vstupni soubory rozdelit na testovaci a trenovaci v urcitem pomeru - 2 sessions test, 7 sessions trenink
    #
    def task_2(self, split):

        print("Experiment 2 - soubory (sessions) rozděleny na: 7 trénovacích a 2 testovací ")

        if self.loadDataForNN == True:

            print("Načínání vstupních polí do NN z připravených souborů...")

            self.allDataForNN = np.load("data/tasks/task2/forNN/"+split+"/"+self.challengeSet+"/featuresForNN.npy")
            self.allLabelsForNN = np.load("data/tasks/task2/forNN/"+split+"/"+self.challengeSet+"/labelsForNN.npy")
            self.allTestDataForNN = np.load("data/tasks/task2/forNN/"+split+"/"+self.challengeSet+"/testFeaturesForNN.npy")
            self.allTestLabelsForNN = np.load("data/tasks/task2/forNN/"+split+"/"+self.challengeSet+"/testLabelsForNN.npy")

            print("Data úspěšně načtena!")

            #allTestDataForNN = allDataForNN
            #allTestLabelsForNN = allLabelsForNN

        elif self.loadDataForNN == False:

            print("Vstupní pole do NN budou nově vytvořena")


            self.fPath = 'data/tasks/src/'
            self.lPath = 'data/tasks/src/'

            self.fTestPath = 'data/tasks/src/'
            self.lTestPath = 'data/tasks/src/'

            if split == 'split_1':
                # TRAINING
                self.fFilenames.extend(['0feat', '1feat', '2feat', '3feat', '4feat', '5feat', '6feat'])
                self.lFilenames.extend(['0lab', '1lab', '2lab', '3lab', '4lab', '5lab', '6lab'])
                # TESTING
                self.fTestFilenames.extend(['7feat', '8feat'])
                self.lTestFilenames.extend(['7lab', '8lab'])
            elif split == 'split_2':
                # TRAINING
                self.fFilenames.extend(['7feat', '1feat', '2feat', '3feat', '8feat', '5feat', '6feat'])
                self.lFilenames.extend(['7lab', '1lab', '2lab', '3lab', '8lab', '5lab', '6lab'])
                # TESTING
                self.fTestFilenames.extend(['0feat', '4feat'])
                self.lTestFilenames.extend(['0lab', '4lab'])
            elif split == 'split_3':
                # TRAINING
                self.fFilenames.extend(['0feat', '7feat', '2feat', '3feat', '4feat', '8feat', '6feat'])
                self.lFilenames.extend(['0lab', '7lab', '2lab', '3lab', '4lab', '8lab', '6lab'])
                # TESTING
                self.fTestFilenames.extend(['1feat', '5feat'])
                self.lTestFilenames.extend(['1lab', '5lab'])
            elif split == 'split_4':
                # TRAINING
                self.fFilenames.extend(['0feat', '1feat', '2feat', '8feat', '4feat', '5feat', '7feat'])
                self.lFilenames.extend(['0lab', '1lab', '2lab', '8lab', '4lab', '5lab', '7lab'])
                # TESTING
                self.fTestFilenames.extend(['6feat', '3feat'])
                self.lTestFilenames.extend(['6lab', '3lab'])
            else:
                # TRAINING
                self.fFilenames.extend(['0feat', '1feat', '7feat', '3feat', '4feat', '5feat', '6feat'])
                self.lFilenames.extend(['0lab', '1lab', '7lab', '3lab', '4lab', '5lab', '6lab'])
                # TESTING
                self.fTestFilenames.extend(['2feat', '8feat'])
                self.lTestFilenames.extend(['2lab', '8lab'])






            print("Zpracovávané features pro trénování (názvy souborů): ", self.fFilenames)
            print("Zpracovávané labels pro trénování (názvy souborů): ", self.lFilenames)
            print("--------------")
            print("Zpracovávané features pro testování(názvy souborů): ", self.fTestFilenames)
            print("Zpracovávané labels pro testování(názvy souborů): ", self.lTestFilenames)

            # FUNKCE PRO ZPRACOVANI

            NNdata.get_NN_data(self)
            NNdata.balance_classes(self, self.allDataForNN, self.allLabelsForNN)
            self.balance_classes(self.allTestDataForNN, self.allTestLabelsForNN)
            NNdata.lists_to_numpy_arrays(self)

            # ________________________________________
            # Ulozeni vstupnich poli do NN do numpy array souboru
            np.save("data/tasks/task2/forNN/"+split+"/"+self.challengeSet+"/featuresForNN", self.allDataForNN)
            np.save("data/tasks/task2/forNN/"+split+"/"+self.challengeSet+"/labelsForNN", self.allLabelsForNN)
            np.save("data/tasks/task2/forNN/"+split+"/"+self.challengeSet+"/testFeaturesForNN", self.allTestDataForNN)
            np.save("data/tasks/task2/forNN/"+split+"/"+self.challengeSet+"/testLabelsForNN", self.allTestLabelsForNN)

    # TASK 3
    # z kazde tridy vzit 15% - ty pouzit jako testovaci, zbytek jako trenovaci
    def task_3(self):

        print("Experiment 3 - z každé třídy v každém souboru se vezme 15% na testování, zbytek jsou trénovací data")

        if self.loadDataForNN == False:

            print("Vstupní pole do NN budou nově vytvořena")

            self.fPath = 'data/tasks/src/'
            self.fFilenames.extend(['0feat','1feat','2feat','3feat','4feat','5feat', '6feat', '7feat', '8feat'])
            print("Zpracovávané features (názvy souborů): ", self.fFilenames)

            self.lPath = 'data/tasks/src/'
            self.lFilenames.extend(['0lab', '1lab', '2lab', '3lab', '4lab', '5lab', '6lab', '7lab', '8lab'])
            print("Zpracovávané labels (názvy souborů): ", self.lFilenames)

            NNdata.get_NN_data(self)
            NNdata.percentage_from_each_challenge(self, 15)


            NNdata.balance_classes(self, self.allDataForNN, self.allLabelsForNN)
            self.balance_classes(self.allTestDataForNN, self.allTestLabelsForNN)
            NNdata.lists_to_numpy_arrays(self)

            # ________________________________________
            # Ulozeni vstupnich poli do NN do numpy array souboru
            np.save("data/tasks/task3/forNN/"+self.challengeSet+"/featuresForNN", self.allDataForNN)
            np.save("data/tasks/task3/forNN/"+self.challengeSet+"/labelsForNN", self.allLabelsForNN)
            np.save("data/tasks/task3/forNN/"+self.challengeSet+"/testFeaturesForNN", self.allTestDataForNN)
            np.save("data/tasks/task3/forNN/"+self.challengeSet+"/testLabelsForNN", self.allTestLabelsForNN)

        else:
            print("Načínání vstupních polí do NN z připravených souborů...")

            self.allDataForNN = np.load("data/tasks/task3/forNN/"+self.challengeSet+"/featuresForNN.npy")
            self.allLabelsForNN = np.load("data/tasks/task3/forNN/"+self.challengeSet+"/labelsForNN.npy")
            self.allTestDataForNN = np.load("data/tasks/task3/forNN/"+self.challengeSet+"/testFeaturesForNN.npy")
            self.allTestLabelsForNN = np.load("data/tasks/task3/forNN/"+self.challengeSet+"/testLabelsForNN.npy")

    def percentage_from_each_challenge(self, perc):
        # f - file
        f = len(self.challengeEnd)-1
        while f >= 0:
            prefix = 0
            # prefix pro kazdy "soubor"
            for p in range(0,f):
                prefix = prefix + self.challengeEnd[p][-1]
                #print("prefix: ", prefix)

            c = len(self.challengeEnd[f])-1
            self.challengeEnd[f] = [0] + self.challengeEnd[f]
            while c >= 0:

                testingDataLen = int(((self.challengeEnd[f][c+1] - self.challengeEnd[f][c])/100)*perc)
                currPos = self.challengeEnd[f][c]
                #print("testingDataLen: ", testingDataLen)

                # x procent zkopirovano do testovacich dat
                self.allTestDataForNN = self.allTestDataForNN + self.allDataForNN[prefix+currPos : prefix+currPos+testingDataLen]
                # nutne smazat zkopirovane data z trenovacich
                if prefix+currPos == 0:
                    self.allDataForNN = self.allDataForNN[prefix+currPos+testingDataLen : len(self.allDataForNN)]
                else:
                    self.allDataForNN = self.allDataForNN[0:(prefix+currPos)] + self.allDataForNN[prefix+currPos+testingDataLen:len(self.allDataForNN)]



                # x procent zkopirovano do testovacich dat
                self.allTestLabelsForNN = self.allTestLabelsForNN + self.allLabelsForNN[prefix + currPos : prefix + currPos + testingDataLen]
                # nutne smazat zkopirovane data z trenovacich
                if prefix + currPos == 0:
                    self.allLabelsForNN = self.allLabelsForNN[prefix + currPos + testingDataLen:len(self.allLabelsForNN)]
                else:
                    self.allLabelsForNN = self.allLabelsForNN[0:(prefix + currPos)] + self.allLabelsForNN[prefix + currPos + testingDataLen:len(self.allLabelsForNN)]


                c -= 1

            f -= 1




    # NASTAVI veskera data z jednoho souboru do vstupnich numpy poli do NN
    def set_all_data(self, dataType, challengeSet, index):

        challengeNumbers = ['3','6'] # defaultní hodnota

        if (challengeSet == 'think_closed'):
            challengeNumbers = ['3',
                                '6']  # v challengeNumbers jsou cisla, ktera odpovidaji prave a leve ruce pro danou sadu

        elif (challengeSet == 'think_open'):
            challengeNumbers = ['2', '5']
        elif (challengeSet == 'do_it'):
            challengeNumbers = ['1', '4']


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

                # PRVNI TYP TRID
                # format labelu pouzitych jako vstup do neuronove site
                # [x,y,z]
                # x == 1 && y==0 && z==0 -> leva ruka
                # x == 0 && y==1 && z==0 -> prava ruka
                # x == 0 && y==0 && z==1 -> nic nedelani (priprava + odpocinek)
                if self.typeOfClasses == 1:
                    challengeTypeBinary = [0, 0, 0]
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

                # DRUHY TYP TRID
                # format labelu pouzitych jako vstup do neuronove site
                # [x,y]
                # x == 1 && y==0 -> cinnost (leva+prava)
                # x == 0 && y==1 -> nic nedelani (priprava + odpocinek)
                elif self.typeOfClasses == 2:
                    challengeTypeBinary = [0, 0]
                    # stage 1 - priprava
                    if i == 1:
                        challengeTypeBinary[0] = 0
                        challengeTypeBinary[1] = 1
                    # stage 2 - samotna challenge
                    elif i == 2:
                        challengeTypeBinary[0] = 1
                        challengeTypeBinary[1] = 0
                    # stage 3 - pauza
                    else:
                        challengeTypeBinary[0] = 0
                        challengeTypeBinary[1] = 1

                challengeType = ch + str(i)
                #print('____________________________________________')
                #print('ChallengeType: ', challengeType)
                #print('ChallengeTypeBinary: ', challengeTypeBinary)
                if dataType == 'training':
                    challengesAttributes = self.EEGlabels[index].get_challenge(challengeType, self.featuresTimestamps[index])  # ziskani ofsetu a delky dane challenge

                    self.EEGdata[index].processData(challengesAttributes, challengeTypeBinary, len(self.channels))  # zpracovani dat pro danou challenge
                elif dataType == 'testing':
                    challengesAttributes = self.EEGTestlabels[index].get_challenge(challengeType, self.featuresTimestamps[index])  # ziskani ofsetu a delky dane challenge

                    self.EEGtestData[index].processData(challengesAttributes, challengeTypeBinary, len(self.channels))  # zpracovani dat pro danou challenge
                else:
                    print('Zadejte správný typ dat (training/testing)')


    # vytvoreni instanci channelData, nacteni souboru
    # volani funkci pro filtraci dat
    # volani funkce set_all_data
    # spojeni dat z vice souboru dohromady - DATA JSOU STALE JAKO PYTHON LIST
    def get_NN_data(self):

        # _________________________________________________________________
        # TRAINING

        # ----------------------------------------------------
        #                 FEATURES - TRAINING
        # ----------------------------------------------------

        # instance tridy channelDataClass
        for index, fFilename in enumerate(self.fFilenames):
            self.EEGdata.append(ChannelDataClass.channelData(self.channels, fFilename, self.fPath))
            # upraveni souboru s features od Zdenka
            # EEGdata[index].repairFeatures()
            # nacteni souboru s features a doplneni tridni promenne data
            self.EEGdata[index].load_json_features()

            self.featuresTimestamps.append(self.EEGdata[index].timestamps)

            self.EEGdata[index].removeDcOffset()  # filtr
            self.EEGdata[index].removeMainInterference()  # filtr

        print("")
        print("Probíhá zpracování trénovacích dat ...")
        # ----------------------------------------------------
        #                 LABELS - TRAINING
        # ----------------------------------------------------

        for index, lFilename in enumerate(self.lFilenames):
            # instance tridy challengeClass
            self.EEGlabels.append(ChallengeClass.challenge(lFilename, self.lPath))
            # upraveni souboru s labely od Zdenka
            # EEGlabels[index].repairLabels()
            # nacteni souboru s labely a doplneni tridni promenne chalenges
            self.EEGlabels[index].load_json_labels()



            # parametr index urcuje ktera instance tridy se prave zpracovava
            NNdata.set_all_data(self, 'training', self.challengeSet, index)

            # doplneni indexu koncu challengi pro danou isntanci (soubor)
            self.challengeEnd.append(self.EEGdata[index].challengeEnd)


            self.allDataForNN = self.allDataForNN + self.EEGdata[index].dataForNN
            self.allLabelsForNN = self.allLabelsForNN + self.EEGdata[index].labelsForNN


        # _________________________________________________________________
        # TESTING

        # ----------------------------------------------------
        #                 FEATURES - TESTING
        # ----------------------------------------------------

        self.featuresTimestamps.clear()

        # instance tridy channelDataClass
        for index, fFilename in enumerate(self.fTestFilenames):
            self.EEGtestData.append(ChannelDataClass.channelData(self.channels, fFilename, self.fTestPath))
            # upraveni souboru s features od Zdenka
            # EEGtestData[index].repairFeatures()
            # nacteni souboru s features a dpolneni tridni promenne data
            self.EEGtestData[index].load_json_features()

            self.featuresTimestamps.append(self.EEGtestData[index].timestamps)  # pole s timestampy z features

            self.EEGtestData[index].removeDcOffset()  # filtr
            self.EEGtestData[index].removeMainInterference()  # filtr

        print("")
        print("Probíhá zpracování testovacích dat ...")
        # ----------------------------------------------------
        #                 LABELS - TESTING
        # ----------------------------------------------------

        for index, lFilename in enumerate(self.lTestFilenames):
            # instance tridy challengeClass
            self.EEGTestlabels.append(ChallengeClass.challenge(lFilename, self.lTestPath))
            # upraveni souboru s labely od Zdenka
            # EEGTestlabels[index].repairLabels()
            # nacteni souboru s labely a doplneni tridni promenne chalenges
            self.EEGTestlabels[index].load_json_labels()



            # parametr index urcuje ktera instance tridy se prave zpracovava
            NNdata.set_all_data(self, 'testing', self.challengeSet, index)


            self.allTestDataForNN = self.allTestDataForNN + self.EEGtestData[index].dataForNN
            self.allTestLabelsForNN = self.allTestLabelsForNN + self.EEGtestData[index].labelsForNN


    # prevedeni listu na numpy array, kvuli NN
    def lists_to_numpy_arrays(self):

        self.allLabelsForNN = np.array(self.allLabelsForNN)
        self.allDataForNN = np.array(self.allDataForNN)


        self.allTestLabelsForNN = np.array(self.allTestLabelsForNN)
        self.allTestDataForNN = np.array(self.allTestDataForNN)

    # __________________________________________________________________________________________
    # pro trenovaci/testovaci vstupni data ->
    # VYVAZOVANI TRID
    # odstraneni prebytecnych ramcu - tzn. pro kazdou tridu stejny pocet ramcu
    # data musi byt ve formatu PYTHON LIST

    def balance_classes(self, features, labels):
        print("___________")
        print("Proběhne vyvažování tříd (funkce balance_classes)")
        left = 0
        right = 0
        pause = 0

        for label in labels:
            if label[0] == 1:
                left += 1
            elif label[1] == 1:
                right += 1
            elif label[2] == 1:
                pause += 1

        print("Before reduce: left: ", left)
        print("Before reduce: right: ", right)
        print("Before reduce: pause: ", pause)

        # odebrani right a pause
        if left < right and left < pause:
            print("nejmeně je left")
            reduceRight = right - left
            reducePause = pause - left

            i = 0
            while i < len(labels):
                if labels[i][1] == 1 and reduceRight != 0:
                    del labels[i]
                    del features[i]
                    reduceRight -= 1
                    i -= 1
                elif labels[i][2] == 1 and reducePause != 0:
                    del labels[i]
                    del features[i]
                    reducePause -= 1
                    i -= 1
                i += 1
        # odebrani left a pause
        elif right < left and right < pause:

            reduceLeft = left - right
            reducePause = pause - right
            print("nejmeně je right")
            print("reduceLeft: ", reduceLeft)
            print("reducePause", reducePause)
            i = 0
            while i < len(labels):
                # print( "i: ", i)
                # print("allLabelsForNN[i] in while: ",allLabelsForNN[i])
                if labels[i][0] == 1 and reduceLeft != 0:
                    # print("delete left")
                    del labels[i]
                    del features[i]
                    reduceLeft -= 1
                    i -= 1
                elif labels[i][2] == 1 and reducePause != 0:
                    # print("delete pause")
                    del labels[i]
                    del features[i]
                    reducePause -= 1
                    i -= 1
                i += 1
        # odebrani left a right
        elif pause < left and pause < right:

            print("nejmeně je pause")
            reduceLeft = left - pause
            reduceRight = right - pause

            i = 0
            while i < len(labels):
                if labels[i][1] == 1 and reduceLeft != 0:
                    del labels[i]
                    del features[i]
                    reduceLeft -= 1
                    i -= 1
                elif labels[i][2] == 1 and reduceRight != 0:
                    del labels[i]
                    del features[i]
                    reduceLeft -= 1
                    i -= 1
                i += 1

        left = 0
        right = 0
        pause = 0

        for label in labels:
            if label[0] == 1:
                left += 1
            elif label[1] == 1:
                right += 1
            elif label[2] == 1:
                pause += 1

        print("After reduce: left: ", left)
        print("After reduce: right: ", right)
        print("After reduce: pause: ", pause)

        return features, labels
