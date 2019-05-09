# author: Petr Ullrich
# VUT FIT - BP

from pylsl import StreamInlet, resolve_stream, StreamInfo, resolve_byprop
import time
from scipy import signal
import random
import numpy as np
import NNdataClass
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.metrics import categorical_accuracy


class EEGstream:

    def __init__(self):
        self.eeg = None
        self.aux = None
        self.fs = 250

        self.timestampsFrame = []
        self.samplesFrame = [[], [], [], [], [], [], [], []]
        self.model = load_model('data/kerasModels/firstModel.h5')

        self.frameLength = 128
        self.channels = []

    def stream(self):

        NNdataInstance = NNdataClass.NNdata()
        self.channels = NNdataInstance.channels

        EEG_stream = resolve_stream('type', 'EEG')
        print("function stram in EEGstream class")
        self.eeg = StreamInlet(EEG_stream[0])


        firstTimestamp = 0
        countOfSamples = 0
        timestampsOneSec = []
        timestamps = []


        while True:

            sample, timestamp = self.eeg.pull_sample()

            timestampsOneSec.append(timestamp)

            self.timestampsFrame.append(timestamp)

            for i, channel in enumerate(sample):
                self.samplesFrame[i].append(sample[i])

            #print(timestamp,sample)
            #print(len(sample))
            #print(self.samplesFrame)
            #print(self.timestampsFrame)

            countOfSamples += 1

            if firstTimestamp == 0:
                firstTimestamp = timestamp


            if len(self.timestampsFrame) == 128:
                EEGstream.processEEGstream(self)
                self.samplesFrame = [[], [], [], [], [], [], [], []]
                self.timestampsFrame = []

            # kontrola, zda ubehl0 10 vterin
            if timestamp - firstTimestamp > 10:
                #print("Vteřina!")
                print(len(timestampsOneSec))
                print(countOfSamples)
                print("___________")

                #--------------
                #downsample
                #--------------
                downsample = False

                if downsample == True:
                    if(len(timestampsOneSec) > 250):
                        numberOfDeleted = int(len(timestampsOneSec)-250)
                        part = int(len(timestampsOneSec) / numberOfDeleted)
                        i = numberOfDeleted - 1
                        pos = int(len(timestampsOneSec) - 1)
                        while i >= 0:
                            del timestampsOneSec[pos]
                            pos = pos - part
                            i -= 1


                timestamps = timestamps + timestampsOneSec
                timestampsOneSec = []

                firstTimestamp = 0
                countOfSamples = 0




    def processEEGstream(self):
        #todo
        #print("Function processEEGstream")


        EEGstream.removeDcOffset(self,self.samplesFrame)
        EEGstream.removeMainInterference(self, self.samplesFrame)


        dataForNN = [[]]
        oneChannel = []
        #print(self.samplesFrame)


        for i in self.channels:
            oneChannel = EEGstream.processFrame(self, self.samplesFrame[i-1])
            dataForNN[0] = dataForNN[0] + oneChannel


        dataForNN = np.array(dataForNN)
        #print(dataForNN)

        predictions = self.model.predict(dataForNN)

        # vyhodnoceni presnosti
        if predictions[0][0] > predictions[0][1] and predictions[0][0] > predictions[0][2]:
            print("Levá")
        elif predictions[0][1] > predictions[0][0] and predictions[0][1] > predictions[0][2]:
            print("Pravá")
        else:
            print("Nic")

        #print("Predictions:", predictions)



    # horni propust - filtrovani spodniho 1 Hz
    def removeDcOffset(self, data):
        hzCutOff = 1.0
        b, a = signal.butter(2, (hzCutOff/(self.fs/2)), 'highpass')

        # axis=1   -> provede se pro vsechna pole (channely)
        data = signal.lfilter(b, a, data, axis=1)
        #print("Data after highpass: ", self.data)
        #print("Na data byl aplikován filtr horní propust (",hzCutOff, " Hz )")
        return data


    # pasmova zadrz - filtrovani sitoveho brumu
    def removeMainInterference(self, data):
        hzRange = np.array([50.0, 100.0]) # hlavni + harmonicke frekvence

        for eachHz in np.nditer(hzRange):
            bandstopHz = eachHz + 3.0 * np.array([-1, 1]) # nastaveni pasmove zadrze
            b, a = signal.butter(3, (bandstopHz/(self.fs/2.0)), 'bandstop')
            # axis=1   -> provede se pro vsechna pole (channely)
            data = signal.lfilter(b, a, data, axis=1)
            #print("Data after bandstop: ", self.data)
        #print("Na data byl aplikován filtr pasmová zádrž (",hzRange, " Hz )")
        return data


    # zpracovani daneho ramce:
    # fourierova transformace -> zahozeni druhe poloviny dat -> absolutni hodnota
    # vraci jiz upraveny ramec
    def processFrame(self, dataInFrame):
        #print("Delka dat: ", len(dataInFrame))
        #print('Before FFT: ', dataInFrame)
        dataInFrame = np.fft.fft(dataInFrame) # FFT
        #print("Delka dat: ", len(dataInFrame))
        #print('After FFT: ', dataInFrame)
        dataInFrame = dataInFrame[0 : int(self.frameLength/2)] # potrebujeme pouze prvni polovinu hodnot - argumenty zahodime
        dataInFrame = list(map(abs, dataInFrame)) # kazde cislo prevedeno na absolutni hodnotu
        #print("Delka dat: ", len(dataInFrame))
        #print('AFTER ABS: ', dataInFrame)
        return dataInFrame