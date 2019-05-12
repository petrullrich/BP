# author: Petr Ullrich
# VUT FIT - BP

from pylsl import StreamInlet, resolve_stream, StreamInfo, resolve_byprop
from scipy import signal

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import NNdataClass

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.metrics import categorical_accuracy




class EEGstream:

    xnDC = np.zeros([8,3])
    ynDC = np.zeros([8,3])

    xnRM = np.zeros([8,7])
    ynRM = np.zeros([8,7])

    xnRM2 = np.zeros([8, 7])
    ynRM2 = np.zeros([8, 7])

    def __init__(self):
        self.eeg = None
        self.aux = None
        self.fs = 250

        self.timestampsFrame = []
        self.samplesFrame = [[], [], [], [], [], [], [], []]
        self.samplesFrameNoFilter = [[], [], [], [], [], [], [], []]
        self.model = load_model('data/kerasModels/firstModel.h5')

        self.frameLength = 128
        self.channels = []

        #specgrams
        self.NFFT = 512
        self.overlap = self.NFFT - int(0.25 * self.fs)  # prekryti jednotlivych framu (bloku)
        self.showChannel = 6 # jaky kanal se zobrazi na spektrogramu


    # funkce ktera zarizuje streamovani dat z headsetu
    def stream(self):
        print("Start of streaming...")

        #_______________________________________
        # FILTRY - INIT

        hzCutOff = 5.0
        b, a = signal.butter(3, (hzCutOff / (self.fs / 2.0)), 'highpass')
        #print("b: ", b, "a: ", a)

        bandstopHz = np.array([47.0, 53.0])
        b2, a2 = signal.butter(3, (bandstopHz / (self.fs / 2.0)), 'bandstop')
        #print("b2: ", b2, "a2: ", a2)

        bandstopHz = np.array([97.0, 103.0])
        b3, a3 = signal.butter(3, (bandstopHz / (self.fs / 2.0)), 'bandstop')
        #print("b3: ", b3, "a3: ", a3)


        # KONEC FILTRU - INIT
        #________________________________________

        # instance of class NNdata
        NNdataInstance = NNdataClass.NNdata()
        self.channels = NNdataInstance.channels
        print("Channels: ", self.channels)
        # stream settings
        EEG_stream = resolve_stream('type', 'EEG')
        self.eeg = StreamInlet(EEG_stream[0])


        firstTimestamp = 0
        counter = 250
        #______________________________________________________________________________
        # stream dat v nekonecne smycce
        while True:

            # obdrzeni vzorku ze streamu
            sample, timestamp = self.eeg.pull_sample()


            if firstTimestamp == 0:
                firstTimestamp = timestamp

            #print(sample)

            #_________________________________________
            # filtrovani dat + extrakce prijatych dat ze streamu do listu v listu
            # kazdy jednotlivy list v samplesFrame odpovida jendomu kanalu
            for i in range(0, len(sample)):
                self.samplesFrameNoFilter[i].append(sample[i])
                #print("i: ", i)
                #print("Sample", sample[i])
                sample[i] = self.onlineRemoveDcOffset(sample[i], b, a, i)
                sample[i] = self.onlineRemoveMainInterference(sample[i], b2, a2, i)
                sample[i] = self.onlineRemoveMainInterferenceHarmonic(sample[i], b3, a3, i)
                #print("sample2: ", sample[i])
                self.samplesFrame[i].append(sample[i])



            #for i, channel in enumerate(sample):


            # pole za sebou prichazejicich timestampu
            self.timestampsFrame.append(timestamp)

            if len(self.timestampsFrame) == counter:
                print(counter/250, "sec")
                counter = counter + 250

            # PUVODNE: if len(self.timestampsFrame) ==  7500:
            # 7500 vzorku == 30 sekund nahravani
            # po tomto case sevykresli spektrogramy puvodniho a filtrovaneho signalu
            if len(self.timestampsFrame) ==  5000:
                self.signalplot()
                self.get_spectrum_data()
                self.spectrogram()

                self.samplesFrame = self.samplesFrameNoFilter

                self.signalplot()
                self.get_spectrum_data()
                self.spectrogram()

                plt.show()
                exit(0)



            # PUVODNE: if len(self.timestampsFrame) == 128
            # hlida 128 vzorku
            if len(self.timestampsFrame) == 128:
                EEGstream.processEEGstream(self)
                self.samplesFrame = [[], [], [], [], [], [], [], []]
                self.timestampsFrame = []






    def processEEGstream(self):

        #print("Function processEEGstream")

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

    # filtr - horni propust
    def onlineRemoveDcOffset(self, xn, b, a, i):

        # difrencialni rovnice
        yn = b[0]*xn + b[1]*self.xnDC[i][0] + b[2]*self.xnDC[i][1] + b[3]*self.xnDC[i][2] \
                                            - a[1]*self.ynDC[i][0] - a[2]*self.ynDC[i][1] - a[3]*self.ynDC[i][2]


        self.xnDC[i][2] = self.xnDC[i][1]
        self.xnDC[i][1] = self.xnDC[i][0]
        self.xnDC[i][0] = xn

        self.ynDC[i][2] = self.ynDC[i][1]
        self.ynDC[i][1] = self.ynDC[i][0]
        self.ynDC[i][0] = yn

        return yn

    # filtr - pasmova zadrz
    def onlineRemoveMainInterference(self, xn, b, a, i):

        # diferencialni rovnice
        yn = 0.0
        for x in range(0, len(b)):
            if x == 0:
                yn = b[x]*xn
                continue
            yn = yn + b[x]*self.xnRM[i][x-1]
            yn = yn - a[x]*self.ynRM[i][x-1]


        # prepsani minulych vzorku
        x = len(self.xnRM[i]) - 1
        while x > 0:
            self.xnRM[i][x] = self.xnRM[i][x-1]
            self.ynRM[i][x] = self.ynRM[i][x-1]
            x -= 1
        self.xnRM[i][0] = xn
        self.ynRM[i][0] = yn

        return yn

    def onlineRemoveMainInterferenceHarmonic(self, xn, b, a, i):

        # diferencialni rovnice
        yn = 0.0
        for x in range(0, len(b)):
            if x == 0:
                yn = b[x]*xn
                continue
            yn = yn + b[x]*self.xnRM2[i][x-1]
            yn = yn - a[x]*self.ynRM2[i][x-1]


        # prepsani minulych vzorku
        x = len(self.xnRM2[i]) - 1
        while x > 0:
            self.xnRM2[i][x] = self.xnRM2[i][x-1]
            self.ynRM2[i][x] = self.ynRM2[i][x-1]
            x -= 1
        self.xnRM2[i][0] = xn
        self.ynRM2[i][0] = yn

        return yn


    def get_spectrum_data(self):
        print("Calculating spectrum data...")
        print("Data in get_scpectrum_data: ", self.samplesFrame[self.showChannel])
        self.spec_PSDperHz, self.spec_freqs, self.spec_t = mlab.specgram(np.squeeze(self.samplesFrame[self.showChannel]),
                                                                         NFFT=self.NFFT,
                                                                         window=mlab.window_hanning,
                                                                         Fs=self.fs,
                                                                         noverlap=self.overlap
                                                                         )  # returns PSD power per Hz
        # convert the units of the spectral data
        self.spec_PSDperBin = self.spec_PSDperHz * self.fs / float(self.NFFT)

    def spectrogram(self):
        print("Generating spectrogram...")
        f_lim_Hz = [0, 50]  # frequency limits for plotting
        plt.figure(figsize=(10, 5))
        ax = plt.subplot(1, 1, 1)
        plt.pcolor(self.spec_t, self.spec_freqs, 10 * np.log10(self.spec_PSDperBin))  # dB re: 1 uV
        plt.clim([-25, 26])
        plt.xlim(self.spec_t[0], self.spec_t[-1])
        plt.ylim(f_lim_Hz)
        plt.ylim(0, 60)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title(self.plot_title('Spectrogram'))
        # add annotation for FFT Parameters
        ax.text(0.025, 0.95,
                "NFFT = " + str(self.NFFT) + "\nfs = " + str(int(self.fs)) + " Hz",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                backgroundcolor='w')
        self.plotit(plt)

    # funkce pro vytvoreni grafu signal
    def signalplot(self):
        self.t_sec = np.arange(len(self.timestampsFrame))/self.fs
        print("Generating signal plot...")
        plt.figure(figsize=(30, 5))
        plt.subplot(1, 1, 1)
        plt.plot(self.t_sec, self.samplesFrame[self.showChannel])
        #plt.margins(x=-0.4, y=-0.4)
        #plt.ylim(-50,50) #omezeni osy y
        #plt.xlim(27,28)  #omezeni osy x
        plt.xlabel('Time (sec)')
        plt.ylabel('Power (uV)')
        plt.title(self.plot_title('Signal'))
        self.plotit(plt)

    def plotit(self, plot):
        plot.draw()


    def plot_title(self, title=""):
        return 'Channel ' + str(self.showChannel) + ' ' + title + '\n'







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
