# author: Petr Ullrich
# VUT FIT - BP

import numpy as np
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import json

# https://github.com/curiositry/EEGrunt?fbclid=IwAR13TpUpz0j4QmBGddgeoWrQuG3F2_w7SYfMrSQhYrI7HdLRvqfWl3VYRrM
class EEGrunt:

    def __init__(self, path, filename, source, title = ""):

        self.path = path
        self.filename = filename
        self.source = source
        if(title):
            self.session_title = title
        else:
            self.session_title = source.title()+" data loaded from "+filename

        self.NFFT = 512

        self.col_offset = 0
        self.fs_Hz = 250.0 # vzorkovaci frekvence
        self.nchannels = 8
        self.channels = [1,2,3,4,5,6,7,8]

        self.plot = 'show'
        self.overlap = self.NFFT - int(0.25 * self.fs_Hz) #prekryti jednotlivych framu (bloku)
        print(self.overlap)
        
    # ---------------------
    # PU
    
    def load_json(self, channel):

        data = []
        self.channel = channel
        firstTimestamp = None
        timestamps = []
        samplescount = [0]
        seconds = []

        with open("data/json/1552915099_features.json", "r") as read_file:
            JSONdata = json.load(read_file)
        for blok in JSONdata:
            for timestamp, values in blok.items():
                # doplneni pole s poctem vzorku - tzn. [1,2,3,4, ... ,n]
                samplescount.insert(len(samplescount), samplescount[-1]+1)

                timestamps.insert(len(timestamps), timestamp)
                # print("TIMESTAMP: ", timestamp)
                if firstTimestamp is None:
                    firstTimestamp = timestamp

                seconds.insert(len(seconds), round(float(timestamp) - float(firstTimestamp),3))

                for dataType, EEGdata in values.items():
                    if(dataType == "e"):
                        #print("e data: ", EEGdata)
                        # print("Timestamp: ", timestamp, "|Sensor number", str(channel), ": ", EEGdata[channel-1])
                        data.insert(len(data), EEGdata[channel-1])


        self.data = data
        self.t_sec = np.arange(len(self.data)) / self.fs_Hz
        print("Session length (seconds): " + str(len(self.t_sec) / self.fs_Hz))
        print("Data from prompter: ", self.data)
        print("Sec: ", len(self.data) / self.fs_Hz)
        samplescount.pop(0)
        print("samplescount: ", samplescount)
        print("seconds: ", seconds)
        # vyploteni v jakem case prichazeji samply
        plt.plot(seconds, samplescount)
        plt.show()


    def load_labels(self):

        timestampChange = None
        challengeChange = None
        stageChange = None

        stateChange = 0

        with open("data/labels/1548110507_labels.json", "r") as read_file:
            LABELSdata = json.load(read_file)

        for blok in LABELSdata:
            for timestamp, values in blok.items():
                #print("TIMESTAMP: ", timestamp)
                #print("VALUES: ", values)
            

                # kontrola, zda se zmenila vyzva, nebo stage
                # pokud nepokracuje se na dalsi timestamp
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

    # PU - END
    # ----------------------


    def load_data(self):

        path = self.path
        filename = self.filename
        source = self.source

        print("Loading EEG data: " + path + filename)

        try:
            with open(path + filename) as file:
                pass
        except IOError:
            print("EEG data file not found.")
            exit()


        if source == 'openbci':
            skiprows = 6
            cols = (0, 1, 2, 3, 4, 5, 6, 7, 8)


        raw_data = np.loadtxt(path + filename,
                              delimiter=',',
                              skiprows=skiprows,
                              usecols=cols
                              )

        self.raw_data = raw_data
        self.t_sec = np.arange(len(self.raw_data[:, 0])) / self.fs_Hz

        print("Session length (seconds): " + str(len(self.t_sec) / self.fs_Hz))
        print("t_sec last: " + str(self.t_sec[:-1]))

    def load_channel(self, channel):
        print("Loading channel: " + str(channel))
        channel_data = self.raw_data[:, (channel + self.col_offset)]
        self.channel = channel
        self.data = channel_data
        print("Data from GUI: ", self.data)

    # horni propust
    def remove_dc_offset(self):
        print("Data in remove_dc_offset: ", self.data)
        hp_cutoff_Hz = 5.0
        print("Highpass filtering at: " + str(hp_cutoff_Hz) + " Hz")
        b, a = signal.butter(2, hp_cutoff_Hz / (self.fs_Hz / 2.0), 'highpass')
        self.data = signal.lfilter(b, a, self.data, 0)

    # pasmova zadrz
    def notch_mains_interference(self):
        print("Data in notch_mains_interfaces: ", self.data)
        notch_freq_Hz = np.array([50.0, 100.0])  # main + harmonic frequencies
        for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each target freq
            bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])  # set the stop band
            print(bp_stop_Hz / (self.fs_Hz / 2.0))
            b, a = signal.butter(3, bp_stop_Hz / (self.fs_Hz / 2.0), 'bandstop')
            self.data = signal.lfilter(b, a, self.data, 0)
            print("Notch filter removing: " + str(bp_stop_Hz[0]) + "-" + str(bp_stop_Hz[1]) + " Hz")

    # pasmova propust
    def alpha_filter(self):
        print("bandpass: 8-12 Hz")
        band_pass_Hz = np.array([8, 12])
        b, a = signal.butter(3, band_pass_Hz / (self.fs_Hz / 2.0), "bandpass")
        self.data = signal.lfilter(b, a, self.data, 0)
        print("Bandpass filtering to: " + str(band_pass_Hz[0]) + "-" + str(band_pass_Hz[1]) + " Hz")

    def bandpass(self, start, stop):
        bp_Hz = np.zeros(0)
        bp_Hz = np.array([start, stop])
        b, a = signal.butter(3, bp_Hz / (self.fs_Hz / 2.0), 'bandpass')
        print("Bandpass filtering to: " + str(bp_Hz[0]) + "-" + str(bp_Hz[1]) + " Hz")
        return signal.lfilter(b, a, self.data, 0)

    def plotit(self, plt, filename=""):
        if self.plot == 'show':
            plt.draw()
        if self.plot == 'save':
            plt.savefig(filename)
            plt.close()

    def plot_title(self, title=""):
        return 'Channel ' + str(self.channel) + ' ' + title + '\n' + self.session_title

    def plot_filename(self, title=""):
        fn = self.session_title + ' Channel ' + str(self.channel) + ' ' + title
        valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        filename = 'plots/' + (''.join(c for c in fn if c in valid_chars)).replace(' ', '_') + '.png'
        return filename

    # funkce pro vytvoreni grafu signal
    def signalplot(self):
        print("Generating signal plot...")
        plt.figure(figsize=(30, 5))
        plt.subplot(1, 1, 1)
        plt.plot(self.t_sec, self.data)
        #plt.margins(x=-0.4, y=-0.4)
        plt.ylim(-50,50) #omezeni osy y
        plt.xlim(27,28)  #omezeni osy x
        plt.xlabel('Time (sec)')
        plt.ylabel('Power (uV)')
        plt.title(self.plot_title('Signal'))
        self.plotit(plt)

    def get_spectrum_data(self):
        print("Calculating spectrum data...")
        print("Data in get_scpectrum_data: ", self.data)
        self.spec_PSDperHz, self.spec_freqs, self.spec_t = mlab.specgram(np.squeeze(self.data),
                                                                         NFFT=self.NFFT,
                                                                         window=mlab.window_hanning,
                                                                         Fs=self.fs_Hz,
                                                                         noverlap=self.overlap
                                                                         )  # returns PSD power per Hz
        # convert the units of the spectral data
        self.spec_PSDperBin = self.spec_PSDperHz * self.fs_Hz / float(self.NFFT)

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
                "NFFT = " + str(self.NFFT) + "\nfs = " + str(int(self.fs_Hz)) + " Hz",
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                backgroundcolor='w')
        self.plotit(plt, self.plot_filename('Spectrogram'))

    def showplots(self):
        if self.plot == 'show':
            print("Computation complete! Showing generated plots...")
            plt.show()
