import EEGclass

source = 'openbci'

# Path to EEG data file
path = 'data/SavedData/'

# EEG data file name
# filename = 'openBCI_raw_2014-10-04_18-55-41_O1_Alpha.txt'
# MY OWN EEG DATA
filename = 'OpenBCI-RAW-2019-03-18_14-07-29.txt'

# Session title (used in some plots and such)
session_title = "OpenBCI Alpha detection"

# Channel
channel = 1

# ini
EEG = EEGclass.EEGrunt(path, filename, source, session_title)

# Here we can set some additional properties
# The 'plot' property determines whether plots are displayed or saved.
# Possible values are 'show' and 'save'
EEG.plot = 'show'

# nacte EEG data z txt z GUI
#EEG.load_data()
#EEG.load_channel(channel)

#nebo nacte JSON z prompteru
EEG.load_json(channel)

#nacte JSON labels z prompteru
#EEG.load_labels()


print("Processing channel "+ str(channel))

# Removes OpenBCI DC offset
EEG.remove_dc_offset()

# Notches 60hz noise (if you're in Europe, switch to 50Hz)
EEG.notch_mains_interference()

# Calculates spectrum data and stores as EEGrunt attribute(s) for reuse
EEG.get_spectrum_data()

#alpha frequency filter
EEG.alpha_filter()

# Make signal plot
EEG.signalplot()

# Returns bandpassed data
# (uses scipy.signal butterworth filter)
# start_Hz = 1
# stop_Hz = 50
# EEG.data = EEG.bandpass(start_Hz,stop_Hz)

# Make Spectrogram
EEG.spectrogram()

EEG.showplots()