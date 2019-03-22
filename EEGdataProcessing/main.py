# author: Petr Ullrich
# VUT FIT - BP
import  ChallengeClass
import ChannelDataClass

fPath = 'data/json/'

fFilename = '1552915099_features.json'

# nastaveni elektrod, ze kterych se zpracuji data
# musi byt list i v pripade jednoho channelu
channels = [1,2]

EEGdata = ChannelDataClass.channelData(channels, fFilename, fPath)

lPath = 'data/labels/'
lFilename = '1548110507_labels.json'

EEGlabels = ChallengeClass.challange(lFilename, lPath)