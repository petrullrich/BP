# author: Petr Ullrich
# VUT FIT - BP
import  ChallengeClass
import ChannelDataClass

fPath = 'data/json/'
fFilename = '1548110507_features.json'

# nastaveni elektrod, ze kterych se zpracuji data
# musi byt list i v pripade jednoho channelu
channels = [1,2]

EEGdata = ChannelDataClass.channelData(channels, fFilename, fPath)
featuresTimestamps = EEGdata.timestamps

lPath = 'data/labels/'
lFilename = '1548110507_labels.json'

EEGlabels = ChallengeClass.challange(lFilename, lPath)

chall = EEGlabels.get_challange('61', featuresTimestamps)

print("chall: ", chall)