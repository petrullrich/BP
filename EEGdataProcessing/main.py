# author: Petr Ullrich
# VUT FIT - BP
import  ChallengeClass
import ChannelDataClass

fPath = 'data/json/'
fFilename = '1548110507_features.json'

# nastaveni elektrod, ze kterych se zpracuji data
# musi byt list i v pripade jednoho channelu
channels = [1]

EEGdata = ChannelDataClass.channelData(channels, fFilename, fPath)
featuresTimestamps = EEGdata.timestamps

lPath = 'data/labels/'
lFilename = '1548110507_labels.json'

EEGlabels = ChallengeClass.challange(lFilename, lPath)

#
challengeType = '61'
challengesAttributes = EEGlabels.get_challange(challengeType, featuresTimestamps) # ziskani ofsetu a delky dane challenge
EEGdata.processData(challengesAttributes, challengeType, len(channels)) # zpracovani dat pro danou challenge