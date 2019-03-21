# author: Petr Ullrich
# VUT FIT - BP

import ChannelDataClass

path = 'data/json/'

filename = '1552915099_features.json'

# nastaveni elektrod, ze kterych se zpracuji data
# musi byt list i v pripade jednoho channelu
channels = [1,2]

EEGdata = ChannelDataClass.channelData(channels, filename, path)