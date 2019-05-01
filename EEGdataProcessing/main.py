# author: Petr Ullrich
# VUT FIT - BP
import  ChallengeClass
import ChannelDataClass
import NNdataClass

import json
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.metrics import categorical_accuracy


def predictions_three_classes():
    # ----------------------------------------------
    # vyhodnoceni predikci

    print("predictions: ", predictions)
    print("len of predict", len(predictions))

    left = 0
    right = 0
    pause = 0

    for label in NN.allTestLabelsForNN:
        if label[0] == 1:
            left += 1
        elif label[1] == 1:
            right += 1
        elif label[2] == 1:
            pause += 1

    leftCorrect = 0
    rightCorrect = 0
    pauseCorrect = 0
    correct = 0
    wrong = 0

    for i in range(len(predictions)):
        if np.argmax(NN.allTestLabelsForNN[i]) == np.argmax(predictions[i]):
            indexOfMax = np.argmax(NN.allTestLabelsForNN[i])
            if indexOfMax == 0:
                leftCorrect += 1
                correct += 1
            elif indexOfMax == 1:
                rightCorrect += 1
                correct += 1
            elif indexOfMax == 2:
                pauseCorrect += 1
                correct += 1
        else:
            wrong += 1

    acc = sum(
        [np.argmax(NN.allTestLabelsForNN[i]) == np.argmax(predictions[i]) for i in range(len(predictions))]) / len(
        predictions)

    print("new acc: ", acc)

    print("correct: ", correct)
    print("leva: ", leftCorrect, "z", left)
    print("prava: ", rightCorrect, "z", right)
    print("pause: ", pauseCorrect, "z", pause)
    print("acc from predict: ", (correct / len(predictions)) * 100)
    print("wrong: ", wrong)

#_____________________________________________________________________________________________________________________
# MAIN

# instance tridy NNdata

NN = NNdataClass.NNdata()


# nastaveni potrebnych promennych tridy

# vybrani, kterou sadu challengi chceme trenovat:
# prvni: delani cinnosti (do_it)
# druha: mysleni s otevrenyma ocima (think_open)
# treti: mysleni se zavrenyma ocima (think_closed)

NN.challengeSet = 'think_closed'

NN.loadDataForNN = False

#NN.task_2('split_5')
NN.task_1()





# _______________________________________________________________
# sumarizace vstupnich dat do NN

print("______________________________")
print("Sumarizace vstupních dat do NN (numpy array)")
print("Delka trenovacích dat : ",len(NN.allDataForNN))
print("Trénovací features pro NN: ", NN.allDataForNN)
print("Trénovací labels pro NN", NN.allLabelsForNN)

print("--------")
print("Delka testovacích  dat: ",len(NN.allTestDataForNN))
print("Testovací features pro NN: ", NN.allTestDataForNN)
print("TEstovací labels pro NN", NN.allTestLabelsForNN)
print("______________________________")

# __________________________________________________________________________________________
# keras model



#np.random.seed(7)

model = Sequential()
model.add(Dense(300, activation='sigmoid', input_shape=(3*64*len(NN.channels),)))
model.add(Dense(150, activation='sigmoid'))
#model.add(Dense(300, activation='sigmoid'))
#model.add(Dense(20, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(NN.allDataForNN, NN.allLabelsForNN, epochs=50, batch_size=32, validation_split=0.10)
score = model.evaluate(NN.allTestDataForNN, NN.allTestLabelsForNN)
predictions = model.predict(NN.allTestDataForNN)

# ----------------------------------------------
# vyhodnoceni presnosti
print(model.metrics_names[1], score[1])
print(model.metrics_names[0], score[0])


predictions_three_classes()


