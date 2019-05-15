# author: Petr Ullrich
# VUT FIT - BP

import NNdataClass
import EEGstreamClass

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.metrics import categorical_accuracy


def predictions_three_classes():
    # ----------------------------------------------
    # vyhodnoceni predikci

    #print("predictions: ", predictions)
    print("Len of predictions: ", len(predictions))

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
    rightInsteadLeft = 0
    pauseInsteadLeft = 0

    leftInsteadRight = 0
    pauseInsteadRight = 0

    leftInsteadPause = 0
    rightInsteadPause = 0

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

            shouldBeIndex = np.argmax(NN.allTestLabelsForNN[i])
            predictedIndex = np.argmax(predictions[i])

            if shouldBeIndex == 0:
                if predictedIndex == 1: rightInsteadLeft += 1
                elif predictedIndex == 2: pauseInsteadLeft += 1
            elif shouldBeIndex == 1:
                if predictedIndex == 0: leftInsteadRight += 1
                elif predictedIndex == 2: pauseInsteadRight += 1
            elif shouldBeIndex == 2:
                if predictedIndex == 0: leftInsteadPause += 1
                elif predictedIndex == 1: rightInsteadPause += 1




    #acc = sum([np.argmax(NN.allTestLabelsForNN[i]) == np.argmax(predictions[i]) for i in range(len(predictions))]) / len(predictions)

    print("Přesnost (accuracy): ", round((correct / len(predictions)) * 100, 2),"\%")
    print("__________________")
    print("Počet správných predikcí: ", correct)
    print("Levá: ", leftCorrect, "z", left)
    print("Levá v procentech: ", round((leftCorrect/left)*100, 2), "\%")
    print("Pravá: ", rightCorrect, "z", right)
    print("Pravá v procentech: ", round((rightCorrect / right) * 100, 2), "\%")
    print("Nic nedělání: ", pauseCorrect, "z", pause)
    print("Nic nedělání v procentech: ", round((pauseCorrect / pause) * 100, 2), "\%")
    print("--------")
    print("Počet špatných predikcí: ", wrong, ", z toho:")
    print("Pravá místo levé: ", rightInsteadLeft)
    print("Pauza místo levé: ", pauseInsteadLeft)
    print("")
    print("Levá místo pravé: ", leftInsteadRight)
    print("Pauza místo pravé: ", pauseInsteadRight)
    print("")
    print("Levá místo pauzy: ", leftInsteadPause)
    print("Pravá místo pauzy: ", rightInsteadPause)

def predictions_two_classes():
    # ----------------------------------------------
    # vyhodnoceni predikci

    print("predictions: ", predictions)
    print("len of predict", len(predictions))

    do = 0
    pause = 0

    for label in NN.allTestLabelsForNN:
        if label[0] == 1:
            do += 1
        elif label[1] == 1:
            pause += 1

    doCorrect = 0
    pauseCorrect = 0
    correct = 0
    wrong = 0

    for i in range(len(predictions)):
        if np.argmax(NN.allTestLabelsForNN[i]) == np.argmax(predictions[i]):
            indexOfMax = np.argmax(NN.allTestLabelsForNN[i])
            if indexOfMax == 0:
                doCorrect += 1
                correct += 1
            elif indexOfMax == 1:
                pauseCorrect += 1
                correct += 1
        else:
            wrong += 1

    acc = sum(
        [np.argmax(NN.allTestLabelsForNN[i]) == np.argmax(predictions[i]) for i in range(len(predictions))]) / len(
        predictions)

    print("new acc: ", acc)

    print("correct: ", correct)
    print("činnost: ", doCorrect, "z", do)
    print("pauza: ", pauseCorrect, "z", pause)
    print("acc from predict: ", (correct / len(predictions)) * 100)
    print("špatně: ", wrong)


#_______________________________________________________________________________________________________________________
# MAIN

# instance tridy NNdata
NN = NNdataClass.NNdata()


#________________________________________________________________
# nastaveni potrebnych promennych tridy

NN.channels = [3,4]


# STREAM instance
#EEGstream = EEGstreamClass.EEGstream()
#EEGstream.stream()

# vybrani, kterou sadu challengi chceme trenovat:
# prvni: delani cinnosti (do_it)
# druha: mysleni s otevrenyma ocima (think_open)
# treti: mysleni se zavrenyma ocima (think_closed)

NN.challengeSet = 'think_closed'

NN.loadDataForNN = True



NN.typeOfClasses = 1
#NN.task_2('split_5')
NN.task_3()




leftTraining = 0
rightTraining = 0
pauseTraining = 0

for label in NN.allLabelsForNN:
    if label[0] == 1:
        leftTraining += 1
    elif label[1] == 1:
        rightTraining += 1
    elif label[2] == 1:
        pauseTraining += 1


# _______________________________________________________________
# sumarizace vstupnich dat do NN



print("______________________________")
print("Sumarizace vstupních dat do NN (numpy array)")
print("Delka trenovacích dat : ",len(NN.allDataForNN))
print("Trénovací features pro NN: ", NN.allDataForNN)
print("Trénovací labels pro NN", NN.allLabelsForNN)

print("--------")
print("Delka testovacích  dat: ",len(NN.allTestDataForNN))
print("Delka na indexu 0: ", len(NN.allTestDataForNN[0]))
print("Testovací features pro NN: ", NN.allTestDataForNN)
print("TEstovací labels pro NN", NN.allTestLabelsForNN)

print("--------")
print("Vstup do NN - levá: ", leftTraining)
print("Vstup do NN - levá: ", rightTraining)
print("Vstup do NN - levá: ", pauseTraining)
print("______________________________")

# __________________________________________________________________________________________
# keras model



#np.random.seed(7)

model = Sequential()
model.add(Dense(80, activation='sigmoid', input_shape=(64*len(NN.channels),)))
model.add(Dense(30, activation='sigmoid'))
#model.add(Dense(300, activation='sigmoid'))
#model.add(Dense(20, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(NN.allDataForNN, NN.allLabelsForNN, epochs=50, batch_size=32, validation_split=0.10)
score = model.evaluate(NN.allTestDataForNN, NN.allTestLabelsForNN)
predictions = model.predict(NN.allTestDataForNN)

# ----------------------------------------------
# vyhodnoceni presnosti
print(model.metrics_names[1], score[1])
print(model.metrics_names[0], score[0])

predictions_three_classes()


# ulozeni
model.save('data/kerasModels/firstModel.h5')
#del model

#model = load_model('data/kerasModels/firstModel.h5')



