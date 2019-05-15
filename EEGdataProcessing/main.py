# author: Petr Ullrich
# VUT FIT - BP

import NNdataClass
import EEGstreamClass

import numpy as np
import argparse

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.metrics import categorical_accuracy


def predictions_three_classes():
    # ----------------------------------------------
    # vyhodnoceni predikci
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

    print("Přesnost (accuracy): ", round((correct / len(predictions)) * 100, 2),"%")
    print("__________________")
    print("Počet správných predikcí: ", correct)
    print("Levá: ", leftCorrect, "z", left)
    print("Levá v procentech: ", round((leftCorrect/left)*100, 2), "%")
    print("Pravá: ", rightCorrect, "z", right)
    print("Pravá v procentech: ", round((rightCorrect / right) * 100, 2), "%")
    print("Nic nedělání: ", pauseCorrect, "z", pause)
    print("Nic nedělání v procentech: ", round((pauseCorrect / pause) * 100, 2), "%")
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

# _________________________________________________________
# PARSE ARGUMENTS

parser = argparse.ArgumentParser()
# jaky experiment bude vybran pro rozdeleni
parser.add_argument('-l', '--load', help='Načtení dat připravených pro klasifikaci ze souborů (pokud existují) nebo jejich vytvoření ze vstupních dat (true, false)')
parser.add_argument('-cs', '--class_set', help='Veberte sadu tříd, kterou chcete klasifikovat (think_open, think_closed, do_it) ... defaultně think_closed')
parser.add_argument('-ch', '--channels', help='Zadejte čísla elektrod, které chcete zpracovávat (1-8)', nargs='*')
parser.add_argument('-e', '--experiment', help='Zvolte číslo experimentu/rozdělení (1, 2, 3)')
parser.add_argument('-s', '--split', help='Pokud je zvolen experiment 2, vyberte jeho konkrétní rozdělení (1, 2, 3, 4, 5)')
parser.add_argument('-st', '--stream', help='Pro zapnutí real_time zpracování dat: true, musí již existovat natrénovaný model neuronové sítě')


args = vars(parser.parse_args())

# ARG PRO KANALY
if args['channels']:
    NN.channels = []
    if '1' in args['channels']:
        NN.channels.append(1)
    if '2' in args['channels']:
        NN.channels.append(2)
    if '3' in args['channels']:
        NN.channels.append(3)
    if '4' in args['channels']:
        NN.channels.append(4)
    if '5' in args['channels']:
        NN.channels.append(5)
    if '6' in args['channels']:
        NN.channels.append(6)
    if '7' in args['channels']:
        NN.channels.append(7)
    if '8' in args['channels']:
        NN.channels.append(8)

    if not NN.channels:
        print('Zvoleny neplatné elektrody pro zpracování!')
        exit(1)

#ARG PRO STREAM
if args['stream'] == 'true':
    try:
        load_model('data/kerasModels/firstModel.h5')
    except:
        print('Není k dispozici natrénovaný model neuronové sítě')
        print('Prosím vyberte experiment, nastavte další potřebné parametry a natrénujte model')
        exit(1)

    EEGstream = EEGstreamClass.EEGstream()
    EEGstream.channels = NN.channels
    EEGstream.stream()



# ARG PRO NACTENI DAT PRO NN ZE SOUBORU
if args['load'] == 'true':
    NN.loadDataForNN = True
elif args['load'] == 'false':
    NN.loadDataForNN = False

# ARG PRO SADU TRID
if args['class_set'] == 'think_open':
    NN.challengeSet = 'think_open'
elif args['class_set'] == 'think_closed':
    NN.challengeSet = 'think_closed'
elif args['class_set'] == 'do_it':
        NN.challengeSet = 'do_it'

# ARG PRO EXPERIMENTY A SPLITY
if args['experiment'] == '1':
    print("Experiment 1")
    print("Zpracovávané elektrody: ", NN.channels)
    NN.task_1()

elif args['experiment'] == '2':

    if args['split'] == '1':
        print("Experiment 2, rozdělení 1")
        NN.task_2('split_1')
    elif args['split'] == '2':
        print("Experiment 2, rozdělení 2")
        NN.task_2('split_2')
    elif args['split'] == '3':
        print("Experiment 2, rozdělení 3")
        NN.task_2('split_3')
    elif args['split'] == '4':
        print("Experiment 2, rozdělení 4")
        NN.task_2('split_4')
    elif args['split'] == '5':
        print("Experiment 2, rozdělení 5")
        NN.task_2('split_5')
    else:
        print("Je špatně určen parametr \"--split\" nebo není určen vůbec.")
        exit(1)

elif args['experiment'] == '3':
    print("Experiment 3")
    NN.task_3()

# END OF PARSING
# _________________________________________________________




#________________________________________________________________
# nastaveni potrebnych promennych tridy

#NN.channels = [1,2,3,4,5,6,7,8]


# STREAM instance
#EEGstream = EEGstreamClass.EEGstream()
#EEGstream.stream()

# vybrani, kterou sadu challengi chceme trenovat:
# prvni: delani cinnosti (do_it)
# druha: mysleni s otevrenyma ocima (think_open)
# treti: mysleni se zavrenyma ocima (think_closed)

NN.challengeSet = 'think_closed'

NN.loadDataForNN = False
#NN.task_1()

#NN.task_2('split_5')
#NN.task_1()


if len(NN.allTestDataForNN) == 0:
    print("Neexistují data pro zpracování!")
    exit(1)

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
print("Sumarizace vstupních dat do NN")
print("Delka trenovacích dat : ",len(NN.allDataForNN))
#print("Trénovací features pro NN: ", NN.allDataForNN)
#print("Trénovací labels pro NN", NN.allLabelsForNN)

print("--------")
print("Delka testovacích  dat: ",len(NN.allTestDataForNN))
#print("Testovací features pro NN: ", NN.allTestDataForNN)
#print("TEstovací labels pro NN", NN.allTestLabelsForNN)

print("--------")
print("Vstup do NN - levá: ", leftTraining)
print("Vstup do NN - pravá: ", rightTraining)
print("Vstup do NN - nic nedělání: ", pauseTraining)
print("______________________________")

# __________________________________________________________________________________________
# keras model



#np.random.seed(7)

model = Sequential()
model.add(Dense(80, activation='sigmoid', input_shape=(64*len(NN.channels),)))
model.add(Dense(30, activation='sigmoid'))
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



