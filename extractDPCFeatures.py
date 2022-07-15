import math
import numpy as np
import pandas as pd
import pickle
from keras.models import model_from_json
import tensorflow as tf
from propy import PyPro


# with open("./static/epi_all_Scale.pkl", 'rb') as file:
#     std_scale = pickle.load(file)
#
# with open("./static/epi_all_Model.pkl", 'rb') as file:
#     model = pickle.load(file)


pos = []


std_scale = pickle.load(open('ideepDPCBCE_Scale.pkl', 'rb'))
json_file = open('deepdpc/ideepDPCBCE_CNN_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("deepdpc/ideepDPCBCE_CNN_Model_Weights.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),  # 0.0002500000118743628
                    loss='binary_crossentropy',  # 'sparse_categorical_crossentropy',
                    metrics=['accuracy'])



def calcFV1(seq):
    # fv = [0 for x in range(153)]
    # fvMAT = []
    # fvIter = 0
    myMat = seqToMat(seq)
    myMat_t = []
    for val in myMat:
        for i in val:
            myMat_t.append(i)
    ########################################### why 21? ###############################
    myMat_t = myMat_t[:21]
    myFrequencyVec = frequencyVec(seq)
    myPRIM = PRIM(seq)
    myAAPIV = AAPIV(seq)
    myRPRIM = PRIM(seq[::-1])
    myRAAPIV = AAPIV(seq[::-1])
    myMat_t = pd.DataFrame(myMat_t).T.values
    myFV_T = pd.DataFrame(myFrequencyVec).T.values
    myAAPIV_T = pd.DataFrame(myAAPIV).T.values
    myRAAPIV_T = pd.DataFrame(myRAAPIV).T.values
    fvMAT = np.concatenate((myPRIM, myRPRIM, myFV_T, myAAPIV_T, myRAAPIV_T), axis=0)  # myMat_t,
    # np.savetxt('fvMAT.csv', fvMAT.T, delimiter=',')
    return np.array(fvMAT.T)

def calcFV(seq):
    allFVs = []
    DesObject = PyPro.GetProDes(seq.upper())

    fvs = DesObject.GetDPComp()  # GetPAAC(lamda=5, weight=0.05)
    allFVs.append(list(fvs.values()))
    return allFVs

def format_Sequence(seq):
    seqSize = 36 - len(seq)
    appendTxt = ""
    for val in range(seqSize):
        appendTxt = appendTxt + "X"
    return appendTxt + seq


def processAllStrings(fname):
    seqs = []
    allFVs = []
    with open(fname, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            seqs.append(currentPlace)
    allowed_chars = set('ACDEFGHIKLMNPQRSTVWXY')
    i = 1
    for seq in seqs:
        # print(str(i) + ': ' + seq)
        if seq.startswith('>'):  # seq != '':
            print(seq)
            continue
        else:
            seq = seq.upper()
            if set(seq).issubset(allowed_chars):
                # seq = format_Sequence(seq)
                print('Correct Sequence ' + str(i) + ': ' + seq + " " + str(len(seq)))
                fvs = calcFV(seq).reshape(-1, 945)
                for val in fvs:
                    allFVs.append(val)
                i = i + 1
            else:
                print('Invalid Sequence ' + str(i) + ': ' + seq + " " + str(len(seq)))
                # i = i + 1
    return allFVs


def processSequence(sequences):
    seqs = str(sequences)
    allFVs = []

    allowed_chars = set('ACDEFGHIKLMNPQRSTVWXY')
    i = 1
    # for seq in seqs:
    if seqs.isalpha():
        # print(str(i) + ': ' + seq)
        if seqs.startswith('>'):  # seq != '':
            print(seqs)
            # continue
        else:
            seq = seqs.upper()
            if set(seq).issubset(allowed_chars):
                # seq = format_Sequence(seq)
                # print('Correct Sequence ' +str(i) + ': ' + seq + " " + str(len(seq)))
                fvs = calcFV(seq)#.reshape(-1, 945)
                allFVs = np.array(fvs)  # .reshape(-1, 945)
                allFVs = np.nan_to_num(allFVs)
                # for val in fvs:
                #     allFVs.append(val)
                i = i + 1
            else:
                print('Invalid Sequence ' + str(i) + ': ' + seq + " " + str(len(seq)))
                # i = i + 1
    return allFVs


def processSequence1(sequences):
    seqs = str(sequences)
    allFVs = []

    allowed_chars = set('ACDEFGHIKLMNPQRSTVWXY')
    i = 1
    # for seq in seqs:
    if seqs.isalpha():
        # print(str(i) + ': ' + seq)
        if seqs.startswith('>'):  # seq != '':
            print(seqs)
            # continue
        else:
            seq = seqs.upper()
            if set(seq).issubset(allowed_chars):
                seq = format_Sequence(seq)
                print('Correct Sequence ' + str(i) + ': ' + seq + " " + str(len(seq)))
                fvs = calcFV(seq).reshape(-1, 945)
                for val in fvs:
                    allFVs.append(val)
                i = i + 1
            else:
                print('Invalid Sequence ' + str(i) + ': ' + seq + " " + str(len(seq)))
                # i = i + 1
    return allFVs



def formatSeq(seq, size=24):
    seq = str(seq).upper()
    samples = ''
    # appendString = ''
    seqLength = len(seq)
    stepStart = 0
    stepEnd = size

    i = 1
    # while i <= (36-size):
    #     appendString = appendString + "X"
    #     i += 1
    print(seq)
    while stepEnd <= len(seq):
        seqNew = seq[stepStart:stepEnd]
        # seqNew = appendString + seq[stepStart:stepEnd]
        samples = samples + seqNew + '-'
        pos.append((stepStart))
        # print(seqNew + "\n")
        stepStart += 1
        stepEnd += 1

    i = 1

    samples = samples + seq[stepEnd - 1:]
    pos.append(stepEnd)

    return samples


seq_Glob = ''


def feature_result(seq, size=19):
    seq_list = formatSeq(seq, size).split('-')
    seq_Glob = str(seq)
    result_list = []
    iter = 0
    for one_seq in seq_list:
        if one_seq.isalpha():
            fv = processSequence(one_seq)
            newFV = fv #np.array(fv).reshape(-1, 400)
            # newFV = np.nan_to_num(newFV)
            X = std_scale.transform(newFV)
            # X = np.array(X).reshape(-1, 945, 1)
            out = str(performPrediction(X))  # np.asarray(calcFV(one_seq))
            if out.__contains__("pBCE"):
                result_list.append([one_seq, out, str(pos[iter]) + "-" + str(pos[iter] + len(one_seq))])
                # Replace characters at index positions in list
                for i in range(pos[iter], (pos[iter] + len(one_seq))):
                    seq_Glob = seq_Glob[:i] + '#' + seq_Glob[i + 1:]

        iter = iter + 1
    if result_list:
        df = pd.DataFrame(result_list, columns=['SubSeq', 'Prediction', 'Range in Sequence'])
        # print(similar(one_seq, "GRWDEDGEKRIPLDVA"))
        print(result_list)
        print(seq)
        print(seq_Glob)

        return df, str(seq).upper(), str(seq_Glob).upper()
    else:
        return ['Invalid Sequence, does not contains BCell Epitope or the size of Epitope is different'],\
               str(seq).upper(), str(seq_Glob).upper()


def performPrediction(FV):
    output = model.predict(FV)
    print(output)
    y1 = output#[:, 1]
    # yLabel = np.round(y1)
    if y1 >= 0.9999:  # yLabel == 0:
        return 'pBCE '  + str(np.round(y1, 2))
    else:
        return 'nBCE '  + str(np.round(y1, 2))
