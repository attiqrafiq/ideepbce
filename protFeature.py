import pickle
import pandas as pd
import numpy as np
from propy import PyPro
import requests
from io import BytesIO

from urllib.request import urlopen

def load_model(url):
    modelLink = url
    model = requests.get(modelLink).content
    return model

#### To load from some url ####

# Step 1: Download the model file
url = "https://attique.cheaphost.pk/servermodel/epi__DPC_Model.pkl"
response = requests.get(url)
response.raise_for_status()  # Ensure the request was successful

# Step 2: Load the model using pickle
model_file = BytesIO(response.content)  # Convert the response content to a file-like object
_Clf = pickle.load(model_file)  # Load the model with pickle

# url = "https://attique.cheaphost.pk/servermodel/epi__DPC_Model.pkl"
# modelFile = load_model("https://attique.cheaphost.pk/servermodel/epi__DPC_Model.pkl")
# model = BytesIO(modelFile)
# _Clf = pickle.load(model)

# _Clf = pickle.load(urlopen("https://attique.cheaphost.pk/servermodel/epi__DPC_Model.pkl", 'rb')) 

#### End load from some url ####

# _Clf = pickle.load(open('https://attique.cheaphost.pk/servermodel/epi__DPC_Model.pkl','rb'))
std_scale = pickle.load(open('epi__DPC_Scale.pkl','rb'))


def processAllStrings(fname):
    seqs = []
    allFVs = []
    finalFVs = []
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
                DesObject = PyPro.GetProDes(seq)
                fvs = DesObject.GetPAAC(lamda=5, weight=0.05)  # calcFV(seq).reshape(-1,945)
                finalFVs.append(list(fvs.values()))
                # finalFVs.append(fvs)
                # for val in fvs:
                #     allFVs.append(val)
                i = i + 1
            else:
                print('Invalid Sequence ' + str(i) + ': ' + seq + " " + str(len(seq)))
                # i = i + 1
    return finalFVs


def processSequence(sequences):
    seqs = str(sequences)
    allFVs = []

    allowed_chars = set('ACDEFGHIKLMNPQRSTVWXY')
    i = 1
    # for seq in seqs:
    if seqs.isalpha():
        # print(str(i) + ': ' + seq)
        if seqs.startswith('>'):#seq != '':
            print(seqs)
            # continue
        else:
            seq = seqs.upper()
            if set(seq).issubset(allowed_chars):
                # print('Correct Sequence ' + str(i) + ': ' + seq + " " + str(len(seq)))
                DesObject = PyPro.GetProDes(seq)
                fvs = DesObject.GetPAAC(lamda=5, weight=0.05)#GetDPComp()  # calcFV(seq).reshape(-1,945)
                allFVs.append(list(fvs.values()))
                # for val in fvs:
                #     allFVs.append(val)
                i = i + 1
            else:
                print('Invalid Sequence ' + str(i) + ': ' + seq  + " " + str(len(seq)))
                # i = i + 1
    return allFVs


pos = []


def formatSeq(seq, size=19):
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
    seqNew = seq[stepEnd - 1:]
    # appendString = ''
    # while i <= (36 - len(seqNew)):
    #     appendString = appendString + "X"
    #     i += 1
    samples = samples + seq[stepEnd - 1:]
    # samples = samples + appendString + seq[stepEnd-1:]
    pos.append(stepEnd)

    return samples


seq_Glob = ''


def feature_result(seq, sLen=16):
    seq_list = formatSeq(seq, sLen).split('-')
    seq_Glob = str(seq)
    myStr = ''
    result_list = []
    iter = 0
    for one_seq in seq_list:
        if one_seq.isalpha():
            fv = processSequence(one_seq)
            feature = pd.DataFrame(fv)
            # df_all = feature.iloc[:, :]
            newFV = np.nan_to_num(feature.values)
            X = std_scale.transform(newFV)
            # X = np.array(X).reshape(-1, 1, 25)
            out = str(performPrediction(X))  # np.asarray(calcFV(one_seq))
            if out.__contains__("pBCE"):
                result_list.append([one_seq, out, str(pos[iter]) + "-" + str(pos[iter] + len(one_seq))])
                # if pos[iter] - pos[iter-1] > 5:
                # print([one_seq, out, pos[iter]])
                # result_list.append([one_seq, out, pos[iter]])
                # Replace characters at index positions in list
                for i in range(pos[iter], (pos[iter] + len(one_seq))):
                    seq_Glob = seq_Glob[:i] + 'b' + seq_Glob[i + 1:]

        iter = iter + 1
    if result_list:
        df_Result = pd.DataFrame(result_list, columns=['SubSeq', 'Prediction', 'Index in Sequence'])
        print(result_list)
        print(seq)
        print(seq_Glob)

        return df_Result, str(seq).upper(), str(seq_Glob).upper()
    else:
        result_str=['Invalid Sequence, does not contains BCell Epitope or the size of Epitope is different']
        df_Result = pd.DataFrame(result_str, columns=['Invalid Sequence'])
        return df_Result, str(seq).upper(), str(seq_Glob).upper()


def performPrediction(FV):
    output = _Clf.predict_proba(FV)
    # print(output)
    y1 = output[:, 1]
    # yLabel = np.round(y1)
    if y1 >= 0.75:#yLabel == 0:
        return 'pBCE ' # +str(y1)
    else:
        return 'nBCE ' # +str(y1)
