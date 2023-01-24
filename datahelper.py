import numpy as np
import json
import pickle
from collections import OrderedDict
import math

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

CHARPROTLEN = 25

CHARCANSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
                 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
                 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
                 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
                 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
                 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
                 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
                 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
                 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


## ######################## ##
#
#  Encoding Helpers
#
## ######################## ##
def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)))  # +1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch] - 1)] = 1

    return X  # .tolist()


def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch]) - 1] = 1

    return X  # .tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]
    return X  # .tolist()


def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  # .tolist()


def SW_parser(line, PROTEIN_COUNT):
    X = np.zeros(PROTEIN_COUNT)
    lineNumstr = line.split("|")

    for i in range(0, PROTEIN_COUNT):
        X[i] = float(lineNumstr[i])

    return X


def Pub_parser(line, DRUG_COUNT):
    X = np.zeros(DRUG_COUNT)

    lineNumstr = line.split("|")
    for i in range(0, DRUG_COUNT):
        X[i] = float(lineNumstr[i])

    return X


## ######################## ##
#
#  DATASET Class
#
## ######################## ##
# works for large dataset
class DataSet(object):
    def __init__(self, fpath, setting_no, seqlen, smilen, need_shuffle=False):
        self.SEQLEN = seqlen
        self.SMILEN = smilen
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET
        self.charsmiset_size = CHARISOSMILEN
        self.PROBLEMSET = setting_no

    def read_sets(self, FLAGS):  ### fpath should be the dataset folder /kiba/ or /davis/ or /pdb/ or /bindingdb/
        fpath = FLAGS.dataset_path
        setting_no = FLAGS.problem_type
        print("Reading %s start" % fpath)

        test_fold = json.load(open(fpath + "folds/test_fold_setting" + str(setting_no) + ".txt"))
        train_folds = json.load(open(fpath + "folds/train_fold_setting" + str(setting_no) + ".txt"))

        return test_fold, train_folds

    def parse_data(self, FLAGS,
                   with_label=1):  # 1 -> SimDTA, any matrices, 2 -> lookup-table encoding, 3 -> onehot encoding
        fpath = FLAGS.dataset_path
        print("Read %s start" % fpath)

        if with_label == 1:
            ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
            proteins = json.load(open(fpath + "protVeclzma1.txt"), object_pairs_hook=OrderedDict)
            proteins1 = json.load(open(fpath + "protVecSW.txt"), object_pairs_hook=OrderedDict)

        else:
            ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
            proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)

        Y = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')  ### TODO: read from raw
        if FLAGS.is_log == 1:
            Y = -(np.log10((Y) / (math.pow(10, 9))))
        if FLAGS.is_log == 2:
            Y = -(np.log10((Y + 1) / (math.pow(10, 9))))

        XD = []
        XT = []

        if with_label == 1:
            proteinFeatures = int(proteins["num"])

            ckey = 0
            for t in proteins.keys():
                if ckey >= 1:

                    if fpath == 'data/davis/':
                        XT.append((SW_parser(proteins[t], proteinFeatures)) * (
                                SW_parser(proteins1[t], proteinFeatures) / 100))
                    else:
                        XT.append(
                            (SW_parser(proteins[t], proteinFeatures)) * (SW_parser(proteins1[t], proteinFeatures)))

                ckey = ckey + 1

            for d in ligands.keys():
                XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))

        elif with_label == 2:
            for d in ligands.keys():
                XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))

            for t in proteins.keys():
                XT.append(label_sequence(proteins[t], self.SEQLEN, self.charseqset))
        elif with_label == 3:
            for d in ligands.keys():
                XD.append(one_hot_smiles(ligands[d], self.SMILEN, self.charsmiset))

            for t in proteins.keys():
                XT.append(one_hot_sequence(proteins[t], self.SEQLEN, self.charseqset))

        return XD, XT, Y
