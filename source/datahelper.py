import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
from matplotlib.pyplot import cm
#from keras.preprocessing.sequence import pad_sequences
import math

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
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

def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]
	return X #.tolist()

def lzma_parser(line, PROTEIN_COUNT):
	X = np.zeros(PROTEIN_COUNT)
	
	lineNumstr = line.split("|")

	for i in range(0,PROTEIN_COUNT):	   
	    X[i] =float(lineNumstr[i])

	return X

def SW_parser(line, PROTEIN_COUNT):
	X = np.zeros(PROTEIN_COUNT)
	
	lineNumstr = line.split("|")
	#print("LINENUMBER",lineNumstr)
	#print("PROTEIN_COUNT",PROTEIN_COUNT)

	for i in range(0,PROTEIN_COUNT):	   
	    X[i] =float(lineNumstr[i])

	return X
		
## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
class DataSet(object):
  def __init__(self, fpath, setting_no, smilen, need_shuffle = False):
    
    self.SMILEN = smilen    
    self.charsmiset = CHARCANSMISET 
    self.charsmiset_size = CHARCANSMILEN
    self.PROBLEMSET = setting_no


  def read_sets(self, FLAGS): 
    
    fpath = FLAGS.dataset_path
  
    setting_no = FLAGS.problem_type
    print("Reading %s start" % fpath)

    test_fold = json.load(open(fpath + "folds/test_fold_setting" + str(setting_no)+".txt"))
    train_folds = json.load(open(fpath + "folds/train_fold_setting" + str(setting_no)+".txt"))
    
    return test_fold, train_folds

  def parse_data(self, FLAGS,  with_label=1): #1 -> any matrices
    fpath = FLAGS.dataset_path	
    
    if with_label == 1:
      ligands = json.load(open(fpath+"ligands_can.txt"), object_pairs_hook=OrderedDict)    
      proteins = json.load(open(fpath+"lzma.txt"), object_pairs_hook=OrderedDict)
      proteins1 = json.load(open(fpath+"SW.txt"), object_pairs_hook=OrderedDict)
      
    else:
      ligands = json.load(open(fpath+"ligands_can.txt"), object_pairs_hook=OrderedDict)
      proteins = json.load(open(fpath+"lzma.txt.txt"), object_pairs_hook=OrderedDict)

    Y = pickle.load(open(fpath + "Y","rb"), encoding='latin1') 
    if FLAGS.is_log:
        Y = -(np.log10(Y/(math.pow(10,9))))

    XD = []
    XT = []
        
    if with_label == 1:
        
        proteinFeatures = int(proteins["num"])       

        ckey = 0
        for t in proteins.keys():
          if ckey >= 1:             

            XT.append((lzma_parser(proteins[t], proteinFeatures))*(SW_parser(proteins1[t], proteinFeatures)))
            
          ckey = ckey + 1        

        for d in ligands.keys():
             XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))
        
  
    return XD, XT, Y




