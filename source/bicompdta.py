
# We utilized, adapted and modified DeepDTA source codes(https://github.com/hkmztrk/DeepDTA) for BiComp-DTA experiments.
# @article{ozturk2018deepdta,
  # title={DeepDTA: deep drug--target binding affinity prediction},
  # author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  # journal={Bioinformatics},
  # volume={34},
  # number={17},
  # pages={i821--i829},
  # year={2018},
  # publisher={Oxford University Press}
# }

from __future__ import print_function
import matplotlib
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import random as rn
#import tensorflow.compat.v1 as tf
from tensorflow.python.keras import backend as K
### We modified Pahikkala et al. (2014) source code for cross-val process ###
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#import tensorflow.keras
import keras
#tf.set_random_seed(0)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
from datahelper import *
#import logging
from itertools import product
from arguments import argparser, logging
import keras
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
#from keras.layers.normalization import BatchNormalization
#from keras.layers.normalization import LayerNormalization
#from keras_layer_normalization import LayerNormalization
from tensorflow.python.keras.layers import LeakyReLU
from keras.layers import Concatenate
from keras.layers import Conv2D, GRU, Attention, SeparableConv1D,ConvLSTM1D
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Masking, RepeatVector, Flatten
from keras.models import Model
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers, layers
#from keras.layers import Merge
import sys, pickle, os
import math, json, time
import decimal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from copy import deepcopy
from sklearn import preprocessing
from emetrics import get_aupr, get_cindex, get_rm2

TABSY = "\t"
figdir = "figures/"

def BiCompDTA(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
   
    fpath = FLAGS.dataset_path
    proteins = json.load(open(fpath+"LZMA.txt"), object_pairs_hook=OrderedDict)
    proteinFeatures = int(proteins["num"])
    #proteinFeatures = 442
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32') 
    XTinput = Input(shape=(proteinFeatures, ), dtype='float32')
    
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size+1, output_dim=128, input_length=FLAGS.max_smi_len)(XDinput) 
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS*2, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = SeparableConv1D(filters=NUM_FILTERS*3, kernel_size=FILTER_LENGTH1,  activation='relu', padding='valid',  strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)
    #encode_smiles = MaxPooling1D()(encode_smiles)
    
    encode_protein = Dense(128, activation='linear')(XTinput)
    encode_protein = Dropout(0.2)(encode_protein)
    encode_protein = Dense(128, activation='linear')(encode_protein)
    encode_protein = Dropout(0.2)(encode_protein)
    encode_protein = Dense(76, activation='linear')(encode_protein) 

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein], axis=-1) 

    # Fully connected 
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.2)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.2)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)
    
    predictions = Dense(1, kernel_initializer='normal')(FC2) 

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    opt=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)
    #interactionModel.compile(optimizer=opt, loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    #opt=keras.optimizers.Adadelta(learning_rate=0.0001, rho=0.95, epsilon=1e-07, name="Adadelta", )
    interactionModel.compile(optimizer=opt, loss='mean_squared_error', metrics=[cindex_score]) #, metrics=['cindex_score']
    print(interactionModel.summary())
    plot_model(interactionModel, to_file='figures/model.png')

    return interactionModel
 
def nfold_1_2_3_setting_sample(XD, XT,  Y, label_row_inds, label_col_inds, measure, runmethod,  FLAGS, dataset):

    bestparamlist = []
    test_set, outer_train_sets = dataset.read_sets(FLAGS) 
    
    foldinds = len(outer_train_sets)

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    #logger.info('Start training')
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets)
        otherfolds.pop(val_foldind)
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds)
        test_sets.append(test_set)
        print("val set", str(len(val_fold)))
        print("train set", str(len(otherfoldsinds)))



    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, 
                                                                                                measure, runmethod, FLAGS, train_sets, val_sets)
   
    #print("Test Set len", str(len(test_set)))
    #print("Outer Train Set len", str(len(outer_train_sets)))
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, 
                                                                                                measure, runmethod, FLAGS, train_sets, test_sets)
    
    testperf = all_predictions[bestparamind]##pointer pos 

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" % 
            (bestparamind, bestparam), FLAGS)


    testperfs = []
    testloss= []

    avgperf = 0.

    for test_foldind in range(len(test_sets)):
        foldperf = all_predictions[bestparamind][test_foldind]
        foldloss = all_losses[bestparamind][test_foldind]
        testperfs.append(foldperf)
        testloss.append(foldloss)
        avgperf += foldperf

    avgperf = avgperf / len(test_sets)
    avgloss = np.mean(testloss)
    teststd = np.std(testperfs)

    logging("Test Performance CI", FLAGS)
    logging(testperfs, FLAGS)
    logging("Test Performance MSE", FLAGS)
    logging(testloss, FLAGS)

    return avgperf, avgloss, teststd

def general_nfold_cv(XD, XT,  Y, label_row_inds, label_col_inds, prfmeasure, runmethod, FLAGS, labeled_sets, val_sets): ## BURAYA DA FLAGS LAZIM????
    
    paramset1 = FLAGS.num_windows                              #[32, 128]  # filter numbers
    paramset2 = FLAGS.smi_window_lengths                               #[8, 16]filter length smi
    
    epoch = FLAGS.num_epoch                                 #250, 500, 1000
    batchsz = FLAGS.batch_size                             #706

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2) * len(paramset3)

    all_predictions = [[0 for x in range(w)] for y in range(h)] 
    all_losses = [[0 for x in range(w)] for y in range(h)] 
    print(all_predictions)

    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        Y_train = np.mat(np.copy(Y))

        params = {}
        XD_train = XD
        XT_train = XT
        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        XD_train = XD[trrows]
        XT_train = XT[trcols]

        train_drugs, train_prots,  train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)
        
        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]
        #print("terows", str(terows), str(len(terows)))
        #print("tecols", str(tecols), str(len(tecols)))

        val_drugs, val_prots,  val_Y = prepare_interaction_pairs(XD, XT,  Y, terows, tecols)


        pointer = 0
       
        for param1ind in range(len(paramset1)): #hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)): #learning rate
                param2value = paramset2[param2ind]

                for param3ind in range(len(paramset3)):
                    param3value = paramset3[param3ind]

                    gridmodel = runmethod(FLAGS, param1value, param2value, param3value)
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30,restore_best_weights=True,)
                    #es = EarlyStopping(monitor='val_cindex_score', mode='max', verbose=1, patience=75,restore_best_weights=True,)
                    gridres = gridmodel.fit(([np.array(train_drugs),np.array(train_prots) ]), np.array(train_Y), batch_size=batchsz, epochs=epoch, 
                            validation_data=( ([np.array(val_drugs), np.array(val_prots) ]), np.array(val_Y)),  shuffle=False, callbacks=[es] ) 


                    predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots) ])
                    loss, rperf2 = gridmodel.evaluate(([np.array(val_drugs),np.array(val_prots) ]), np.array(val_Y), verbose=0)
                    rperf = prfmeasure(val_Y, predicted_labels)
                    rperf = rperf[0]


                    logging("P1 = %d,  P2 = %d, P3 = %d, Fold = %d, CI-i = %f, CI-ii = %f, MSE = %f" % 
                    (param1ind, param2ind, param3ind, foldind, rperf, rperf2, loss), FLAGS)

                    plotLoss(gridres, param1ind, param2ind, param3ind, foldind)

                    all_predictions[pointer][foldind] =rperf #TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                    all_losses[pointer][foldind]= loss

                    pointer +=1

    bestperf = -float('Inf')
    bestpointer = None


    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
            for param2ind in range(len(paramset2)):
                for param3ind in range(len(paramset3)):
                
                    avgperf = 0.
                    for foldind in range(len(val_sets)):
                        foldperf = all_predictions[pointer][foldind]
                        avgperf += foldperf
                    avgperf /= len(val_sets)
                    #print(epoch, batchsz, avgperf)
                    if avgperf > bestperf:
                        bestperf = avgperf
                        bestpointer = pointer
                        best_param_list = [param1ind, param2ind, param3ind]

                    pointer +=1
        
    return  bestpointer, best_param_list, bestperf, all_predictions, all_losses

def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.linalg.band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) 
 
def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[] 
        
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target=XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity
      
def experiment(FLAGS, perfmeasure, deepmethod, foldcount=6): #5-fold cross validation + test

    #Input
    #XD: [drugs, features] sized array (features may also be similarities with other drugs
    #XT: [targets, features] sized array (features may also be similarities with other targets
    #Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    #perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    #higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    #foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation


    dataset = DataSet( fpath = FLAGS.dataset_path, ### BUNU ARGS DA GUNCELLE
                      setting_no = FLAGS.problem_type, ##BUNU ARGS A EKLE
                      seqlen = FLAGS.max_seq_len,
                      smilen = FLAGS.max_smi_len,
                      need_shuffle = False )
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size 
    FLAGS.charsmiset_size = dataset.charsmiset_size 

    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print(drugcount)
    targetcount = XT.shape[0]
    print(targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)  #basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                     perfmeasure, deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" % 
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_regression( FLAGS ): 

    perfmeasure = get_cindex
    print("")
    deepmethod = BiCompDTA

    experiment(FLAGS, perfmeasure, deepmethod)

if __name__=="__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression( FLAGS )
