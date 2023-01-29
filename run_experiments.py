from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf2
import time
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random as rn
import keras
from sklearn.metrics import average_precision_score
from keras import backend as K
from datahelper import *
from arguments import argparser, logging
from keras.layers import Conv1D, GlobalMaxPooling1D, SeparableConv1D, Input, Embedding, Dense, Dropout, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping
from copy import deepcopy
from emetrics import get_rm2

tf2.disable_v2_behavior()
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)

session_conf = tf2.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf2.set_random_seed(0)
sess = tf2.Session(graph=tf2.get_default_graph(), config=session_conf)
K.set_session(sess)

sns.set_theme(style='white')

figdir = "figures/"

################################
# SimDTA(BiComp-DTA)
def build_combined_categorical1(FLAGS, NUM_FILTERS, FILTER_LENGTH1):
    
    fpath = FLAGS.dataset_path
    if fpath=='data/davis/':
        proteinFeatures = 442
    elif fpath=='data/kiba/':
        proteinFeatures = 229
    elif fpath=='data/pdb/':
        proteinFeatures = 1606
    elif fpath=='data/bindingdb/':
        proteinFeatures = 1088
        
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='float32')
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=128, input_length=FLAGS.max_smi_len)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = SeparableConv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu',
                                    padding='valid', strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)
    
    XTinput = Input(shape=(proteinFeatures,), dtype='float32')
    encode_protein = Dense(128, activation='relu')(XTinput)
    encode_protein = Dropout(0.2)(encode_protein)
    encode_protein = Dense(128, activation='relu')(encode_protein)
    encode_protein = Dropout(0.2)(encode_protein)
    encode_protein = Dense(76, activation='relu')(encode_protein)
    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein],
                                                  axis=-1)  # merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.2)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.2)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(
        FC2)  # OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)
    interactionModel.compile(optimizer=opt, loss='mean_squared_error',
                             metrics=[cindex_score])  # , metrics=['cindex_score']
    print(interactionModel.summary())

    return interactionModel


def nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds, runmethod, FLAGS, dataset):
    test_set, outer_train_sets = dataset.read_sets(FLAGS)

    foldinds = len(outer_train_sets)

    test_sets = []
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []

    # logger.info('Start training')
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

    bestparamind, best_param_list, bestperf, all_predictions_not_need, losses_not_need = general_nfold_cv(XD, XT, Y,
                                                                                                          label_row_inds,
                                                                                                          label_col_inds,
                                                                                                          runmethod,
                                                                                                          FLAGS,
                                                                                                          train_sets,
                                                                                                          val_sets)

    print("Test Set len", str(len(test_set)))
    print("Outer Train Set len", str(len(outer_train_sets)))
    bestparam, best_param_list, bestperf, all_predictions, all_losses = general_nfold_cv(XD, XT, Y, label_row_inds,
                                                                                         label_col_inds,
                                                                                         runmethod, FLAGS,
                                                                                         train_sets, test_sets)

    logging("---FINAL RESULTS-----", FLAGS)
    logging("best param index = %s,  best param = %.5f" %
            (bestparamind, bestparam), FLAGS)

    testperfs = []
    testloss = []

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


def general_nfold_cv(XD, XT, Y, label_row_inds, label_col_inds, runmethod, FLAGS, labeled_sets,
                     val_sets):  ## BURAYA DA FLAGS LAZIM????

    paramset1 = FLAGS.num_windows
    paramset2 = FLAGS.smi_window_lengths
    epoch = FLAGS.num_epoch
    batchsz = FLAGS.batch_size

    logging("---Parameter Search-----", FLAGS)

    w = len(val_sets)
    h = len(paramset1) * len(paramset2)

    all_predictions = [[0 for x in range(w)] for y in range(h)]
    all_losses = [[0 for x in range(w)] for y in range(h)]
    print(all_predictions)

    for foldind in range(len(val_sets)):
        valinds = val_sets[foldind]
        labeledinds = labeled_sets[foldind]

        trrows = label_row_inds[labeledinds]
        trcols = label_col_inds[labeledinds]

        train_drugs, train_prots, train_Y = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        terows = label_row_inds[valinds]
        tecols = label_col_inds[valinds]

        val_drugs, val_prots, val_Y = prepare_interaction_pairs(XD, XT, Y, terows, tecols)

        pointer = 0

        for param1ind in range(len(paramset1)):  # hidden neurons
            param1value = paramset1[param1ind]
            for param2ind in range(len(paramset2)):  # learning rate
                param2value = paramset2[param2ind]

                gridmodel = runmethod(FLAGS, param1value, param2value)
                es = EarlyStopping(monitor='val_cindex_score', mode='max', verbose=1, patience=300,
                                   restore_best_weights=True, )
                gridres = gridmodel.fit(([np.array(train_drugs), np.array(train_prots)]), np.array(train_Y),
                                        batch_size=batchsz, epochs=epoch,
                                        validation_data=(
                                            ([np.array(val_drugs), np.array(val_prots)]), np.array(val_Y)),
                                        shuffle=False, callbacks=[es])

                predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots)])

                lst = gridres.history['val_cindex_score']
                lst2 = gridres.history['val_loss']
                rperf = max(lst)
                index = lst.index(rperf)
                loss = lst2[index]

                rm2 = get_rm2(val_Y, predicted_labels)

                predicted_labels = predicted_labels.tolist()
                labels = []
                for i in range(len(predicted_labels)):
                    labels.append(predicted_labels[i][0])
                
                fpath = FLAGS.dataset_path
                if fpath=='data/kiba/':
                    thresh = 12.1
                else:
                    thresh = 7
              
                for i in range(len(val_Y)):
                    if (val_Y[i] > thresh):
                        val_Y[i] = 1
                    else:
                        val_Y[i] = 0

                aupr = average_precision_score(val_Y, labels)

                logging(
                    "P1 = %d,  P2 = %d, Fold = %d, CI = %f, MSE = %f, aupr = %f , r2m = %f" %
                    (param1ind, param2ind, foldind, rperf, loss, aupr, rm2), FLAGS)

                df = pd.DataFrame(list(zip(labels, val_Y)),
                                  columns=['Predicted', 'Measured'])

                x = sns.relplot(data=df, x="Predicted", y="Measured", s=20)
                x.fig.set_size_inches(6.6, 5.5)
                plt.savefig("figures/scatter.tiff", bbox_inches='tight', pad_inches=0.5, dpi=300,
                            pil_kwargs={"compression": "tiff_lzw"})
                plt.close()

                plotLoss(gridres, param1ind, param2ind, foldind)

                all_predictions[pointer][foldind] = rperf  # TODO FOR EACH VAL SET allpredictions[pointer][foldind]
                all_losses[pointer][foldind] = loss

                pointer += 1

    bestperf = -float('Inf')
    bestpointer = None

    best_param_list = []
    ##Take average according to folds, then chooose best params
    pointer = 0
    for param1ind in range(len(paramset1)):
        for param2ind in range(len(paramset2)):

                avgperf = 0.
                for foldind in range(len(val_sets)):
                    foldperf = all_predictions[pointer][foldind]
                    avgperf += foldperf
                avgperf /= len(val_sets)
                if avgperf > bestperf:
                    bestperf = avgperf
                    bestpointer = pointer
                    best_param_list = [param1ind, param2ind]

                pointer += 1

    return bestpointer, best_param_list, bestperf, all_predictions, all_losses


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf2.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g / f)  # select


def plotLoss(history, batchind, epochind, foldind):
    figname = "b" + str(batchind) + "_e" + str(epochind) + "_" + str(foldind) + "_" + str(
        time.time())
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.savefig("figures/" + figname + ".png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
    plt.close()

    ## PLOT CINDEX
    plt.figure()
    plt.ylabel('c_index(CI)')
    plt.xlabel('Epoch')
    plt.plot(history.history['cindex_score'])
    plt.plot(history.history['val_cindex_score'])
    plt.legend(['train_c_index', 'val_c_index'], loc='lower right')
    plt.savefig("figures/" + figname + "_acc.png", dpi=None, facecolor='w', edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None)
    plt.close()


def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    affinity = []

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def experiment(FLAGS, deepmethod, foldcount=6):  # 5-fold cross validation + test

    # Input
    # XD: [drugs, features] sized array (features may also be similarities with other drugs
    # XT: [targets, features] sized array (features may also be similarities with other targets
    # Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    # perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    # higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    # foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation

    dataset = DataSet(fpath=FLAGS.dataset_path,
                      setting_no=FLAGS.problem_type,
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)

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

    label_row_inds, label_col_inds = np.where(
        np.isnan(Y) == False)  # basically finds the point address of affinity [x,y]

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    print(FLAGS.log_dir)
    S1_avgperf, S1_avgloss, S1_teststd = nfold_1_2_3_setting_sample(XD, XT, Y, label_row_inds, label_col_inds,
                                                                    deepmethod, FLAGS, dataset)

    logging("Setting " + str(FLAGS.problem_type), FLAGS)
    logging("avg_perf = %.5f,  avg_mse = %.5f, std = %.5f" %
            (S1_avgperf, S1_avgloss, S1_teststd), FLAGS)


def run_regression(FLAGS):
    deepmethod = build_combined_categorical1

    experiment(FLAGS, deepmethod)


if __name__ == "__main__":
    FLAGS = argparser()
    FLAGS.log_dir = FLAGS.log_dir + str(time.time()) + "/"

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging(str(FLAGS), FLAGS)
    run_regression(FLAGS)
