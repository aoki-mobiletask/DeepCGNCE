import numpy as np
import time
import model
import layers
import tensorflow as tf
from constants import *

# datapreparation
IN_FEATUREDIR = inFeatureDirLocal
IN_LABELDIR = inLabelDirLocal

with open(prolist) as file:
    lines = file.readlines()

x1d = []
x2d = []
y = []

for i, line in enumerate(lines):
    line = line[:line.find("_min")]
    onehot = np.load(IN_FEATUREDIR+line+"_caOneHot.npy")
    f1d = tf.convert_to_tensor(np.array(np.where(onehot > 0))[
                               1:, :], dtype=tf.uint8)
    x1d.append(f1d)

    dismat = np.load(IN_FEATUREDIR+line+"_caDisMat.npy")
    f2dmat = dismat[:, :, :, 0]
    x2d.append(tf.convert_to_tensor(f2dmat[np.newaxis, ...], dtype=tf.float32))

    econ = np.load(IN_LABELDIR+line+"_caEcon.npy")
    y.append(tf.convert_to_tensor(econ[np.newaxis, :], dtype=tf.float32))

SEED = 10
np.random.seed(SEED)
np.random.shuffle(lines)
np.random.seed(SEED)
np.random.shuffle(x1d)
np.random.seed(SEED)
np.random.shuffle(x2d)
np.random.seed(SEED)
np.random.shuffle(y)

trVBound, vTeBound = int(len(lines)*0.8), int(len(lines)*0.9)
x1d_train, x1d_valid, x1d_test = x1d[:
                                     trVBound], x1d[trVBound:vTeBound], x1d[vTeBound:]
x2d_train, x2d_valid, x2d_test = x2d[:
                                     trVBound], x2d[trVBound:vTeBound], x2d[vTeBound:]
y_train, y_valid, y_test = y[:trVBound], y[trVBound:vTeBound], y[vTeBound:]

mean, meansquare, eleCount = 0., 0., 0
for f2d in x2d_train+x2d_valid:
    f2dmat = f2d[0,:,:,0]
    L = f2dmat.shape[0]
    mean = np.sum(f2dmat)/(eleCount+L**2)+mean*(eleCount/(eleCount+L**2))
    meansquare = np.sum(f2dmat**2)/(eleCount+L**2) + \
        meansquare*(eleCount/(eleCount+L**2))
    eleCount += L**2
stddev = np.math.sqrt(meansquare-mean**2)

# training
model = model.Model(input_nchannels, nblocks1d, arch1d, mean, stddev,
                    nblocks2d,  arch2d, output_nchannels,
                    optimizer, isDebugging, showMeanStd,
                    nepochs, patience, lr, batch_size,
                    moment, clip_grad, clip_lambda, verbose)

if doShowModelSize:
    layers.countsize = 0.
if doLoadParams:
    model.config(paramSavePath)
else:
    model.config()
if doShowModelSize:
    print(layers.countsize)

if doTrain:
    np.random.seed(int(time.time()))
    iProFilm = np.random.randint(len(y_valid))
    sProFilm = lines[len(y_train)+iProFilm]
    sProFilm = sProFilm[:sProFilm.find("_min")]
    model.fit((x1d_train, x2d_train, y_train), (x1d_valid, x2d_valid, y_valid),
              saveParamTo=paramSavePath, saveHistoryTo=historySavePath,
              saveFilmsTo=filmSavePath, iProFilm=iProFilm, sProFilm=sProFilm)

if doEvaluate:
    model.evaluate((x1d_train, x2d_train, y_train),
                   saveHistogramTo=historySavePath, label="train")
    model.evaluate((x1d_valid, x2d_valid, y_valid),
                   saveHistogramTo=historySavePath, label="valid")
    model.evaluate((x1d_test, x2d_test, y_test),
                   saveHistogramTo=historySavePath, label="test")

while nVisualization > 0:
    # randomly choose a datapoint
    np.random.seed(int(time.time()))
    ipro = np.random.randint(len(lines))
    spro = lines[ipro]
    spro = spro[:spro.find("_min")]
    if ipro < len(y_train):
        x1d = x1d_train[ipro]
        x2d = x2d_train[ipro]
        y = y_train[ipro]
    elif ipro < len(y_train)+len(y_valid):
        x1d = x1d_valid[ipro-len(y_train)]
        x2d = x2d_valid[ipro-len(y_train)]
        y = y_valid[ipro-len(y_train)]
    else:
        x1d = x1d_test[ipro-len(y_train)-len(y_valid)]
        x2d = x2d_test[ipro-len(y_train)-len(y_valid)]
        y = y_test[ipro-len(y_train)-len(y_valid)]
    model.visualization(
        (x1d, x2d, y), spro, visualization_clip[0], visualization_clip[1])
    nVisualization -= 1
