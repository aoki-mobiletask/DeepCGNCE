import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import model
from constants import *

# drawPics
IN_FEATUREDIR = inFeatureDirLocal
IN_LABELDIR = inLabelDirLocal

with open(prolist) as file:
    lines = file.readlines()

x1d = []
x2d = []
y = []

mean, meansquare, eleCount = 0., 0., 0
for i, line in enumerate(lines):
    line = line[:line.find("_min")]
    onehot = np.load(IN_FEATUREDIR+line+"_caOneHot.npy")
    f1d = tf.convert_to_tensor(np.array(np.where(onehot > 0))[
                               1:, :], dtype=tf.uint8)
    x1d.append(f1d)

    dismat = np.load(IN_FEATUREDIR+line+"_caDisMat.npy")
    L = dismat.shape[1]
    f2dmat = dismat[:, :, :, 0]
    mean = np.sum(f2dmat)/(eleCount+L**2)+mean*(eleCount/(eleCount+L**2))
    meansquare = np.sum(f2dmat**2)/(eleCount+L**2) + \
        meansquare*(eleCount/(eleCount+L**2))
    eleCount += L**2
    x2d.append(tf.convert_to_tensor(f2dmat[np.newaxis, ...], dtype=tf.float32))

    econ = np.load(IN_LABELDIR+line+"_caEcon.npy")
    y.append(tf.convert_to_tensor(econ[np.newaxis, :], dtype=tf.float32))

stddev = np.math.sqrt(meansquare-mean**2)
for i in range(len(x2d)):
    x2d[i] = (x2d[i]-mean)/stddev

model = model.Model(input_nchannels, nblocks1d, arch1d, nblocks2d,  arch2d, output_nchannels,
                    optimizer, isDebugging, showMeanStd,
                    nepochs, patience, lr, batch_size,
                    moment, clip_grad, clip_lambda, verbose)
model.config(paramSavePath)

econ_range = [[-120., -15.], [0., 50.]]


def inrange_crds(econmat) -> set:
    ret: set = set()
    for econ_min, econ_max in econ_range:
        xx, yy = np.where(econmat > econ_min)
        tmpset1 = {(x, y) for x, y in zip(xx, yy)}
        xx, yy = np.where(econmat < econ_max)
        tmpset2 = {(x, y) for x, y in zip(xx, yy)}
        ret |= (tmpset1 & tmpset2)
    return ret


restypeCount = np.zeros((20, 20))
resseqCount = np.zeros((300, 300))
resoffCount = np.zeros((300))
restypeCountTotal = np.zeros((20, 20))
resseqCountTotal = np.zeros((300, 300))
resoffCountTotal = np.zeros((300))

predSeq, trueSeq = {}, {}
predType, trueType = {}, {}
pearsonType = np.zeros((20, 20))
pearsonSeq = np.zeros((300, 300))
predOff, trueOff = {}, {}
pearsonOff = np.zeros((300))


for (f1d, f2d, econ) in zip(x1d, x2d, y):
    econP = model.forwardPropagation(f1d, f2d)
    f1d = f1d[0].numpy()
    econ = econ[0, :, :, 0]
    econP = econP[0, :, :, 0]
    for x in range(len(f1d)):
        for y in range(len(f1d)):
            resseqCountTotal[x, y] += 1
            iresx, iresy = f1d[x], f1d[y]
            restypeCountTotal[iresx, iresy] += 1
            off = np.abs(x-y)
            resoffCountTotal[off] += 1
            if (x, y) in predSeq.keys():
                predSeq[(x, y)].append(econP[x, y])
            else:
                predSeq[(x, y)] = [econP[x, y]]
            if (x, y) in trueSeq.keys():
                trueSeq[(x, y)].append(econ[x, y])
            else:
                trueSeq[(x, y)] = [econ[x, y]]
            if (iresx, iresy) in predType.keys():
                predType[(iresx, iresy)].append(econP[x, y])
            else:
                predType[(iresx, iresy)] = [econP[x, y]]
            if (iresx, iresy) in trueType.keys():
                trueType[(iresx, iresy)].append(econ[x, y])
            else:
                trueType[(iresx, iresy)] = [econ[x, y]]
            if off in predOff.keys():
                predOff[off].append(econP[x, y])
            else:
                predOff[off] = [econP[x, y]]
            if off in trueOff.keys():
                trueOff[off].append(econ[x, y])
            else:
                trueOff[off] = [econ[x, y]]
    crds = inrange_crds(econ)
    for (x, y) in crds:
        resseqCount[x, y] += 1
        iresx, iresy = f1d[x], f1d[y]
        restypeCount[iresx, iresy] += 1
        off = np.abs(x-y)
        resoffCount[off] += 1


resseqCount /= resseqCountTotal
restypeCount /= restypeCountTotal
resoffCount /= resoffCountTotal

for x in range(300):
    for y in range(300):
        if (x, y) in predSeq.keys():
            pearsonSeq[x, y] = np.corrcoef(
                predSeq[(x, y)], trueSeq[(x, y)])[0, 1]
        else:
            pearsonSeq[x, y] = 0
for x in range(20):
    for y in range(20):
        pearsonType[x, y] = np.corrcoef(
            predType[(x, y)], trueType[(x, y)])[0, 1]
for d in range(300):
    if d in predOff.keys():
        pearsonOff[d] = np.corrcoef(predOff[d], trueOff[d])[0, 1]
    else:
        pearsonOff[d] = 0


natural_AA1C = ['G', 'A', 'V', 'P', 'L', 'I', 'F', 'S', 'T',
                'Y', 'D', 'N', 'E', 'Q', 'C', 'M', 'R', 'H', 'Q', 'K']
fig, ax = plt.subplots(2, 3, figsize=(14, 8))
norm = mpl.colors.Normalize(vmin=0., vmax=0.3, clip=False)
im00 = ax[0, 0].imshow(resseqCount, cmap="Greys", norm=norm)
plt.colorbar(im00, ax=ax[0, 0])
im01 = ax[0, 1].imshow(restypeCount, cmap="coolwarm")
ax[0, 1].set_xticks(ticks=np.arange(len(natural_AA1C)), labels=natural_AA1C)
ax[0, 1].set_yticks(ticks=np.arange(len(natural_AA1C)), labels=natural_AA1C)
plt.colorbar(im01, ax=ax[0, 1])
im02 = ax[0, 2].plot(resoffCount)
norm = mpl.colors.Normalize(vmin=0., vmax=1, clip=False)
im10 = ax[1, 0].imshow(pearsonSeq, cmap="Greys", norm=norm)
plt.colorbar(im10, ax=ax[1, 0])
im11 = ax[1, 1].imshow(pearsonType, cmap="coolwarm")
ax[1, 1].set_xticks(ticks=np.arange(len(natural_AA1C)), labels=natural_AA1C)
ax[1, 1].set_yticks(ticks=np.arange(len(natural_AA1C)), labels=natural_AA1C)
plt.colorbar(im11, ax=ax[1, 1])
im12 = ax[1, 2].plot(pearsonOff)
plt.show()
