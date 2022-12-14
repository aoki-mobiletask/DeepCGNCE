import model
from constants import *
import numpy as np
import torch
from torch import cuda

device = torch.device("cuda:0" if cuda.is_available() else "cpu")

torch.set_num_threads(4)

inFeatureDir = INDIRFEATURES_remote
inLabelDir = INDIRLABEL_remote

# data preparation
with open(proList) as file:
    lines = file.readlines()
pros = [line[:7] for line in lines]

# # data readin
xSeqN, kListSeqN, envmatSeqN, dismatN, econmatN = [], [], [], [], []
for pro in pros:
    xSeq = torch.tensor(np.load(inFeatureDir+pro+"_caOnehotPosSeq.npy"),
                        dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    xSeqN.append(xSeq)
    kListSeq = torch.tensor(
        np.load(inFeatureDir+pro+"_caKListSeq.npy"), dtype=torch.long)
    kListSeqN.append(kListSeq)
    envmatSeq = torch.tensor(
        np.load(inFeatureDir+pro+"_caEnvmatSeq.npy"), dtype=torch.float32)
    envmatSeqN.append(envmatSeq)
    dismat = torch.tensor(np.load(inFeatureDir+pro+"_caDismat.npy"),
                          dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    dismatN.append(dismat)
    econmat = torch.tensor(np.load(inLabelDir+pro+"_caEcon.npy"),
                           dtype=torch.float32)[:, :, 0].unsqueeze(0).unsqueeze(0)
    econmatN.append(econmat)

# # data shuffle
np.random.seed(shuffleSeed)
np.random.shuffle(pros)
np.random.seed(shuffleSeed)
np.random.shuffle(xSeqN)
np.random.seed(shuffleSeed)
np.random.shuffle(kListSeqN)
np.random.seed(shuffleSeed)
np.random.shuffle(envmatSeqN)
np.random.seed(shuffleSeed)
np.random.shuffle(dismatN)
np.random.seed(shuffleSeed)
np.random.shuffle(econmatN)

# # data train test split
N = len(pros)
train_valid_bound, valid_test_bound = int(N*0.8), int(N*0.9)
train_pros, valid_pros, test_pros = pros[:train_valid_bound], pros[
    train_valid_bound:valid_test_bound], pros[valid_test_bound:]
train_xSeqN, valid_xSeqN, test_xSeqN = xSeqN[:train_valid_bound], xSeqN[
    train_valid_bound:valid_test_bound], xSeqN[valid_test_bound:]
train_kListSeqN, valid_kListSeqN, test_kListSeqN = kListSeqN[:train_valid_bound], kListSeqN[
    train_valid_bound:valid_test_bound], kListSeqN[valid_test_bound:]
train_envmatSeqN, valid_envmatSeqN, test_envmatSeqN = envmatSeqN[:train_valid_bound], envmatSeqN[
    train_valid_bound:valid_test_bound], envmatSeqN[valid_test_bound:]
train_dismatN, valid_dismatN, test_dismatN = dismatN[:train_valid_bound], dismatN[
    train_valid_bound:valid_test_bound], dismatN[valid_test_bound:]
train_econmatN, valid_econmatN, test_econmatN = econmatN[:train_valid_bound], econmatN[
    train_valid_bound:valid_test_bound], econmatN[valid_test_bound:]


# model init
modelMpnn = model.Model(archSeqEnc, archGraphEnc, archRO).to(device)
paramsLoadPath = paramsSavePath if doLoadParams else ""
modelMpnn.config(paramsLoadPath)

# # train
if doTrain:
    np.random.seed()
    iFilm = np.random.randint(len(valid_pros))
    sFilm = valid_pros[iFilm]
    modelMpnn.fit(train_xSeqN, train_kListSeqN, train_envmatSeqN, train_dismatN, train_econmatN,
                  valid_xSeqN, valid_kListSeqN, valid_envmatSeqN, valid_dismatN, valid_econmatN,
                  batch_size, nepoch, patience, lr_max, lr_min, grad_norm_cutoff,
                  paramsSavePath=paramsSavePath, historySaveDir=historySaveDir, filmsSaveDir=filmsSaveDir, iFilm=iFilm, sFilm=sFilm, verbose=verbose)

# # evaluate
if doEvaluate:
    modelMpnn.evaluate(train_xSeqN, train_dismatN, train_econmatN, kListSeqN=train_kListSeqN,
                       envmatSeqN=train_envmatSeqN, label="train", savetoDir=historySaveDir)
    modelMpnn.evaluate(valid_xSeqN, valid_dismatN, valid_econmatN, kListSeqN=valid_kListSeqN,
                       envmatSeqN=valid_envmatSeqN, label="valid", savetoDir=historySaveDir)
    modelMpnn.evaluate(test_xSeqN, test_dismatN, test_econmatN, kListSeqN=test_kListSeqN,
                       envmatSeqN=test_envmatSeqN, label="test", savetoDir=historySaveDir)

# # visualize
while(nVisualize):
    np.random.seed()
    iFilm = np.random.randint(len(test_pros))
    sFilm = test_pros[iFilm]
    xSeq, kListSeq, envmatSeq, dismat, econmat = test_xSeqN[iFilm], test_kListSeqN[
        iFilm], test_envmatSeqN[iFilm], test_dismatN[iFilm], test_econmatN[iFilm]
    modelMpnn.visualize(xSeq, dismat, econmat, kListSeq,
                        envmatSeq, clip_min, clip_max, label=sFilm, savetoDir="")
    nVisualize -= 1
