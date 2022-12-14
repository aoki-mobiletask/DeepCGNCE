import model
import torch
from torch import cuda
from constants import *
import numpy as np

device = torch.device("cuda:0" if cuda.is_available() else "cpu")

torch.set_num_threads(4)

FDIR = "/home/io/workspace/database/SCOPe2.08/data_mined_by_rosetta/cgFeatures/"
PDIR = "/home/io/workspace/database/SCOPe2.08/data_mined_by_rosetta/cgPredictions_5083/"
LDIR = "/home/io/workspace/database/SCOPe2.08/data_mined_by_rosetta/cgLabels/"

modelMpnn = model.Model(archSeqEnc, archGraphEnc, archRO).to(device)
modelMpnn.config(paramsSavePath)

with open("list") as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    line = line[:7]
    xSeq = torch.tensor(np.load(FDIR+line+"_caOnehotPosSeq.npy"),
                        dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    kListSeq = torch.tensor(
        np.load(FDIR+line+"_caKListSeq.npy"), dtype=torch.long)
    envmatSeq = torch.tensor(
        np.load(FDIR+line+"_caEnvmatSeq.npy"), dtype=torch.float32)
    dismat = torch.tensor(np.load(FDIR+line+"_caDismat.npy"),
                          dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    econmat_pred = modelMpnn.forward(
        xSeq, kListSeq, envmatSeq, dismat).to("cpu")
    econmat_pred = econmat_pred.detach().numpy()[0, 0]
    econmat = np.load(LDIR+line+"_caEcon.npy")[:, :, 0]
    corrcoef = np.corrcoef([np.reshape(econmat, -1),
                            np.reshape(econmat_pred, -1)])[0, 1]
    print("%s,%d/%d,%.4f" % (line, i+1, len(lines), corrcoef))
    np.save(PDIR+line+"_pred.npy", econmat_pred)
