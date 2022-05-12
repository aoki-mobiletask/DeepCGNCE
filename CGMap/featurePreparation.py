import ProCG
import numpy as np

IN_PDBDIR = "../pdbEminimized/"
OUT_FEATUREDIR = "../cgFeatures/"

with open("list") as f:
    lines = f.readlines()

PRO_NUM = len(lines)
for ipro, line in enumerate(lines):
    spro = line[:line.find("_min")]
    pro = ProCG.Pro(IN_PDBDIR+spro+"_min.pdb")
    proCA = pro.pdbMap("CAlpha.cgin")
    onehot1d = proCA.encodingOneHot(1, "CAlpha.cgin", "RES")[' ']
    massCharge1d = proCA.encodingPhyProperty(1, ["mass", "charge"], "RES")[' ']
    np.save(OUT_FEATUREDIR+spro+"_caOneHot.npy", onehot1d)
    np.save(OUT_FEATUREDIR+spro+"_caMassCharge.npy", massCharge1d)
    print("\rprocessing struture %s" % spro)
    print("\r{:.2%}".format(ipro/PRO_NUM))
print("completed")
