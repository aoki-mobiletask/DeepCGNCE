from os import access
import sys
import re
import copy
import numpy as np


''' # it's a simple example on how to use this module
# create a empty pdb structure
pro = Pro()
# read in a .pdb file
pro.readPdb("pro.pdb")
# check it with output-form of a .cgin file
pro.pdbCheck("pdbCheck.cgin", outform=True)
# do martini CG map
proCG = pro.pdbMap("CAlpha.cgin")
# write CG model in a pdb-like format
proCG.writePdb("pro_out.pdb")
# compute res-bead mode distance matrix
proCG.dismat(1, "RES-BEAD")
# compute bead-bead mode distance matrix
proCG.dismat(1, "BEAD-BEAD")
# onehot encoding residues according to .cgin file
proCG.encodingOneHot(1, "CAlpha.cgin", "RES")
# onehot encoding beads according to .cgin file
proCG.encodingOneHot(1, "CAlpha.cgin", "BEAD")
# read amber contact energy record and return a numpy energy matrix
econ = proCG.readAmberEcon(1, "CAEcon.para", "RES-BEAD", {'1': ' '})
'''

# assume changeline is '\n', not '\r\n'
PATTERNS = {
    "atom/bead": re.compile(
        r"(ATOM|BEAD)  ([ \-\d]{5}) ([ \w]{4}).[A-Z]{3} [ A-Z][ \-\d]{4}.   "
        r"([ \-\d]{4}\.\d{3})([ \-\d]{4}\.\d{3})([ \-\d]{4}\.\d{3})([ \d]{3}\.\d{2})([ \d]{3}\.\d{2})"
        r"([ \d]{3}\.\d{2}| {6}).{4}([ a-zA-Z]{2})([ \d+\-]{2})"
    ),
    "cgin": re.compile(
        r"(\w*)[\s\S]*RESIDUE([\s\S]*)BEAD([\s\S]*)MASS([\s\S]*)COORDINATE([\s\S]*)CG_MODE([\s\S]*)"
        r"BEAD_CHARGE([\s\S]*)BEAD_CHARGENT([\s\S]*)BEAD_CHARGECT([\s\S]*)CHARGEMODE([\s\S]*)"
        r"VARIABLES([\s\S]*$)"
    ),
    "amberECon": re.compile(
        r"[ \d]{5}([ \w]{5})([ \w]{5})([ \d]{5})([ \d]{5}) (\w{3})  (\w{3})([ \w]{5})([ \w]{5})([ \-\d]{7}\.\d{3})"
    )
}

DATA = {
    "atom_mass": {'H': 1.01, 'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00, 'Si': 28.09, 'P': 30.97, 'S': 32.07, 'Se': 78.96, },
    "amino12_charge": {
        'ALA': {
            'N': -0.4157, 'H': 0.2719, 'CA': 0.0337, 'HA': 0.0823,
            'CB': -0.1825, 'HB1': 0.0603, 'HB2': 0.0603, 'HB3': 0.0603,
            'C': 0.5973, 'O': -0.5679
        },
        'ARG': {
            'N': -0.3479, 'H': 0.2747, 'CA': -0.2637, 'HA': 0.156,
            'CB': -0.0007, 'HB2': 0.0327, 'HB3': 0.0327,
            'CG': 0.039, 'HG2': 0.0285, 'HG3': 0.0285,
            'CD': 0.0486, 'HD2': 0.0687, 'HD3': 0.0687,
            'NE': -0.5295, 'HE': 0.3456, 'CZ': 0.8076,
            'NH1': -0.8627, 'HH11': 0.4478, 'HH12': 0.4478,
            'NH2': -0.8627, 'HH21': 0.4478, 'HH22': 0.4478,
            'C': 0.7341, 'O': -0.5894
        },
        'ASH': {
            'N': -0.4157, 'H': 0.2719, 'CA': 0.0341, 'HA': 0.0864,
            'CB': -0.0316, 'HB2': 0.0488, 'HB3': 0.0488,
            'CG': 0.6462, 'OD1': -0.5554, 'OD2': -0.6376, 'HD2': 0.4747,
            'C': 0.5973, 'O': -0.5679
        },
        'ASN': {
            'N': -0.4157, 'H': 0.2719, 'CA': 0.0143, 'HA': 0.1048,
            'CB': -0.2041, 'HB2': 0.0797, 'HB3': 0.0797,
            'CG': 0.713, 'OD1': -0.5931, 'ND2': -0.9191, 'HD21': 0.4196, 'HD22': 0.4196,
            'C': 0.5973, 'O': -0.5679
        },
        'ASP': {
            'N': -0.5163, 'H': 0.2936, 'CA': 0.0381, 'HA': 0.088,
            'CB': -0.0303, 'HB2': -0.0122, 'HB3': -0.0122,
            'CG': 0.7994, 'OD1': -0.8014, 'OD2': -0.8014,
            'C': 0.5366, 'O': -0.5819
        },
        'CYM': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0351, 'HA': 0.0508,
            'CB': -0.2413, 'HB3': 0.1122, 'HB2': 0.1122, 'SG': -0.8844,
            'C': 0.5973, 'O': -0.5679
        },
        'CYS': {
            'N': -0.4157, 'H': 0.2719, 'CA': 0.0213, 'HA': 0.1124,
            'CB': -0.1231, 'HB2': 0.1112, 'HB3': 0.1112, 'SG': -0.3119, 'HG': 0.1933,
            'C': 0.5973, 'O': -0.5679
        },
        'CYX': {
            'N': -0.4157, 'H': 0.2719, 'CA': 0.0429, 'HA': 0.0766,
            'CB': -0.079, 'HB2': 0.091, 'HB3': 0.091, 'SG': -0.1081,
            'C': 0.5973, 'O': -0.5679
        },
        'GLH': {
            'N': -0.4157, 'H': 0.2719, 'CA': 0.0145, 'HA': 0.0779,
            'CB': -0.0071, 'HB2': 0.0256, 'HB3': 0.0256,
            'CG': -0.0174, 'HG2': 0.043, 'HG3': 0.043,
            'CD': 0.6801, 'OE1': -0.5838, 'OE2': -0.6511, 'HE2': 0.4641,
            'C': 0.5973, 'O': -0.5679
        },
        'GLN': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0031, 'HA': 0.085,
            'CB': -0.0036, 'HB2': 0.0171, 'HB3': 0.0171,
            'CG': -0.0645, 'HG2': 0.0352, 'HG3': 0.0352,
            'CD': 0.6951, 'OE1': -0.6086, 'NE2': -0.9407, 'HE21': 0.4251, 'HE22': 0.4251,
            'C': 0.5973, 'O': -0.5679
        },
        'GLU': {
            'N': -0.5163, 'H': 0.2936, 'CA': 0.0397, 'HA': 0.1105,
            'CB': 0.056, 'HB2': -0.0173, 'HB3': -0.0173,
            'CG': 0.0136, 'HG2': -0.0425, 'HG3': -0.0425,
            'CD': 0.8054, 'OE1': -0.8188, 'OE2': -0.8188,
            'C': 0.5366, 'O': -0.5819
        },
        'GLY': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0252, 'HA2': 0.0698, 'HA3': 0.0698,
            'C': 0.5973, 'O': -0.5679
        },
        'HID': {
            'N': -0.4157, 'H': 0.2719, 'CA': 0.0188, 'HA': 0.0881,
            'CB': -0.0462, 'HB2': 0.0402, 'HB3': 0.0402,
            'CG': -0.0266, 'ND1': -0.3811, 'HD1': 0.3649,
            'CE1': 0.2057, 'HE1': 0.1392, 'NE2': -0.5727,
            'CD2': 0.1292, 'HD2': 0.1147,
            'C': 0.5973, 'O': -0.5679
        },
        'HIE': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0581, 'HA': 0.136,
            'CB': -0.0074, 'HB2': 0.0367, 'HB3': 0.0367,
            'CG': 0.1868, 'ND1': -0.5432, 'CE1': 0.1635, 'HE1': 0.1435,
            'NE2': -0.2795, 'HE2': 0.3339,
            'CD2': -0.2207, 'HD2': 0.1862,
            'C': 0.5973, 'O': -0.5679
        },
        'HIP': {
            'N': -0.3479, 'H': 0.2747, 'CA': -0.1354, 'HA': 0.1212,
            'CB': -0.0414, 'HB2': 0.081, 'HB3': 0.081, 'CG': -0.0012,
            'ND1': -0.1513, 'HD1': 0.3866, 'CE1': -0.017, 'HE1': 0.2681,
            'NE2': -0.1718, 'HE2': 0.3911, 'CD2': -0.1141, 'HD2': 0.2317,
            'C': 0.7341, 'O': -0.5894
        },
        'HYP': {
            'N': -0.2548, 'CD': 0.0595, 'HD22': 0.07, 'HD23': 0.07,
            'CG': 0.04, 'HG': 0.0416, 'OD1': -0.6134, 'HD1': 0.3851,
            'CB': 0.0203, 'HB2': 0.0426, 'HB3': 0.0426,
            'CA': 0.0047, 'HA': 0.077, 'C': 0.5896, 'O': -0.5748
        },
        'ILE': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0597, 'HA': 0.0869,
            'CB': 0.1303, 'HB': 0.0187,
            'CG2': -0.3204, 'HG21': 0.0882, 'HG22': 0.0882, 'HG23': 0.0882,
            'CG1': -0.043, 'HG12': 0.0236, 'HG13': 0.0236,
            'CD1': -0.066, 'HD11': 0.0186, 'HD12': 0.0186, 'HD13': 0.0186,
            'C': 0.5973, 'O': -0.5679
        },
        'LEU': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0518, 'HA': 0.0922,
            'CB': -0.1102, 'HB2': 0.0457, 'HB3': 0.0457,
            'CG': 0.3531, 'HG': -0.0361,
            'CD1': -0.4121, 'HD11': 0.1, 'HD12': 0.1, 'HD13': 0.1,
            'CD2': -0.4121, 'HD21': 0.1, 'HD22': 0.1, 'HD23': 0.1,
            'C': 0.5973, 'O': -0.5679
        },
        'LYN': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.07206, 'HA': 0.0994,
            'CB': -0.04845, 'HB2': 0.034, 'HB3': 0.034,
            'CG': 0.06612, 'HG2': 0.01041, 'HG3': 0.01041,
            'CD': -0.03768, 'HD2': 0.01155, 'HD3': 0.01155,
            'CE': 0.32604, 'HE2': -0.03358, 'HE3': -0.03358,
            'NZ': -1.03581, 'HZ2': 0.38604, 'HZ3': 0.38604,
            'C': 0.5973, 'O': -0.5679
        },
        'LYS': {
            'N': -0.3479, 'H': 0.2747, 'CA': -0.24, 'HA': 0.1426,
            'CB': -0.0094, 'HB2': 0.0362, 'HB3': 0.0362,
            'CG': 0.0187, 'HG2': 0.0103, 'HG3': 0.0103,
            'CD': -0.0479, 'HD2': 0.0621, 'HD3': 0.0621,
            'CE': -0.0143, 'HE2': 0.1135, 'HE3': 0.1135,
            'NZ': -0.3854, 'HZ1': 0.34, 'HZ2': 0.34, 'HZ3': 0.34,
            'C': 0.7341, 'O': -0.5894
        },
        'MET': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0237, 'HA': 0.088,
            'CB': 0.0342, 'HB2': 0.0241, 'HB3': 0.0241,
            'CG': 0.0018, 'HG2': 0.044, 'HG3': 0.044,
            'SD': -0.2737, 'CE': -0.0536, 'HE1': 0.0684, 'HE2': 0.0684, 'HE3': 0.0684,
            'C': 0.5973, 'O': -0.5679
        },
        'PHE': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0024, 'HA': 0.0978,
            'CB': -0.0343, 'HB2': 0.0295, 'HB3': 0.0295,
            'CG': 0.0118, 'CD1': -0.1256, 'HD1': 0.133,
            'CE1': -0.1704, 'HE1': 0.143, 'CZ': -0.1072, 'HZ': 0.1297,
            'CE2': -0.1704, 'HE2': 0.143, 'CD2': -0.1256, 'HD2': 0.133,
            'C': 0.5973, 'O': -0.5679
        },
        'PRO': {
            'N': -0.2548, 'CD': 0.0192, 'HD2': 0.0391, 'HD3': 0.0391,
            'CG': 0.0189, 'HG2': 0.0213, 'HG3': 0.0213,
            'CB': -0.007, 'HB2': 0.0253, 'HB3': 0.0253,
            'CA': -0.0266, 'HA': 0.0641, 'C': 0.5896, 'O': -0.5748
        },
        'SER': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0249, 'HA': 0.0843,
            'CB': 0.2117, 'HB2': 0.0352, 'HB3': 0.0352,
            'OG': -0.6546, 'HG': 0.4275,
            'C': 0.5973, 'O': -0.5679
        },
        'THR': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0389, 'HA': 0.1007,
            'CB': 0.3654, 'HB': 0.0043,
            'CG2': -0.2438, 'HG21': 0.0642, 'HG22': 0.0642, 'HG23': 0.0642,
            'OG1': -0.6761, 'HG1': 0.4102,
            'C': 0.5973, 'O': -0.5679
        },
        'TRP': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0275, 'HA': 0.1123,
            'CB': -0.005, 'HB2': 0.0339, 'HB3': 0.0339,
            'CG': -0.1415, 'CD1': -0.1638, 'HD1': 0.2062,
            'NE1': -0.3418, 'HE1': 0.3412, 'CE2': 0.138,
            'CZ2': -0.2601, 'HZ2': 0.1572, 'CH2': -0.1134, 'HH2': 0.1417,
            'CZ3': -0.1972, 'HZ3': 0.1447, 'CE3': -0.2387, 'HE3': 0.17, 'CD2': 0.1243,
            'C': 0.5973, 'O': -0.5679
        },
        'TYR': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0014, 'HA': 0.0876,
            'CB': -0.0152, 'HB2': 0.0295, 'HB3': 0.0295,
            'CG': -0.0011, 'CD1': -0.1906, 'HD1': 0.1699, 'CE1': -0.2341, 'HE1': 0.1656,
            'CZ': 0.3226, 'OH': -0.5579, 'HH': 0.3992,
            'CE2': -0.2341, 'HE2': 0.1656, 'CD2': -0.1906, 'HD2': 0.1699,
            'C': 0.5973, 'O': -0.5679
        },
        'VAL': {
            'N': -0.4157, 'H': 0.2719, 'CA': -0.0875, 'HA': 0.0969,
            'CB': 0.2985, 'HB': -0.0297,
            'CG1': -0.3192, 'HG11': 0.0791, 'HG12': 0.0791, 'HG13': 0.0791,
            'CG2': -0.3192, 'HG21': 0.0791, 'HG22': 0.0791, 'HG23': 0.0791,
            'C': 0.5973, 'O': -0.5679
        }
    },
    "aminont12_charge": {
        'ACE': {
            'H1': 0.1123, 'CH3': -0.3662, 'H2': 0.1123, 'H3': 0.1123,
            'C': 0.5972, 'O': -0.5679
        },
        'ALA': {
            'N': 0.1414, 'H1': 0.1997, 'H2': 0.1997, 'H3': 0.1997, 'CA': 0.0962, 'HA': 0.0889,
            'CB': -0.0597, 'HB1': 0.03, 'HB2': 0.03, 'HB3': 0.03,
            'C': 0.6163, 'O': -0.5722
        },
        'ARG': {
            'N': 0.1305, 'H1': 0.2083, 'H2': 0.2083, 'H3': 0.2083, 'CA': -0.0223, 'HA': 0.1242,
            'CB': 0.0118, 'HB2': 0.0226, 'HB3': 0.0226, 'CG': 0.0236, 'HG2': 0.0309, 'HG3': 0.0309,
            'CD': 0.0935, 'HD2': 0.0527, 'HD3': 0.0527, 'NE': -0.565, 'HE': 0.3592,
            'CZ': 0.8281, 'NH1': -0.8693, 'HH11': 0.4494, 'HH12': 0.4494,
            'NH2': -0.8693, 'HH21': 0.4494, 'HH22': 0.4494,
            'C': 0.7214, 'O': -0.6013
        },
        'ASN': {
            'N': 0.1801, 'H1': 0.1921, 'H2': 0.1921, 'H3': 0.1921, 'CA': 0.0368, 'HA': 0.1231,
            'CB': -0.0283, 'HB2': 0.0515, 'HB3': 0.0515, 'CG': 0.5833, 'OD1': -0.5744,
            'ND2': -0.8634, 'HD21': 0.4097, 'HD22': 0.4097,
            'C': 0.6163, 'O': -0.5722
        },
        'ASP': {
            'N': 0.0782, 'H1': 0.22, 'H2': 0.22, 'H3': 0.22, 'CA': 0.0292, 'HA': 0.1141,
            'CB': -0.0235, 'HB2': -0.0169, 'HB3': -0.0169, 'CG': 0.8194, 'OD1': -0.8084, 'OD2': -0.8084,
            'C': 0.5621, 'O': -0.5889
        },
        'CYS': {
            'N': 0.1325, 'H1': 0.2023, 'H2': 0.2023, 'H3': 0.2023, 'CA': 0.0927, 'HA': 0.1411,
            'CB': -0.1195, 'HB2': 0.1188, 'HB3': 0.1188, 'SG': -0.3298, 'HG': 0.1975,
            'C': 0.6123, 'O': -0.5713
        },
        'CYX': {
            'N': 0.2069, 'H1': 0.1815, 'H2': 0.1815, 'H3': 0.1815, 'CA': 0.1055, 'HA': 0.0922,
            'CB': -0.0277, 'HB2': 0.068, 'HB3': 0.068, 'SG': -0.0984,
            'C': 0.6123, 'O': -0.5713
        },
        'GLN': {
            'N': 0.1493, 'H1': 0.1996, 'H2': 0.1996, 'H3': 0.1996, 'CA': 0.0536, 'HA': 0.1015,
            'CB': 0.0651, 'HB2': 0.005, 'HB3': 0.005,
            'CG': -0.0903, 'HG2': 0.0331, 'HG3': 0.0331,
            'CD': 0.7354, 'OE1': -0.6133, 'NE2': -1.0031, 'HE21': 0.4429, 'HE22': 0.4429,
            'C': 0.6123, 'O': -0.5713
        },
        'GLU': {
            'N': 0.0017, 'H1': 0.2391, 'H2': 0.2391, 'H3': 0.2391, 'CA': 0.0588, 'HA': 0.1202,
            'CB': 0.0909, 'HB2': -0.0232, 'HB3': -0.0232,
            'CG': -0.0236, 'HG2': -0.0315, 'HG3': -0.0315, 'CD': 0.8087, 'OE1': -0.8189, 'OE2': -0.8189,
            'C': 0.5621, 'O': -0.5889},
        'GLY': {
            'N': 0.2943, 'H1': 0.1642, 'H2': 0.1642, 'H3': 0.1642,
            'CA': -0.01, 'HA2': 0.0895, 'HA3': 0.0895,
            'C': 0.6163, 'O': -0.5722
        },
        'HID': {
            'N': 0.1542, 'H1': 0.1963, 'H2': 0.1963, 'H3': 0.1963, 'CA': 0.0964, 'HA': 0.0958,
            'CB': 0.0259, 'HB2': 0.0209, 'HB3': 0.0209,
            'CG': -0.0399, 'ND1': -0.3819, 'HD1': 0.3632, 'CE1': 0.2127, 'HE1': 0.1385,
            'NE2': -0.5711, 'CD2': 0.1046, 'HD2': 0.1299,
            'C': 0.6123, 'O': -0.5713
        },
        'HIE': {
            'N': 0.1472, 'H1': 0.2016, 'H2': 0.2016, 'H3': 0.2016, 'CA': 0.0236, 'HA': 0.138,
            'CB': 0.0489, 'HB2': 0.0223, 'HB3': 0.0223, 'CG': 0.174, 'ND1': -0.5579,
            'CE1': 0.1804, 'HE1': 0.1397, 'NE2': -0.2781, 'HE2': 0.3324, 'CD2': -0.2349, 'HD2': 0.1963,
            'C': 0.6123, 'O': -0.5713},
        'HIP': {
            'N': 0.256, 'H1': 0.1704, 'H2': 0.1704, 'H3': 0.1704, 'CA': 0.0581, 'HA': 0.1047,
            'CB': 0.0484, 'HB2': 0.0531, 'HB3': 0.0531,
            'CG': -0.0236, 'ND1': -0.151, 'HD1': 0.3821, 'CE1': -0.0011, 'HE1': 0.2645,
            'NE2': -0.1739, 'HE2': 0.3921, 'CD2': -0.1433, 'HD2': 0.2495,
            'C': 0.7214, 'O': -0.6013
        },
        'ILE': {
            'N': 0.0311, 'H1': 0.2329, 'H2': 0.2329, 'H3': 0.2329, 'CA': 0.0257, 'HA': 0.1031,
            'CB': 0.1885, 'HB': 0.0213, 'CG2': -0.372, 'HG21': 0.0947, 'HG22': 0.0947, 'HG23': 0.0947,
            'CG1': -0.0387, 'HG12': 0.0201, 'HG13': 0.0201,
            'CD1': -0.0908, 'HD11': 0.0226, 'HD12': 0.0226, 'HD13': 0.0226,
            'C': 0.6123, 'O': -0.5713
        },
        'LEU': {
            'N': 0.101, 'H1': 0.2148, 'H2': 0.2148, 'H3': 0.2148, 'CA': 0.0104, 'HA': 0.1053,
            'CB': -0.0244, 'HB2': 0.0256, 'HB3': 0.0256,
            'CG': 0.3421, 'HG': -0.038, 'CD1': -0.4106, 'HD11': 0.098, 'HD12': 0.098, 'HD13': 0.098,
            'CD2': -0.4104, 'HD21': 0.098, 'HD22': 0.098, 'HD23': 0.098,
            'C': 0.6123, 'O': -0.5713
        },
        'LYS': {
            'N': 0.0966, 'H1': 0.2165, 'H2': 0.2165, 'H3': 0.2165, 'CA': -0.0015, 'HA': 0.118,
            'CB': 0.0212, 'HB2': 0.0283, 'HB3': 0.0283, 'CG': -0.0048, 'HG2': 0.0121, 'HG3': 0.0121,
            'CD': -0.0608, 'HD2': 0.0633, 'HD3': 0.0633, 'CE': -0.0181, 'HE2': 0.1171, 'HE3': 0.1171,
            'NZ': -0.3764, 'HZ1': 0.3382, 'HZ2': 0.3382, 'HZ3': 0.3382,
            'C': 0.7214, 'O': -0.
        },
        'MET': {
            'N': 0.1592, 'H1': 0.1984, 'H2': 0.1984, 'H3': 0.1984, 'CA': 0.0221, 'HA': 0.1116,
            'CB': 0.0865, 'HB2': 0.0125, 'HB3': 0.0125, 'CG': 0.0334, 'HG2': 0.0292, 'HG3': 0.0292,
            'SD': -0.2774, 'CE': -0.0341, 'HE1': 0.0597, 'HE2': 0.0597, 'HE3': 0.0597,
            'C': 0.6123, 'O': -0.5713
        },
        'PHE': {
            'N': 0.1737, 'H1': 0.1921, 'H2': 0.1921, 'H3': 0.1921, 'CA': 0.0733, 'HA': 0.1041,
            'CB': 0.033, 'HB2': 0.0104, 'HB3': 0.0104, 'CG': 0.0031, 'CD1': -0.1392, 'HD1': 0.1374,
            'CE1': -0.1602, 'HE1': 0.1433, 'CZ': -0.1208, 'HZ': 0.1329,
            'CE2': -0.1603, 'HE2': 0.1433, 'CD2': -0.1391, 'HD2': 0.1374,
            'C': 0.6123, 'O': -0.5713
        },
        'PRO': {
            'N': -0.202, 'H2': 0.312, 'H3': 0.312, 'CD': -0.012, 'HD2': 0.1, 'HD3': 0.1,
            'CG': -0.121, 'HG2': 0.1, 'HG3': 0.1, 'CB': -0.115, 'HB2': 0.1, 'HB3': 0.1,
            'CA': 0.1, 'HA': 0.1, 'C': 0.526, 'O': -0.5
        },
        'SER': {
            'N': 0.1849, 'H1': 0.1898, 'H2': 0.1898, 'H3': 0.1898, 'CA': 0.0567, 'HA': 0.0782,
            'CB': 0.2596, 'HB2': 0.0273, 'HB3': 0.0273, 'OG': -0.6714, 'HG': 0.4239,
            'C': 0.6163, 'O': -0.5722
        },
        'THR': {
            'N': 0.1812, 'H1': 0.1934, 'H2': 0.1934, 'H3': 0.1934, 'CA': 0.0034, 'HA': 0.1087,
            'CB': 0.4514, 'HB': -0.0323, 'CG2': -0.2554, 'HG21': 0.0627, 'HG22': 0.0627, 'HG23': 0.0627,
            'OG1': -0.6764, 'HG1': 0.407, 'C': 0.6163, 'O': -0.5722
        },
        'TRP': {
            'N': 0.1913, 'H1': 0.1888, 'H2': 0.1888, 'H3': 0.1888, 'CA': 0.0421, 'HA': 0.1162,
            'CB': 0.0543, 'HB2': 0.0222, 'HB3': 0.0222, 'CG': -0.1654, 'CD1': -0.1788, 'HD1': 0.2195,
            'NE1': -0.3444, 'HE1': 0.3412, 'CE2': 0.1575, 'CZ2': -0.271, 'HZ2': 0.1589, 'CH2': -0.108, 'HH2': 0.1411,
            'CZ3': -0.2034, 'HZ3': 0.1458, 'CE3': -0.2265, 'HE3': 0.1646, 'CD2': 0.1132,
            'C': 0.6123, 'O': -0.5713
        },
        'TYR': {
            'N': 0.194, 'H1': 0.1873, 'H2': 0.1873, 'H3': 0.1873, 'CA': 0.057, 'HA': 0.0983,
            'CB': 0.0659, 'HB2': 0.0102, 'HB3': 0.0102, 'CG': -0.0205, 'CD1': -0.2002, 'HD1': 0.172,
            'CE1': -0.2239, 'HE1': 0.165, 'CZ': 0.3139, 'OH': -0.5578, 'HH': 0.4001, 'CE2': -0.2239, 'HE2': 0.165,
            'CD2': -0.2002, 'HD2': 0.172, 'C': 0.6123, 'O': -0.5713
        },
        'VAL': {
            'N': 0.0577, 'H1': 0.2272, 'H2': 0.2272, 'H3': 0.2272, 'CA': -0.0054, 'HA': 0.1093,
            'CB': 0.3196, 'HB': -0.0221, 'CG1': -0.3129, 'HG11': 0.0735, 'HG12': 0.0735, 'HG13': 0.0735,
            'CG2': -0.3129, 'HG21': 0.0735, 'HG22': 0.0735, 'HG23': 0.0735,
            'C': 0.6163, 'O': -0.5722
        }
    },
    "aminoct12_charge": {
        'ALA': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.1747, 'HA': 0.1067,
            'CB': -0.2093, 'HB1': 0.0764, 'HB2': 0.0764, 'HB3': 0.0764,
            'C': 0.7731, 'O': -0.8055, 'OXT': -0.8055
        },
        'ARG': {
            'N': -0.3481, 'H': 0.2764, 'CA': -0.3068, 'HA': 0.1447,
            'CB': -0.0374, 'HB2': 0.0371, 'HB3': 0.0371,
            'CG': 0.0744, 'HG2': 0.0185, 'HG3': 0.0185,
            'CD': 0.1114, 'HD2': 0.0468, 'HD3': 0.0468,
            'NE': -0.5564, 'HE': 0.3479, 'CZ': 0.8368,
            'NH1': -0.8737, 'HH11': 0.4493, 'HH12': 0.4493,
            'NH2': -0.8737, 'HH21': 0.4493, 'HH22': 0.4493,
            'C': 0.8557, 'O': -0.8266, 'OXT': -0.8266
        },
        'ASN': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.208, 'HA': 0.1358,
            'CB': -0.2299, 'HB2': 0.1023, 'HB3': 0.1023,
            'CG': 0.7153, 'OD1': -0.601,
            'ND2': -0.9084, 'HD21': 0.415, 'HD22': 0.415,
            'C': 0.805, 'O': -0.8147, 'OXT': -0.8147
        },
        'ASP': {
            'N': -0.5192, 'H': 0.3055, 'CA': -0.1817, 'HA': 0.1046,
            'CB': -0.0677, 'HB2': -0.0212, 'HB3': -0.0212,
            'CG': 0.8851, 'OD1': -0.8162, 'OD2': -0.8162,
            'C': 0.7256, 'O': -0.7887, 'OXT': -0.7887
        },
        'CYS': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.1635, 'HA': 0.1396,
            'CB': -0.1996, 'HB2': 0.1437, 'HB3': 0.1437,
            'SG': -0.3102, 'HG': 0.2068,
            'C': 0.7497, 'O': -0.7981, 'OXT': -0.7981
        },
        'CYX': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.1318, 'HA': 0.0938,
            'CB': -0.1943, 'HB2': 0.1228, 'HB3': 0.1228, 'SG': -0.0529,
            'C': 0.7618, 'O': -0.8041, 'OXT': -0.8041
        },
        'GLN': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.2248, 'HA': 0.1232,
            'CB': -0.0664, 'HB2': 0.0452, 'HB3': 0.0452,
            'CG': -0.021, 'HG2': 0.0203, 'HG3': 0.0203, 'CD': 0.7093,
            'OE1': -0.6098, 'NE2': -0.9574, 'HE21': 0.4304, 'HE22': 0.4304,
            'C': 0.7775, 'O': -0.8042, 'OXT': -0.8042
        },
        'GLU': {
            'N': -0.5192, 'H': 0.3055, 'CA': -0.2059, 'HA': 0.1399,
            'CB': 0.0071, 'HB2': -0.0078, 'HB3': -0.0078,
            'CG': 0.0675, 'HG2': -0.0548, 'HG3': -0.0548,
            'CD': 0.8183, 'OE1': -0.822, 'OE2': -0.822,
            'C': 0.742, 'O': -0.793, 'OXT': -0.793
        },
        'GLY': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.2493, 'HA2': 0.1056, 'HA3': 0.1056,
            'C': 0.7231, 'O': -0.7855, 'OXT': -0.7855
        },
        'HID': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.1739, 'HA': 0.11,
            'CB': -0.1046, 'HB2': 0.0565, 'HB3': 0.0565,
            'CG': 0.0293, 'ND1': -0.3892, 'HD1': 0.3755,
            'CE1': 0.1925, 'HE1': 0.1418, 'NE2': -0.5629,
            'CD2': 0.1001, 'HD2': 0.1241,
            'C': 0.7615, 'O': -0.8016, 'OXT': -0.8016
        },
        'HIE': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.2699, 'HA': 0.165,
            'CB': -0.1068, 'HB2': 0.062, 'HB3': 0.062,
            'CG': 0.2724, 'ND1': -0.5517, 'CE1': 0.1558, 'HE1': 0.1448,
            'NE2': -0.267, 'HE2': 0.3319, 'CD2': -0.2588, 'HD2': 0.1957,
            'C': 0.7916, 'O': -0.8065, 'OXT': -0.8065
        },
        'HIP': {
            'N': -0.3481, 'H': 0.2764, 'CA': -0.1445, 'HA': 0.1115,
            'CB': -0.08, 'HB2': 0.0868, 'HB3': 0.0868,
            'CG': 0.0298, 'ND1': -0.1501, 'HD1': 0.3883,
            'CE1': -0.0251, 'HE1': 0.2694,
            'NE2': -0.1683, 'HE2': 0.3913,
            'CD2': -0.1256, 'HD2': 0.2336,
            'C': 0.8032, 'O': -0.8177, 'OXT': -0.8177
        },
        'HYP': {
            'N': -0.2802, 'CD': 0.0778, 'HD22': 0.0331, 'HD23': 0.0331,
            'CG': 0.081, 'HG': 0.0416, 'OD1': -0.6134, 'HD1': 0.3851,
            'CB': 0.0547, 'HB2': 0.0426, 'HB3': 0.0426, 'CA': -0.0993, 'HA': 0.0776,
            'C': 0.6631, 'O': -0.7697, 'OXT': -0.7697
        },
        'ILE': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.31, 'HA': 0.1375,
            'CB': 0.0363, 'HB': 0.0766,
            'CG2': -0.3498, 'HG21': 0.1021, 'HG22': 0.1021, 'HG23': 0.1021,
            'CG1': -0.0323, 'HG12': 0.0321, 'HG13': 0.0321,
            'CD1': -0.0699, 'HD11': 0.0196, 'HD12': 0.0196, 'HD13': 0.0196,
            'C': 0.8343, 'O': -0.819, 'OXT': -0.819
        },
        'LEU': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.2847, 'HA': 0.1346,
            'CB': -0.2469, 'HB2': 0.0974, 'HB3': 0.0974, 'CG': 0.3706, 'HG': -0.0374,
            'CD1': -0.4163, 'HD11': 0.1038, 'HD12': 0.1038, 'HD13': 0.1038,
            'CD2': -0.4163, 'HD21': 0.1038, 'HD22': 0.1038, 'HD23': 0.1038,
            'C': 0.8326, 'O': -0.8199, 'OXT': -0.8199
        },
        'LYS': {
            'N': -0.3481, 'H': 0.2764, 'CA': -0.2903, 'HA': 0.1438,
            'CB': -0.0538, 'HB2': 0.0482, 'HB3': 0.0482,
            'CG': 0.0227, 'HG2': 0.0134, 'HG3': 0.0134,
            'CD': -0.0392, 'HD2': 0.0611, 'HD3': 0.0611,
            'CE': -0.0176, 'HE2': 0.1121, 'HE3': 0.1121,
            'NZ': -0.3741, 'HZ1': 0.3374, 'HZ2': 0.3374, 'HZ3': 0.3374,
            'C': 0.8488, 'O': -0.8252, 'OXT': -0.8252
        },
        'MET': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.2597, 'HA': 0.1277,
            'CB': -0.0236, 'HB2': 0.048, 'HB3': 0.048,
            'CG': 0.0492, 'HG2': 0.0317, 'HG3': 0.0317,
            'SD': -0.2692, 'CE': -0.0376, 'HE1': 0.0625, 'HE2': 0.0625, 'HE3': 0.0625,
            'C': 0.8013, 'O': -0.8105, 'OXT': -0.8105
        },
        'PHE': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.1825, 'HA': 0.1098,
            'CB': -0.0959, 'HB2': 0.0443, 'HB3': 0.0443,
            'CG': 0.0552, 'CD1': -0.13, 'HD1': 0.1408,
            'CE1': -0.1847, 'HE1': 0.1461, 'CZ': -0.0944, 'HZ': 0.128,
            'CE2': -0.1847, 'HE2': 0.1461, 'CD2': -0.13, 'HD2': 0.1408,
            'C': 0.766, 'O': -0.8026, 'OXT': -0.8026
        },
        'PRO': {
            'N': -0.2802, 'CD': 0.0434, 'HD2': 0.0331, 'HD3': 0.0331,
            'CG': 0.0466, 'HG2': 0.0172, 'HG3': 0.0172,
            'CB': -0.0543, 'HB2': 0.0381, 'HB3': 0.0381, 'CA': -0.1336, 'HA': 0.0776,
            'C': 0.6631, 'O': -0.7697, 'OXT': -0.7697
        }, 'SER': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.2722, 'HA': 0.1304,
            'CB': 0.1123, 'HB2': 0.0813, 'HB3': 0.0813,
            'OG': -0.6514, 'HG': 0.4474,
            'C': 0.8113, 'O': -0.8132, 'OXT': -0.8132
        },
        'THR': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.242, 'HA': 0.1207,
            'CB': 0.3025, 'HB': 0.0078,
            'CG2': -0.1853, 'HG21': 0.0586, 'HG22': 0.0586, 'HG23': 0.0586,
            'OG1': -0.6496, 'HG1': 0.4119,
            'C': 0.781, 'O': -0.8044, 'OXT': -0.8044
        },
        'TRP': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.2084, 'HA': 0.1272,
            'CB': -0.0742, 'HB2': 0.0497, 'HB3': 0.0497,
            'CG': -0.0796, 'CD1': -0.1808, 'HD1': 0.2043,
            'NE1': -0.3316, 'HE1': 0.3413, 'CE2': 0.1222, 'CZ2': -0.2594, 'HZ2': 0.1567,
            'CH2': -0.102, 'HH2': 0.1401, 'CZ3': -0.2287, 'HZ3': 0.1507,
            'CE3': -0.1837, 'HE3': 0.1491, 'CD2': 0.1078,
            'C': 0.7658, 'O': -0.8011, 'OXT': -0.8011
        },
        'TYR': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.2015, 'HA': 0.1092,
            'CB': -0.0752, 'HB2': 0.049, 'HB3': 0.049, 'CG': 0.0243,
            'CD1': -0.1922, 'HD1': 0.178, 'CE1': -0.2458, 'HE1': 0.1673,
            'CZ': 0.3395, 'OH': -0.5643, 'HH': 0.4017,
            'CE2': -0.2458, 'HE2': 0.1673, 'CD2': -0.1922, 'HD2': 0.178,
            'C': 0.7817, 'O': -0.807, 'OXT': -0.807
        },
        'VAL': {
            'N': -0.3821, 'H': 0.2681, 'CA': -0.3438, 'HA': 0.1438,
            'CB': 0.194, 'HB': 0.0308,
            'CG1': -0.3064, 'HG11': 0.0836, 'HG12': 0.0836, 'HG13': 0.0836,
            'CG2': -0.3064, 'HG21': 0.0836, 'HG22': 0.0836, 'HG23': 0.0836,
            'C': 0.835, 'O': -0.8173, 'OXT': -0.8173
        },
        'NHE': {
            'N': -0.463, 'HN1': 0.2315, 'HN2': 0.2315
        },
        'NME': {
            'N': -0.4157, 'H': 0.2719,
            'CH3': -0.149, 'HH31': 0.0976, 'HH32': 0.0976, 'HH33': 0.0976
        }
    },
}


def cginParser(content: str, control: str, variables: dict = {}):
    if control == "RESIDUE":
        cginRes = []
        for sres in content.split(','):
            sres = sres.strip()
            if sres:
                cginRes.append(sres)
        return cginRes

    if control == "VARIABLES":
        cginVariables = {}
        for var in content.split(','):
            var = var.strip()
            if var:
                varName, varValue = var.split(':')
                cginVariables[varName.strip()] = varValue.strip().split()
        return cginVariables

    if control == "RES-BEAD":
        cginResSBead = {}
        for resSBead in content.split(','):
            sreSBead = resSBead.strip()
            if sreSBead:
                sres, sbeads = sreSBead.split(':')
                sres, sbeads = sres.strip(), sbeads.strip()
                cginResSBead[sres] = []
                for sbead in sbeads.split('|'):
                    sbead = sbead.strip()
                    if sbead:
                        cginResSBead[sres].append(sbead)
        return cginResSBead

    if control == "RES-ATOM":
        cginResSAtom = {}
        for resSBeadSAtom in content.split(','):
            resSBeadSAtom = resSBeadSAtom.strip()
            if resSBeadSAtom:
                sres, beadSAtoms = resSBeadSAtom.split(':')
                sres, beadSAtoms = sres.strip(), beadSAtoms.strip()
                cginResSAtom[sres] = []
                for beadSAtom in beadSAtoms.split('|'):
                    beadSAtom = beadSAtom.strip()
                    for satom in beadSAtom.split():
                        if re.match(r"\w+", satom):
                            cginResSAtom[sres].append(satom)
                        else:  # satom should be a variable
                            satom = satom.strip("[]")
                            if satom in variables.keys():
                                varAtomList = variables[satom]
                                for varSAtom in varAtomList:
                                    cginResSAtom[sres].append(varSAtom)
                            else:
                                # skip the res when checking
                                cginResSAtom[sres] = ['*']
        return cginResSAtom

    if control == "RES-BEAD-ATOM":
        cginResSBeadSAtom = {}
        for resSBeadSAtom in content.split(','):
            resSBeadSAtom = resSBeadSAtom.strip()
            if resSBeadSAtom:
                sres, beadSAtoms = resSBeadSAtom.split(':')
                sres, beadSAtoms = sres.strip(), beadSAtoms.strip()
                cginResSBeadSAtom[sres] = []
                for beadSAtom in beadSAtoms.split('|'):
                    beadSAtom = beadSAtom.strip()
                    satoms = []
                    for satom in beadSAtom.split():
                        if re.match(r"\w+", satom):
                            satoms.append(satom)
                        else:
                            satom = satom.strip("[]")
                            if satom in variables.keys():
                                # satom is a variable
                                varAtomList = variables[satom]
                                for varSAtom in varAtomList:
                                    satoms.append(varSAtom)
                            else:  # satom is a special character */-
                                satoms.append(satom)
                    cginResSBeadSAtom[sres].append(satoms)
        return cginResSBeadSAtom

    if control == "RES-BEAD-ATTRI":
        cginResSBeadProp = {}
        for resSBeadProp in content.split(','):
            resSBeadProp = resSBeadProp.strip()
            if resSBeadProp:
                sres, beadProps = resSBeadProp.split(':')
                sres, beadProps = sres.strip(), beadProps.strip()
                cginResSBeadProp[sres] = {}
                for beadProp in beadProps.split('|'):
                    beadProp = beadProp.strip()
                    if beadProp:
                        sbead, propValue = beadProp.split()
                        propValue = float(propValue)
                        cginResSBeadProp[sres][sbead] = propValue
        return cginResSBeadProp


class Residue:
    def __init__(self, residueLines: list, loc: list = [1, 'A', (1, "")]):
        self.loc = copy.deepcopy(loc)
        self.name3C = self.loc[2][1]
        self.natom = 0
        self.atoms = {}
        pattern = PATTERNS["atom/bead"]
        pre_iatom, iatom = 0, 0
        for i, line in enumerate(residueLines):
            mobj = pattern.match(line)
            if not(mobj):
                print(line)
                print("error: unrcognized format at (model %d, chain %s, %d-th residue %s, %d-th atom/bead)" %
                      (loc[0], loc[1], loc[2][0], loc[2][1], i+1), file=sys.stderr)
                quit()
            beadType = mobj.group(1)
            pre_iatom = iatom
            iatom = int(re.sub(r"\s", "", mobj.group(2)))
            if pre_iatom > 0 and iatom-pre_iatom != 1:
                # here, I should have consider alternative conformations with occupancy < 1, but finally give up,
                # because of the complexity and limited use in my work
                print("warning: confusing atom index at (model %d, chain %s, %d-th residue %s, %d-th atom/bead)" %
                      (loc[0], loc[1],  loc[2][0], loc[2][1], i+1), file=sys.stderr)
            satom = re.sub(r"\s", "", mobj.group(3))
            atomCrdx = float(re.sub(r"\s", "", mobj.group(4)))
            atomCrdy = float(re.sub(r"\s", "", mobj.group(5)))
            atomCrdz = float(re.sub(r"\s", "", mobj.group(6)))
            atomOcp = float(re.sub(r"\s", "", mobj.group(7)))
            atomTempfac = float(re.sub(r"\s", "", mobj.group(8)))
            atomMass = re.sub(r"\s", "", mobj.group(9))
            atomType = re.sub(r"\s", "", mobj.group(10))
            if atomMass:
                atomMass = float(atomMass)
            else:
                if beadType == "ATOM" and atomType in DATA["atom_mass"].keys():
                    atomMass = DATA["atom_mass"][atomType]
                else:
                    atomMass = 0.
            atomCharge = re.sub(r"\s", "", mobj.group(11))
            if atomCharge:
                charge = int(atomCharge[0])
                if atomCharge[1] == '-':
                    charge = -charge
            else:
                charge = 200
            atomCharge = charge

            atom = {"name": satom, "type": atomType, "xyz": [atomCrdx, atomCrdy, atomCrdz],
                    "x": atomCrdx, "y": atomCrdy, "z": atomCrdz, "ocp": atomOcp, "bfac": atomTempfac,
                    }
            if atomMass > 0:
                atom["mass"] = atomMass
            if atomCharge != 200:
                atom["charge"] = atomCharge

            if satom in self.atoms.keys():
                print("error: duplicated atoms at (model %d, chain %s, %d-th residue %s, atom/bead %s)" %
                      (loc[0], loc[1], loc[2][0], loc[2][1], satom), file=sys.stderr)
                quit()
            else:
                self.atoms[satom] = atom
                self.natom += 1

    def appendAtom(self, atom: dict):
        keyCompulsory = {"name", "type", "xyz", "x", "y", "z", "ocp", "bfac"}
        keyOptional = keyCompulsory | {"mass", "charge"}
        keyDoHave = set(atom.keys())
        if self.name3C and keyCompulsory <= keyDoHave and keyDoHave <= keyOptional:
            satom = atom["name"]
            self.natom += 1
            self.atoms[satom] = atom
        return self


class Chain:
    def __init__(self, chainLines: list, loc: list = [1, 'A']):
        self.loc = copy.deepcopy(loc)
        self.seq3C = []
        self.residues = []
        pre_resName, resName = "", ""
        pre_ires, ires = 0, 0
        resStart, resEnd = -1, -1
        loc.append((1, ""))
        for i, line in enumerate(chainLines):
            pre_resName = resName
            resName = line[17:20]
            pre_ires = ires
            ires = int(line[22:26])

            if pre_ires != ires or line[:3] == "TER":
                if pre_resName:
                    self.seq3C.append(pre_resName)
                    resEnd = i
                    loc[2] = (pre_ires, pre_resName)
                    residue = Residue(chainLines[resStart:resEnd], loc)
                    self.residues.append(residue)
                    # check residue index
                    if ires > pre_ires or line[:3] == "TER":
                        for j in range(pre_ires+1, ires):
                            loc[2] = (j, "")
                            residue = Residue([], loc)
                            self.residues.append(residue)
                            print("warning: missing residue at (model %d, chain %s, residue %d)" % (
                                loc[0], loc[1], loc[2][0]), file=sys.stderr)
                    else:
                        print("error: confusing residue index at (model %d, chain %s, residue %d)" % (
                            loc[0], loc[1], loc[2][0]), file=sys.stderr)
                        quit()
                resStart = i

        return None

    def appendResidue(self, residue: Residue):
        sres = residue.name3C
        self.seq3C.append(sres)
        self.residues.append(residue)
        return self


class Model:
    def __init__(self, modelLines: list, loc: list = [1]):
        self.loc = copy.deepcopy(loc)
        self.chains = {}
        chainIndex = {}
        loc.append('A')
        for i, line in enumerate(modelLines):
            if len(line) >= 22:
                schain = line[21]
                if(schain in chainIndex.keys()):
                    chainIndex[schain].append(line)
                else:
                    chainIndex[schain] = [line]

        for key in chainIndex.keys():
            loc[1] = key
            chain = Chain(chainIndex[key], loc)
            self.chains[key] = chain

        return None

    def appendChain(self, chain: Chain):
        idchain = chain.loc[1]
        self.chains[idchain] = chain
        return self


class Pro:
    def readPdb(self, file, name=""):
        if name:
            self.name = name
        else:
            self.name = file.split('/')[-1].split('.')[0]
        self.nmodel = 0
        self.models = []
        with open(file, 'r') as f:
            lines = f.readlines()

        modelStart, modelEnd = -1, -1
        atomStart, atomEnd = -1, -1
        for i, line in enumerate(lines):
            if len(line) >= 5 and line[:5] == "MODEL":
                modelStart = i
            if len(line) >= 6 and line[:6] == "ENDMDL":
                modelEnd = i
                self.nmodel += 1
                loc = [self.nmodel]
                model = Model(lines[modelStart + 1:modelEnd], loc)
                self.models.append(model)
            if len(line) >= 4 and line[:4] in ["ATOM", "BEAD"] and atomStart < 0:
                atomStart = i
            if atomStart >= 0 and len(line) >= 3 and line[:3] == "TER":
                atomEnd = i

        if modelStart == -1:
            if atomStart == -1:
                print("pdb parsing error: cannot find at least 1 model",
                      file=sys.stderr)
                quit()
            else:
                if atomEnd == -1:
                    print("pdb parsing error: cannot find a TER for the model",
                          file=sys.stderr)
                    quit()
                else:
                    if atomEnd <= atomStart:
                        print("pdb parsing error: unexpected ATOM after TER",
                              file=sys.stderr)
                        quit()
                    else:
                        self.nmodel = 1
                        loc = [1]
                        self.models.append(
                            Model(lines[atomStart:atomEnd+1], loc))
            return self

    def __init__(self, file: str = "", name="") -> None:
        if file:
            self.readPdb(file, name)
        else:
            if name:
                self.name = name
            else:
                self.name = "pro"
            self.nmodel = 0
            self.models = []
        return None

    def appendModel(self, model: Model):
        self.nmodel += 1
        self.models.append(model)
        return self

    # check pdb on atomic/bead level, against .cgin file input format or output format
    def pdbCheck(self, cgin: str = "pdb2012_amino12.cgin", outform: bool = True) -> bool:
        ret: bool = True
        print("checking pdb structure %s against standard %s" %
              (self.name, cgin))
        with open(cgin, 'r') as file:
            content = file.read()

        pattern = PATTERNS["cgin"]
        mobj = pattern.match(content)
        if not(mobj):
            print("error:invalid .cgin file %s" % cgin, file=sys.stderr)
            quit()

        tmpstr = mobj.group(2)
        cginRes = cginParser(tmpstr, "RESIDUE")
        tmpstr = mobj.group(11)
        cginVariables = cginParser(tmpstr, "VARIABLES")
        if outform:
            tmpstr = mobj.group(3)
            cginResSAtom = cginParser(tmpstr, "RES-BEAD")
        else:
            tmpstr = mobj.group(5)
            cginResSAtom = cginParser(tmpstr, "RES-ATOM", cginVariables)
            for sres in cginResSAtom.keys():
                if cginResSAtom[sres][0] == '*':
                    print("exception: invalid word for checking in .cgin file %s, atom checking of residue %s will be skipped" % (
                        cgin, sres))

        for model in self.models:
            print("checking model %d" % model.loc[0])
            for idchain in model.chains.keys():
                chain = model.chains[idchain]
                print("checking model %d - chain %s" %
                      (chain.loc[0], chain.loc[1]))

                for i, residue in enumerate(chain.residues):
                    sres = residue.name3C
                    if sres:
                        if sres in cginRes:
                            if cginResSAtom[sres][0] != '*':
                                # checking atom names
                                isNTer: bool = (i == 0)
                                isCTer: bool = (i == len(chain.seq3C)-1)
                                isPro: bool = (residue.name3C == "PRO")
                                resDoContain: set = set(residue.atoms.keys())
                                resShouldContain: set = set(cginResSAtom[sres])
                                if isNTer:
                                    if isPro:
                                        resShouldContain |= {"H1", "H2"}
                                    else:
                                        resShouldContain -= {"H"}
                                        resShouldContain |= {"H1", "H2", "H3"}
                                if isCTer:
                                    resShouldContain |= {"OXT"}
                                setUnrecog = resDoContain-resShouldContain
                                setMissing = resShouldContain-resDoContain
                                if setUnrecog:
                                    print("exception: unrecognized atom(s) in (model %d, chain %s, %d-th residue %s): " %
                                          (residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1]), setUnrecog)
                                    ret = False
                                if setMissing:
                                    print("exception: missing atom(s) in (model %d, chain %s, %d-th residue %s): " %
                                          (residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1]), setMissing)
                                    ret = False
                                # checking atom ocp, bfac, mass and charge
                                for key in residue.atoms.keys():
                                    atom = residue.atoms[key]
                                    if atom["bfac"] > 50.0:
                                        print("exception: noticable B-factor %f at (model %d, chain %s, %d-th residue %s, atom %s)" %
                                              (atom["bfac"], residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1], atom["name"]))
                                    if atom["ocp"] < 1.0:
                                        # pdb allows alternative conformations, but this module doesn't yet support it,
                                        # so alternative is regarded as exceptions
                                        print("exception: occupancy %f <1 at (model %d, chain %s, %d-th residue %s, atom %s)" %
                                              (atom["ocp"], residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1], atom["name"]))
                                    if "mass" not in atom.keys():
                                        print("exception: no atom mass info for (model %d, chain %s, %d-th residue %s, atom %s)" %
                                              (residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1], atom["name"]))
                                    if "charge" not in atom.keys():
                                        print("exception: no atom charge info for (model %d, chain %s, %d-th residue %s, atom %s)" %
                                              (residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1], atom["name"]))
                        else:
                            print("exception: unrecognized residue at (model %s, chain %s, %d-th residue)" %
                                  (residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1]))
                            ret = False
                    else:
                        print("exception: missing residue at (model %d, chain %s, %d-th residue %s)" %
                              (residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1]))
                        ret = False
        print("All chains in model %d checked, no news is good news" %
              (model.loc[0]))
        return ret

    def assignMass(self, atomMass: dict = DATA["atom_mass"], mode: int = 0):
        for model in self.models:
            for idchain, chain in model.chains.items():
                for residue in chain.residues:
                    ires = residue.loc[2][0]
                    sres = residue.name3C
                    for satom, atom in residue.atoms.items():
                        if mode == 0:  # supplement
                            if "mass" not in atom.keys() and satom in atomMass.keys():
                                atom["mass"] = atomMass[satom]
                            else:
                                print("caution: need to assign mass at (model %d, chain %s, %d-th residue %s, atom %s)" % (
                                    residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1]) +
                                    " of structure %s, but failed to find a mathing in param atomMass" % (self.name))
                        elif mode == 1:  # calibrate
                            if satom in atomMass.keys():
                                atom["mass"] = atomMass[satom]
                            else:
                                print("caution: need to assign mass at (model %d, chain %s, %d-th residue %s, atom %s)" % (
                                    residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1]) +
                                    " of structure %s, but failed to find a mathing in param atom_mass" % (self.name))
                        else:
                            print("warning: mode %d not supported yet, nothing is done. try 0 or 1" %
                                  mode, file=sys.stderr)
        return self

    def assignCharge(self, resAtomCharge: dict = DATA["amino12_charge"],
                     resAtomCharge_NT: dict = DATA["aminont12_charge"],
                     resAtomCharge_CT: dict = DATA["aminoct12_charge"], mode: int = 0):
        for model in self.models:
            for chain in model.chains.values():
                for i, residue in chain.residues:
                    sres = residue.name3C
                    if i == 0:
                        atomCharge = resAtomCharge_NT[sres]
                    elif i == len(chain.seq3C)-1:
                        atomCharge = resAtomCharge_CT[sres]
                    else:
                        atomCharge = resAtomCharge[sres]
                    for satom, atom in residue.atoms.items():
                        if mode == 0:  # supplement
                            if "charge" not in atom.keys() and satom in atomCharge.keys():
                                atom["charge"] = atomCharge[satom]
                            else:
                                print("caution: need to assign charge at (model %d, chain %s, %d-th residue %s, atom %s)" % (
                                    residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1]) +
                                    " of structure %s, but failed to find a match in param atom_charge" % (self.name))
                        elif mode == 1:  # calibrate
                            if satom in atomCharge.keys():
                                atom["charge"] = atomCharge[satom]
                            print("caution: need to assign charge at (model %d, chain %s, %d-th residue %s, atom %s)" % (
                                residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1]) +
                                " of structure %s, but failed to find a match in param atom_charge" % (self.name))
                        else:
                            print("warning: mode %d not supported yet, nothing is done. try 0 or 1" %
                                  mode, file=sys.stderr)
        return self

    # do coarse-grain mapping accoding to .cgin
    def pdbMap(self, cgin: str):
        newPro = Pro()
        with open(cgin, 'r') as file:
            content = file.read()

        pattern = PATTERNS["cgin"]
        mobj = pattern.match(content)
        if not(mobj):
            print("error:invalid .cgin file %s" % cgin, file=sys.stderr)
            quit()

        cginName = mobj.group(1).strip()
        tmpstr = mobj.group(2)
        cginRes = cginParser(tmpstr, "RESIDUE")
        tmpstr = mobj.group(11)
        cginVariables = cginParser(tmpstr, "VARIABLES")
        tmpstr = mobj.group(3)
        cginResSBead = cginParser(tmpstr, "RES-BEAD")
        tmpstr = mobj.group(4)
        cginMassResSBeadSAtom = cginParser(
            tmpstr, "RES-BEAD-ATOM", cginVariables)
        tmpstr = mobj.group(5)
        cginCrdResSBeadSAtom = cginParser(
            tmpstr, "RES-BEAD-ATOM", cginVariables)
        cginCGMode: int = int(re.sub(r"\s", "", mobj.group(6).strip()))
        tmpstr = mobj.group(7)
        cginResBeadCharge = cginParser(tmpstr, "RES-BEAD-ATTRI")
        tmpstr = mobj.group(8)
        cginResBeadCharge_NT = cginParser(tmpstr, "RES-BEAD-ATTRI")
        tmpstr = mobj.group(9)
        cginResBeadCharge_CT = cginParser(tmpstr, "RES-BEAD-ATTRI")
        cginChargeMode: int = int(re.sub(r"\s", "", mobj.group(10).strip()))

        print("doing Coarse-Grain mapping of pdb structure %s according to rule %s" %
              (self.name, cgin))
        for model in self.models:
            newModel = Model([], model.loc)
            for idchain in model.chains.keys():
                chain = model.chains[idchain]
                newChain = Chain([], chain.loc)
                for iires, residue in enumerate(chain.residues):
                    newResidue = Residue([], residue.loc)
                    sres = residue.name3C
                    ires = residue.loc[2][0]
                    # test whether residue is empty
                    if not(sres):
                        print("error: missing residue at (model %d, chain %s, %d-th residue %s)" %
                              (residue.loc[0], residue.loc[1], ires, sres), file=sys.stderr)
                        quit()
                    if sres not in cginRes:
                        print("error: unrecognized residue at (model %s, chain %s, %d-th residue)" %
                              (residue.loc[0], residue.loc[1], ires, sres), file=sys.stderr)
                        quit()
                    # compute implicit atom set for '-'
                    resSBeads = cginResSBead[sres]
                    massResSBeadSAtoms = cginMassResSBeadSAtom[sres]
                    crdResSBeadSAtoms = cginCrdResSBeadSAtom[sres]
                    if(len(massResSBeadSAtoms) < len(resSBeads)):
                        print("error: incomplete mass mapping rule for  residue-type %s, check your .cgin file %s" %
                              (sres, cgin), file=sys.stderr)
                        quit()
                    if(len(massResSBeadSAtoms) < len(resSBeads)):
                        print("error: incomplete coordinate mapping rule for  residue-type %s, check your .cgin file %s" %
                              (sres, cgin), file=sys.stderr)
                        quit()
                    massResRestAtoms, crdResRestAtoms = set(), set()
                    for beadAtomList_mass, beadAtomList_crd in \
                            zip(massResSBeadSAtoms, crdResSBeadSAtoms):
                        massResRestAtoms |= set(beadAtomList_mass)
                        crdResRestAtoms |= set(beadAtomList_crd)
                    massResRestAtoms = set(
                        residue.atoms.keys())-massResRestAtoms
                    crdResRestAtoms = set(residue.atoms.keys())-crdResRestAtoms
                    # default atom and bead charge dict
                    if iires == 0:
                        if cginChargeMode in [1, 2]:
                            resAtomCharge = DATA["aminont12_charge"][sres]
                        elif cginChargeMode == 3:
                            resBeadCharge = cginResBeadCharge_NT[sres]
                    elif iires == len(chain.seq3C)-1:
                        if cginChargeMode in [1, 2]:
                            resAtomCharge = DATA["aminoct12_charge"][sres]
                        elif cginChargeMode == 3:
                            resBeadCharge = cginResBeadCharge_CT[sres]
                    else:
                        if cginChargeMode in [1, 2]:
                            resAtomCharge = DATA["amino12_charge"][sres]
                        elif cginChargeMode == 3:
                            resBeadCharge = cginResBeadCharge[sres]
                    # mapping
                    for sbead, beadAtomList_mass, beadAtomList_crd in \
                            zip(resSBeads, massResSBeadSAtoms, crdResSBeadSAtoms):
                        bead = {"name": sbead, "type": "", "xyz": [0.0, 0.0, 0.0],
                                "x": 0.0, "y": 0.0, "z": 0.0, "ocp": 1.0, "bfac": 0.0,
                                "mass": 0.0, "charge": 0.0,
                                }
                        # final process the bead-atom set for mass and crd
                        beadAtomListJustify_crd = set()
                        for beadAtom_crd in set(beadAtomList_crd):
                            if beadAtom_crd == '*':
                                beadAtomListJustify_crd |= set(
                                    residue.atoms.keys())
                            elif beadAtom_crd == '-':
                                beadAtomListJustify_crd |= crdResRestAtoms
                            elif beadAtom_crd in residue.atoms.keys():
                                beadAtomListJustify_crd.add(beadAtom_crd)
                        beadAtomListJustify_mass = set()
                        for beadAtom_mass in set(beadAtomList_mass):
                            if beadAtom_mass == '*':
                                beadAtomListJustify_mass |= set(
                                    residue.atoms.keys())
                            elif beadAtom_mass == '-':
                                beadAtomListJustify_mass |= massResRestAtoms
                            elif beadAtom_mass in residue.atoms.keys():
                                beadAtomListJustify_mass.add(beadAtom_mass)
                        if not(beadAtomListJustify_crd):
                            if not(beadAtomListJustify_mass):
                                print("warning: empty bead %s for both mass and crd in residue-type %s, automatically removed it" % (
                                    sbead, sres), file=sys.stderr)
                                continue
                            else:
                                print("error: empty bead %s for crd but not mass in residue-type %s, check your .cgin file %s. " % (
                                    sbead, sres, cgin), file=sys.stderr)
                                quit()
                        elif not(beadAtomListJustify_mass):
                            print("error: empty bead %s for mass but not crd in residue-type %s, check your .cgin file %s. " % (
                                sbead, sres, cgin), file=sys.stderr)
                            quit()

                        # compute bead mass and coordinates
                        if cginCGMode in [0, 1, 2]:
                            mList, xList, yList, zList = [], [], [], []
                            for beadAtom_crd in beadAtomListJustify_crd:
                                atom = residue.atoms[beadAtom_crd]
                                if "mass" not in atom.keys():
                                    print("error: atom mass required but not found at (model %d, chain %s, %d-th residue %s, atom %s)" % (
                                        residue.loc[0], residue.loc[1], ires, sres, atom["name"]),
                                        file=sys.stderr)
                                    quit()
                                else:
                                    mList.append(atom["mass"])
                                    xList.append(atom["x"])
                                    yList.append(atom["y"])
                                    zList.append(atom["z"])
                            bead["mass"] = sum(mList)
                            bead["x"] = sum(
                                [m*x for m, x in zip(mList, xList)])/bead["mass"]
                            bead["y"] = sum(
                                [m*y for m, y in zip(mList, yList)])/bead["mass"]
                            bead["z"] = sum(
                                [m*z for m, z in zip(mList, zList)])/bead["mass"]
                            bead["xyz"] = [bead["x"], bead["y"], bead["z"]]

                            if cginCGMode == 2:
                                mList = []
                                for beadAtom_mass in beadAtomListJustify_mass:
                                    atom = residue.atoms[beadAtom_mass]
                                    if "mass" not in atom.keys():
                                        print("error: atom mass required but not found at (model %d, chain %s, %d-th residue %s, atom %s)" % (
                                            residue.loc[0], residue.loc[1], ires, sres, atom["name"]),
                                            file=sys.stderr)
                                        quit()
                                    else:
                                        mList.append(atom["mass"])
                                bead["mass"] = sum(mList)
                            elif cginCGMode == 0:
                                bead["type"] = atom["type"]
                                bead["ocp"] = atom["ocp"]
                                bead["bfac"] = atom["bfac"]

                        # compute bead charge
                        if cginChargeMode == 0:  # assign default 0.0
                            pass
                        elif cginChargeMode == 1:  # use atom-default charge and the mass set
                            qList = []
                            for beadAtom_mass in beadAtomListJustify_mass:
                                atom = residue.atoms[beadAtom_mass]
                                if "charge" in atom.keys():
                                    qList.append(atom["charge"])
                                else:
                                    if beadAtom_mass in resAtomCharge.keys():
                                        qList.append(
                                            resAtomCharge[beadAtom_mass])
                                        print("caution: atom charge required but not found at (model %d, chain %s" % (
                                            residue.loc[0], residue.loc[1])+" %d-th residue %s, atom %s) of structure %s. " % (
                                            ires, sres, beadAtom_mass, self.name)+"use default %f e" % resAtomCharge[beadAtom_mass],
                                            file=sys.stderr)
                                    else:
                                        print("error: atom charge required but not found at (model %d, chain %s" % (
                                            residue.loc[0], residue.loc[1])+" %d-th residue %s, atom %s) of structure" % (
                                            ires, sres, atom["name"], self.name), file=sys.stderr)
                                        quit()
                            bead["charge"] = sum(qList)
                        elif cginChargeMode == 2:  # use default-atom charge and the mass set
                            qList = []
                            for beadAtom_mass in beadAtomListJustify_mass:
                                atom = residue.atoms[beadAtom_mass]
                                if beadAtom_mass in resAtomCharge.keys():
                                    qList.append(resAtomCharge[beadAtom_mass])
                                else:
                                    if "charge" in atom.keys():
                                        print("caution: atom charge required but not found at (model %d, chain %s" % (
                                            residue.loc[0], residue.loc[1])+" %d-th residue %s, atom %s) of structure %s. " % (
                                            ires, sres, beadAtom_mass, self.name)+"use origin %f e" % atom["charge"],
                                            file=sys.stderr)
                                        qList.append(atom["charge"])
                                    else:
                                        print("error: atom charge required but not found at (model %d, chain %s" % (
                                            residue.loc[0], residue.loc[1])+" %d-th residue %s, atom %s) of structure" % (
                                            ires, sres, atom["name"], self.name), file=sys.stderr)
                                        quit()
                            bead["charge"] = sum(qList)
                        elif cginChargeMode == 3:  # use bead charge assigned in cgin file
                            if sbead in resBeadCharge.keys():
                                bead["charge"] = resBeadCharge[sbead]
                            else:
                                print("error: bead charge required at (model %d, chain %s, %d-th residue %s, bead %s)" % (
                                    residue.loc[0], residue.loc[1], ires, sres, sbead)+"of structure %s, but not found in cgin file %s"
                                    % (self.name, cgin), file=sys.stderr)
                                quit()

                        newResidue.appendAtom(bead)
                    newChain.appendResidue(newResidue)
                newModel.appendChain(newChain)
            newPro.appendModel(newModel)
        print("complete Coarse-Grain mapping of pdb structure %s according to rule %s" %
              (self.name, cgin))
        newPro.name = self.name+'_'+cginName
        return newPro

    def __atomBeadNameConvert(self, atom: dict) -> str:
        tatom = atom["type"]
        satom = atom["name"]
        ret = satom
        if len(ret) < 4:
            if len(tatom) <= 1:
                ret = " {:<3s}".format(ret)
            else:
                ret = "{:<4s}".format(ret)
        return ret

    def writePdb(self, outfile: str = "pro_out.pdb"):
        print("writing structure %s in a pdb-like format to file %s" %
              (self.name, outfile))
        with open(outfile, 'w') as file:
            file.write("TITLE     %s\n" % self.name)
            for imodel, model in enumerate(self.models):
                if self.nmodel > 1:
                    file.write("MODEL     %d\n" % imodel)
                iatom = 0
                for idchain, chain in model.chains.items():
                    for sres, residue in zip(chain.seq3C, chain.residues):
                        ires = residue.loc[2][0]
                        for atom in residue.atoms.values():
                            iatom += 1
                            title, smass = "BEAD", ""
                            if atom["type"]:
                                title = "ATOM"
                            if "mass" in atom.keys():
                                smass = "{:>6.2f}".format(atom["mass"])
                            satom = self.__atomBeadNameConvert(atom)
                            file.write("{:<6s}{:>5d} {:>4s} {:>3s} {:>1s}{:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}{:>6s}    {:>2s}  \n"
                                       .format(title, iatom, satom, sres, idchain, ires, atom["x"], atom["y"], atom["z"],
                                               atom["ocp"], atom["bfac"], smass, atom["type"])
                                       )
                    iatom += 1
                    file.write(
                        "TER    {:>4d}       {:>3s} {:>1s}{:>4d} \n".format(iatom, sres, idchain, ires+1))
                if self.nmodel > 1:
                    file.write("ENDMDL\n")
            file.write("END\n")
            print("structure %s written" % self.name)
            return None

    def dismat(self, imodel: int, mode: str = "RES-BEAD") -> np.ndarray:
        print("computing distance matrix of model %d of structure %s" %
              (imodel, self.name))
        if imodel < 1 or imodel > self.nmodel:
            print("error: invalid parameter, pdb do not have %d-th model" %
                  imodel, file=sys.stderr)
            quit()
        # a res-level computation will create a res-res matrix
        #  with channels of bead-pair distance
        if mode not in ["RES-BEAD", "BEAD-BEAD"]:
            print("error: invalid mode %s, only support \'RES-BEAD\' and \'BEAD-BEAD\'" %
                  mode, file=sys.stderr)
            quit()

        model = self.models[imodel-1]
        nres, nbead = 0, 0
        resNAtom_max = 0
        for chain in model.chains.values():
            nres += len(chain.seq3C)
            for residue in chain.residues:
                resNAtom = residue.natom
                nbead += resNAtom
                if resNAtom_max < resNAtom:
                    resNAtom_max = resNAtom
        if mode == "RES-BEAD":
            ret = np.zeros((nres, nres, resNAtom_max**2, 2), dtype=float)
        else:
            ret = np.zeros((nbead, nbead, 1, 2), dtype=float)

        def distance(r1: list, r2: list) -> float:
            return np.sqrt(sum([(x1-x2)**2 for x1, x2 in zip(r1, r2)]))

        ires1, ibead1, ires2, ibead2 = 1, 1, 1, 1
        for idchain1, chain1 in model.chains.items():
            for residue1 in chain1.residues:
                ires2, ibead2 = 1, 1
                for idchain2, chain2 in model.chains.items():
                    for residue2 in chain2.residues:
                        ibead1_offset = 0
                        # the following 'for' block rely on sequence of dict,
                        # which requires python 3.6 or newer
                        for bead1 in residue1.atoms.values():
                            ibead2_offset = 0
                            for bead2 in residue2.atoms.values():
                                distanceBead12 = distance(
                                    bead1["xyz"], bead2["xyz"])
                                connectStatBead12 = 0.0
                                if ibead1+ibead1_offset == ibead2+ibead2_offset:
                                    connectStatBead12 = 1.0
                                elif ires1 == ires2:
                                    connectStatBead12 = 2.0
                                elif idchain1 == idchain2:
                                    connectStatBead12 = 3.0
                                else:
                                    connectStatBead12 = 4.0
                                if mode == "RES-BEAD":
                                    ret[ires1-1, ires2-1, ibead1_offset*resNAtom_max +
                                        ibead2_offset] = np.array([distanceBead12, connectStatBead12])
                                else:
                                    ret[ibead1+ibead1_offset-1, ibead2+ibead2_offset-1,
                                        0] = np.array([distanceBead12, connectStatBead12])
                                ibead2_offset += 1
                            ibead1_offset += 1
                        ires2 += 1
                        ibead2 += residue2.natom
                ires1 += 1
                ibead1 += residue1.natom
        print("computation completed, return a distance matrix with shape ", ret.shape)
        return ret

    def encodingOneHot(self, imodel: int, cgin: str, mode: str = "RES") -> dict:
        print("one-hot encoding model %d of pdb structure %s on %s level, words defined by .cgin file %s" %
              (imodel, self.name, mode, cgin))
        ret = {}
        if imodel < 1 or imodel > self.nmodel:
            print("error: invalid parameter, pdb do not have %d-th model" %
                  imodel, file=sys.stderr)
            quit()
        if mode not in ["RES", "BEAD"]:
            print("error: invalid parameter, currently do not support computation in mode %s" %
                  mode, file=sys.stderr)
            quit()

        with open(cgin, 'r') as file:
            content = file.read()

        pattern = PATTERNS["cgin"]
        mobj = pattern.match(content)
        if not(mobj):
            print("error:invalid .cgin file %s" % cgin, file=sys.stderr)
            quit()

        tmpstr = mobj.group(2)
        cginRes = cginParser(tmpstr, "RESIDUE")
        tmpstr = mobj.group(3)
        cginResSBead = cginParser(tmpstr, "RES-BEAD")

        words = []
        if mode == "RES":
            words = cginRes
        else:
            for sres, sbeadList in cginResSBead.items():
                for sbead in sbeadList:
                    words.append(sres+sbead)

        model = self.models[imodel-1]
        for idchain, chain in model.chains.items():
            if mode == "RES":
                ret[idchain] = np.zeros(
                    (len(chain.seq3C), len(words)), dtype=int)
                for i, sres in enumerate(chain.seq3C):
                    if sres not in words:
                        print("warning: unrecognized residue %s, according to .cgin file %s" % (
                            sres, cgin), file=sys.stderr)
                    ret[idchain][i] = np.array(
                        [1 if resName == sres else 0 for resName in words])
            else:
                nbead = 0
                for residue in chain.residues:
                    nbead += residue.natom
                ret[idchain] = np.zeros((nbead, len(words)), dtype=int)
                nbead = 0
                for sres, residue in zip(chain.seq3C, chain.residues):
                    for sbead in residue.atoms.keys():
                        sresBead = sres+sbead
                        if sresBead not in words:
                            print("warning: unrecognized bead %s in residue %s, according to .cgin file %s" % (
                                sbead, sres, cgin), file=sys.stderr)
                        ret[idchain][nbead] = np.array(
                            [1 if beadName == sresBead else 0 for beadName in words])
                        nbead += 1
            print("chain %s encoded in a numpy array with shape" %
                  idchain, ret[idchain].shape)
        print("one-hot encoded")
        return ret

# physical properties support mass, charge
    def encodingPhyProperty(self, imodel: int, props=None, mode: str = "RES", ):
        print("encoding phycial properties of model %d of pdb structure %s on %s level" %
              (imodel, self.name, mode))
        print("phycical properties (", props, ")")
        if imodel < 1 or imodel > self.nmodel:
            print("error: invalid parameter, pdb do not have %d-th model" %
                  imodel, file=sys.stderr)
            quit()
        if mode not in ["RES", "BEAD"]:
            print("error: invalid parameter, currently do not support computation in mode %s" %
                  mode, file=sys.stderr)
            quit()

        ret = {}
        model = self.models[imodel-1]
        resNBead_max = 0
        for idchain, chain in model.chains.items():
            ret[idchain] = []
            for residue in chain.residues:
                resProps = []
                if resNBead_max < residue.natom:
                    resNBead_max = residue.natom
                for bead in residue.atoms.values():
                    beadProps = []
                    phyProps = copy.deepcopy(props)
                    while(phyProps):
                        sprop = phyProps.pop(0)
                        if sprop == "mass":
                            if "mass" in bead.keys():
                                beadProps.append(bead["mass"])
                                if bead["mass"] <= 0:
                                    print("warning: bead mass = 0.0 at (structure %s, model %d, chain %s, %d-th residue %s, bead %s)" % (
                                        self.name, residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1], bead["name"]),
                                        file=sys.stderr)
                            else:
                                print("error: bead mass required but not found at (structure %s, model %d, chain %s, %d-th residue %s, bead %s)" % (
                                    self.name, residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1], bead["name"]),
                                    file=sys.stderr)
                                quit()
                        if sprop == "charge":
                            if "charge" in bead.keys():
                                beadProps.append(bead["charge"])
                                if bead["charge"] == 0:
                                    print("warning: bead charge = 0.0 at (structure %s, model %d, chain %s, %d-th residue %s, bead %s)" % (
                                        self.name, residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1], bead["name"]),
                                        file=sys.stderr)
                            else:
                                print("error: bead charge required but not found at (structure %s, model %d, chain %s, %d-th residue %s, bead %s)" % (
                                    self.name, residue.loc[0], residue.loc[1], residue.loc[2][0], residue.loc[2][1], bead["name"]),
                                    file=sys.stderr)
                                quit()
                        # if sprop == "hydro":
                    beadProps = np.array(beadProps)[np.newaxis, :]
                    resProps.append(beadProps)
                ret[idchain].append(resProps)

        for idchain in model.chains.keys():
            chainProps = []
            for resProps in ret[idchain]:
                if mode == "RES":
                    for i in range(resNBead_max - len(resProps)):
                        resProps.append(np.zeros(len(props))[np.newaxis, :])
                    resProps = np.concatenate(resProps, axis=0)
                    chainProps.append(resProps[np.newaxis, :])
                else:
                    chainProps += resProps
            ret[idchain] = np.concatenate(chainProps, axis=0)
            print("chain %s encoded in a numpy array with shape" %
                  idchain, ret[idchain].shape)
        print("phycical properties (", props, ") encoded")
        return ret

# read contact energy from an amber energy decomposition file
    def readAmberEcon(self, imodel, aeconFile: str, mode: str = "RES-BEAD",
                      idchainMap: dict = {"1": 'A'}) -> np.ndarray:
        print("reading amber energy file %s to model %d of pdb structure %s in mode %s" %
              (aeconFile, imodel, self.name, mode))
        if imodel < 1 or imodel > self.nmodel:
            print("error: invalid parameter, pdb do not have %d-th model" %
                  imodel, file=sys.stderr)
            quit()
        # a res-level computation will create a res-res matrix
        #  with channels of bead-pair distance
        if mode not in ["RES-BEAD", "BEAD-BEAD"]:
            print("error: invalid mode %s, only support \'RES-BEAD\' and \'BEAD-BEAD\'" %
                  mode, file=sys.stderr)
            quit()

        model = self.models[imodel-1]
        nres, nbead = 0, 0
        resNAtom_max = 0
        nResBeads = []
        for chain in model.chains.values():
            nres += len(chain.seq3C)
            for residue in chain.residues:
                resNAtom = residue.natom
                nResBeads.append(resNAtom)
                nbead += resNAtom
                if resNAtom_max < resNAtom:
                    resNAtom_max = resNAtom
        if mode == "RES-BEAD":
            ret = np.zeros((nres, nres, resNAtom_max**2), dtype=float)
        else:
            ret = np.zeros((nbead, nbead, 1), dtype=float)

        with open(aeconFile, 'r') as file:
            lines = file.readlines()
        nlecIndex = -1
        for i, line in enumerate(lines):
            if len(line) >= 8 and line[:8] == " [nlocal":
                nlecIndex = i
                break
        if nlecIndex < 0:
            print("error: failed to find a non-local contact energy part in input file %s" %
                  aeconFile, file=sys.stderr)
            quit()

        nlecIndex += 1
        for line in lines[nlecIndex:]:
            pattern = PATTERNS["amberECon"]
            mobj = pattern.match(line)
            if not(mobj):
                print("error: invalid record %s in input energy file %s" %
                      (line, aeconFile), file=sys.stderr)
                quit()

            idchain1 = idchainMap[re.sub(r'\s', "", mobj.group(1))]
            idchain2 = idchainMap[re.sub(r'\s', "", mobj.group(2))]
            ires1 = int(re.sub(r'\s', "", mobj.group(3)))
            ires2 = int(re.sub(r'\s', "", mobj.group(4)))
            sres1 = re.sub(r'\s', "", mobj.group(5))
            sres2 = re.sub(r'\s', "", mobj.group(6))
            sbead1 = re.sub(r'\s', "", mobj.group(7))
            sbead2 = re.sub(r'\s', "", mobj.group(8))
            econ12 = float(re.sub(r'\s', "", mobj.group(9)))

            if idchain1 not in model.chains.keys():
                print("error: model %d of pro %s doesn\'t have chain %s" %
                      (imodel, self.name, idchain1), file=sys.stderr)
                quit()
            if idchain2 not in model.chains.keys():
                print("error: model %d of pro %s doesn\'t have chain %s" %
                      (imodel, self.name, idchain2), file=sys.stderr)
                quit()

            def getKeyIndex(dt: dict, key):
                for i, k in enumerate(dt.keys()):
                    if k == key:
                        return i
                return -1

            resBead1 = model.chains[idchain1].residues[ires1 -
                                                       model.chains[idchain1].residues[0].loc[2][0]].atoms
            ibead1_offset = getKeyIndex(resBead1, sbead1)
            if ibead1_offset == -1:
                print("error: model %d of pro %s doesn\'t have bead %s in %d-th residue %s" %
                      (imodel, self.name, sbead1, ires1, sres1), file=sys.stderr)
                quit()
            resBead2 = model.chains[idchain2].residues[ires2 -
                                                       model.chains[idchain2].residues[0].loc[2][0]].atoms
            ibead2_offset = getKeyIndex(resBead2, sbead2)
            if ibead2_offset == -1:
                print("error: model %d of pro %s doesn\'t have bead %s in %d-th residue %s" %
                      (imodel, self.name, sbead2, ires2, sres2), file=sys.stderr)
                quit()
            ibead1 = nResBeads[ires1-1]+ibead1_offset
            ibead2 = nResBeads[ires2-1]+ibead2_offset
            if mode == "RES-BEAD":
                ret[ires1-1, ires2-1, ibead1_offset *
                    resNAtom_max+ibead2_offset] = econ12
                ret[ires2-1, ires1-1, ibead2_offset *
                    resNAtom_max+ibead1_offset] = econ12
            else:
                ret[ibead1-1, ibead2-1, 0] = econ12
                ret[ibead2-1, ibead1-1, 0] = econ12
        print("amber energy file %s reading complete, return a energy matrix with shape" %
              aeconFile, ret.shape)
        return ret
