INDIRFEATURES_local = "../cgFeatures/"
INDIRLABEL_local = "../cgLabels/"
INDIRFEATURES_remote = "../cgFeatures/"
INDIRLABEL_remote = "../cgLabels/"

proList = "list"
paramsSavePath = "cp"
historySaveDir = "history"
filmsSaveDir = "film"
verbose = 1

shuffleSeed = 0
nepoch = 500
patience = 60
batch_size = 1
lr_max = 1e-4
lr_min = 1e-5
grad_norm_cutoff = 1e12
clip_min = -10.0
clip_max = 0.0

doLoadParams = False
doTrain = True
doEvaluate = True
nVisualize = 0

archSeqEnc = [
    {  # resblock1d 1
        "branch": [
            {
                "ioc": (23, 16),
                "ks": 21,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (16, 16),
                "ks": 21,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (16, 32),
                "ks": 21,
                "st": 1,
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock1d 2
        "branch": [
            {
                "ioc": (32, 32),
                "ks": 19,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (32, 32),
                "ks": 19,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (32, 64),
                "ks": 19,
                "st": 1,
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock1d 3
        "branch": [
            {
                "ioc": (64, 64),
                "ks": 17,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (64, 64),
                "ks": 17,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (64, 64),
                "ks": 17,
                "st": 1,
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock1d 4
        "branch": [
            {
                "ioc": (64, 64),
                "ks": 15,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (64, 64),
                "ks": 15,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (64, 128),
                "ks": 15,
                "st": 1,
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock1d 5
        "branch": [
            {
                "ioc": (128, 128),
                "ks": 21,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (128, 128),
                "ks": 21,
                "st": 1,
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (128, 64),
                "ks": 15,
                "st": 1,
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
]

archGraphEnc = []

archRO = [
    {  # resblock2d 1
        "branch": [
            {
                "ioc": (8, 16),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (16, 16),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (16, 16),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock2d 2
        "branch": [
            {
                "ioc": (16, 32),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (32, 32),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (32, 32),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock2d 3
        "branch": [
            {
                "ioc": (32, 64),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (64, 64),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (64, 64),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock2d 4
        "branch": [
            {
                "ioc": (64, 64),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (64, 64),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (64, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock2d 5
        "branch": [
            {
                "ioc": (128, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (128, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (128, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock2d 6
        "branch": [
            {
                "ioc": (128, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (128, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (128, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock2d 7
        "branch": [
            {
                "ioc": (128, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (128, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (128, 128),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
    {  # resblock2d 8
        "branch": [
            {
                "ioc": (128, 256),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (256, 256),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
                "act": "leaky_relu",
            },
            {
                "ioc": (256, 256),
                "ks": (3, 3),
                "st": (1, 1),
                "pd": "same",
            },
        ],
        "others": {
            "bw_init": 0.0,
            "act": "leaky_relu"
        }
    },
]
