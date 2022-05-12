# data io control
inFeatureDirLocal = "../cgFeatures/"
inLabelDirLocal = "../cgLabels/"
inFeatureDirRemote = ""
inLabelDirRemote = ""
prolist = "list2"

# training control
isDebugging: bool = False
showMeanStd: bool = False
doShowModelSize: bool = True
doLoadParams: bool = False
doTrain: bool = True
doEvaluate: bool = False
nVisualization: int = 5
paramSavePath: str = "cp"
historySavePath: str = "history"
filmSavePath: str = "film"

# training hyperparams
nepochs = 500
patience = 60
lr: list = [1e-2, 1e-3, 20]
batch_size: list = [1, 512]
moment: list = [0.9, 0.99, 0.01]
clip_grad = 1e16
clip_lambda = 0.02
optimizer = "adam_adaclip"
verbose = 1
visualization_clip = [-10., 0.]

# model arhitecture
nblocks1d = 3
arch1d = [
    [  # block1d 1
        {  # conv layer 1-1
            "io": [20, 16],
            "kernel_size":[19],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 1-2
            "io": [16, 32],
            "kernel_size":[19],
            "activation":"",
            "strides":1,
            "padding":"SAME",
        },
    ],
    [  # block1d 2
        {  # conv layer 2-1
            "io": [32, 32],
            "kernel_size":[17],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 2-2
            "io": [32, 64],
            "kernel_size":[17],
            "activation":"",
            "strides":1,
            "padding":"SAME",
        },
    ],
    [  # block1d 3
        {  # conv layer 3-1
            "io": [64, 64],
            "kernel_size":[15],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 3-2
            "io": [64, 3],
            "kernel_size":[15],
            "activation":"",
            "strides":1,
            "padding":"SAME",
        },
    ],
    {"activations": ["leaky_relu"]*nblocks1d}
]
input_nchannels = 7
nblocks2d = 8
arch2d = [
    [  # block2d 1
        {  # conv layer 1-1
            "io": [8, 16],
            "kernel_size":[3, 3],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 1-2
            "io": [16, 16],
            "kernel_size":[3, 3],
            "activation":"",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 2
        {  # conv layer 2-1
            "io": [16, 16],
            "kernel_size":[3, 3],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 2-2
            "io": [16, 16],
            "kernel_size":[3, 3],
            "activation":"",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 3
        {  # conv layer 3-1
            "io": [16, 32],
            "kernel_size":[3, 3],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 3-2
            "io": [32, 32],
            "kernel_size":[3, 3],
            "activation":"",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 4
        {  # conv layer 4-1
            "io": [32, 32],
            "kernel_size":[3, 3],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 4-2
            "io": [32, 32],
            "kernel_size":[3, 3],
            "activation":"",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 5
        {  # conv layer 5-1
            "io": [32, 64],
            "kernel_size":[3, 3],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 5-2
            "io": [64, 64],
            "kernel_size":[3, 3],
            "activation":"",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 6
        {  # conv layer 6-1
            "io": [64, 64],
            "kernel_size":[3, 3],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 6-2
            "io": [64, 64],
            "kernel_size":[3, 3],
            "activation":"",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 7
        {  # conv layer 7-1
            "io": [64, 128],
            "kernel_size":[3, 3],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 7-2
            "io": [128, 128],
            "kernel_size":[3, 3],
            "activation":"",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 8
        {  # conv layer 8-1
            "io": [128, 128],
            "kernel_size":[3, 3],
            "activation":"leaky_relu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 8-2
            "io": [128, 128],
            "kernel_size":[3, 3],
            "activation":"",
            "strides":1,
            "padding":"SAME"
        },
    ],
    {"activations": ["leaky_relu"]*nblocks2d}
]

output_nchannels = 1
