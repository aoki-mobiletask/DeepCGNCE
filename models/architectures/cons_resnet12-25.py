# data io control
inFeatureDirLocal = "../cgFeatures/"
inLabelDirLocal = "../cgLabels/"
inFeatureDirRemote = ""
inLabelDirRemote = ""
prolist = "list"

# training control
isDebugging: bool = False
showMeanStd: bool = False
doShowModelSize: bool = True
doLoadParams: bool = True
doTrain: bool = True
doEvaluate: bool = False
nVisualization: int = 5
paramSavePath: str = "cp"
historySavePath: str = "history"
filmSavePath: str = "film"

# training hyperparams
nepochs = 500
patience = 60
lr: list = [5e-2, 4e-2, 20]
batch_size: list = [1, 2]
moment: list = [0.9, 0.99, 0.01]
clip_grad = 1e16
clip_lambda = 0.02
optimizer = "adam_adaclip"
verbose = 1
visualization_clip = [-10., 0.]

# model arhitecture
nblocks1d = 4
arch1d = [
    [  # block1d 1
        {  # conv layer 1-1
            "io": [20, 8],
            "kernel_size":[21],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 1-2
            "io": [8, 16],
            "kernel_size":[21],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 1-3
            "io": [16, 16],
            "kernel_size":[21],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
    ],
    [  # block1d 2
        {  # conv layer 2-1
            "io": [16, 32],
            "kernel_size":[19],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 2-2
            "io": [32, 32],
            "kernel_size":[19],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 2-3
            "io": [32, 32],
            "kernel_size":[19],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
    ],
    [  # block1d 3
        {  # conv layer 3-1
            "io": [32, 64],
            "kernel_size":[17],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 3-2
            "io": [64, 64],
            "kernel_size":[17],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 3-3
            "io": [64, 64],
            "kernel_size":[17],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
    ],
    [  # block1d 4
        {  # conv layer 4-1
            "io": [64, 64],
            "kernel_size":[17],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 4-2
            "io": [64, 64],
            "kernel_size":[17],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
        {  # conv layer 4-3
            "io": [64, 4],
            "kernel_size":[17],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME",
        },
    ],
    {"activations": ["gelu"]*nblocks1d}
]
nblocks2d = 8
arch2d = [
    [  # block2d 1
        {  # conv layer 1-1
            "io": [9, 16],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 1-3
            "io": [16, 16],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 2
        {  # conv layer 2-1
            "io": [16, 32],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 2-3
            "io": [32, 32],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 2-3
            "io": [32, 32],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 3
        {  # conv layer 3-1
            "io": [32, 32],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 3-2
            "io": [32, 32],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 3-3
            "io": [32, 32],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 4
        {  # conv layer 4-1
            "io": [32, 64],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 4-2
            "io": [64, 64],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 4-3
            "io": [64, 64],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 5
        {  # conv layer 5-1
            "io": [64, 64],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 5-2
            "io": [64, 64],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 5-3
            "io": [64, 64],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 6
        {  # conv layer 6-1
            "io": [64, 128],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 6-2
            "io": [128, 128],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 6-3
            "io": [128, 128],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 7
        {  # conv layer 7-1
            "io": [128, 128],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 7-2
            "io": [128, 128],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # conv layer 7-3
            "io": [128, 128],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
    ],
    [  # block2d 8
        {  # conv layer 8-1
            "io": [128, 128],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
        {  # lambdaconv layer 8-2
            "dkvhCon": [128, 128, 128, 16, [7, 7]],
            "K":{
                "nlayers": 1,
                "kernel_size": [[3, 3]],
                "activations":["gelu"],
                "strides":[1],
                "paddings":["SAME"],
                "out_nchannels":[],
            },
            "V":{
                "nlayers": 1,
                "kernel_size": [[3, 3]],
                "activations":["gelu"],
                "strides":[1],
                "paddings":["SAME"],
                "out_nchannels":[],
            },
            "Q":{
                "nlayers": 1,
                "kernel_size": [[3, 3]],
                "activations":["gelu"],
                "strides":[1],
                "paddings":["SAME"],
                "out_nchannels":[],
            },
        },
        {  # conv layer 8-3
            "io": [128, 128],
            "kernel_size":[5, 5],
            "activation":"gelu",
            "strides":1,
            "padding":"SAME"
        },
    ],
    {"activations": ["gelu"]*nblocks2d}
]

output_nchannels = 1
