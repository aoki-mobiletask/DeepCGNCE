# this file is unnecessary for model-running
# as the name suggests, it's for testing whether a layer in 'layers.py' works well
# more specifically, "work well" here means:
# - output subjects to an approximately-normalized distribution (mean~0, stddev~1)
# - work as predicted

import layers
import tensorflow as tf
import numpy as np

# layer init
testlayer = layers.Conv2d(64, 64, activation="gelu", showMeanStd=True)
testlayer.config()
# generate data
N, H, W = 100, 256, 256
C = testlayer.ic
x = tf.random.normal([N, H, W, C], 0, 1)
# run
print(np.mean(x.numpy()), np.std(x.numpy()))
y = testlayer.forwardPropagation(x)
print(np.mean(y.numpy()), np.std(y.numpy()))
