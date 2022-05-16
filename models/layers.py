import activations
import optimizers
import numpy as np
import tensorflow as tf
import sys

countsize: float = 0.


class GlobalNorm:
    def __init__(self, mean=0, stddev=1):
        self.offset = mean
        self.scale = stddev

    def config(self, path: str = ""):
        if path:
            self.offset = tf.convert_to_tensor(
                np.load(path+"_goffset.npy"), dtype=tf.float32)
            self.scale = tf.convert_to_tensor(
                np.load(path+"_gscale.npy"), dtype=tf.float32)
        else:
            self.offset = tf.convert_to_tensor(self.offset, dtype=tf.float32)
            self.scale = tf.convert_to_tensor(self.scale, dtype=tf.float32)

    def storeWeights(self, path: str = ""):
        if path:
            np.save(path+"_goffset.npy", self.offset.numpy())
            np.save(path+"_gscale.npy", self.scale.numpy())

    def forwardPropagation(self, x):
        return (x-self.offset)/self.scale


class Conv1d:
    def __init__(self, input_nchannels, output_nchannels,  kernel_size=[3],
                 activation='leaky_relu', strides=[1, 1, 1], padding='SAME',
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.ic = input_nchannels
        self.oc = output_nchannels
        if type(kernel_size) is int:
            self.ks = [kernel_size]
        else:
            self.ks = list(kernel_size)
        self.activation = activations.summary[activation]
        self.optimizer = optimizers.summary[optimizer]
        self.optimizer = optimizers.adam_adaclip
        self.strides = [1, 1, 1]
        if type(strides) is int:
            self.strides = [1, strides, 1]
        elif len(strides) == 1:
            self.strides = [1, strides[0], 1]
        elif len(strides) >= 3:
            self.strides = strides[:3]
        self.padding = padding
        self.weights = []
        self.m1wbs = []
        self.m2wbs = []
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        if path:
            w = tf.Variable(tf.convert_to_tensor(
                np.load(path+"_w1d.npy"), dtype=tf.float32))
            b = tf.Variable(tf.convert_to_tensor(
                np.load(path+"_b1d.npy"), dtype=tf.float32))
        else:
            shape = self.ks+[self.ic, self.oc]
            w = tf.Variable(tf.random.truncated_normal(
                shape=shape, mean=0, stddev=np.math.sqrt(1.31/np.prod(shape[: -1]))))
            b = tf.Variable(tf.zeros(self.oc))
        self.weights = [w, b]

        m1w, m1b = tf.zeros_like(w), tf.zeros_like(b)
        m2w, m2b = tf.zeros_like(w), tf.zeros_like(b)
        self.m1wbs = [m1w, m1b]
        self.m2wbs = [m2w, m2b]

        global countsize
        countsize += (sys.getsizeof(w)+sys.getsizeof(b))*2

    def storeWeights(self, path: str = ""):
        if path:
            np.save(path+"_w1d.npy", self.weights[0].numpy())
            np.save(path+"_b1d.npy", self.weights[1].numpy())

    def forwardPropagation(self, x):
        x = tf.nn.conv1d(
            x, self.weights[0], stride=self.strides, padding=self.padding)
        x = tf.nn.bias_add(x, self.weights[1])
        if self.showMeanStd:
            print("conv1d", tf.math.reduce_mean(x).numpy(),
                  tf.math.reduce_std(x).numpy())
        y = self.activation(x)
        return y

    def updateWeights(self, grads, args):
        for i in range(len(self.weights)):
            self.optimizer(self.weights[i], grads[i], [
                self.m1wbs[i], self.m2wbs[i]], args)


class Conv2d:
    def __init__(self, input_nchannels, output_nchannels,  kernel_size=[3, 3],
                 activation='leaky_relu', strides=[1, 1, 1, 1], padding='SAME',
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=True):
        self.ic = input_nchannels
        self.oc = output_nchannels
        if type(kernel_size) is int:
            self.ks = [kernel_size, kernel_size]
        else:
            self.ks = list(kernel_size)
        self.activation = activations.summary[activation]
        self.optimizer = optimizers.summary[optimizer]
        self.strides = [1, 1, 1, 1]
        if type(strides) is int:
            self.strides = [1, strides, strides, 1]
        elif len(strides) == 1:
            self.strides = [1, strides[0], strides[0], 1]
        elif len(strides) == 2:
            self.strides = [1, strides[0], strides[1], 1]
        elif len(strides) >= 4:
            self.strides = strides[:4]
        self.padding = padding
        self.weights = []
        self.m1wbs = []
        self.m2wbs = []
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        if path:
            w = tf.Variable(tf.convert_to_tensor(
                np.load(path+"_w2d.npy"), dtype=tf.float32))
            b = tf.Variable(tf.convert_to_tensor(
                np.load(path+"_b2d.npy"), dtype=tf.float32))
        else:
            shape = self.ks+[self.ic, self.oc]
            w = tf.Variable(tf.random.truncated_normal(
                shape=shape, mean=0, stddev=np.math.sqrt(1.31/np.prod(shape[: -1]))))
            b = tf.Variable(tf.zeros(self.oc))
        self.weights = [w, b]

        m1w, m1b = tf.zeros_like(w), tf.zeros_like(b)
        m2w, m2b = tf.zeros_like(w), tf.zeros_like(b)
        self.m1wbs = [m1w, m1b]
        self.m2wbs = [m2w, m2b]

        global countsize
        countsize += (sys.getsizeof(w)+sys.getsizeof(b))*2

    def storeWeights(self, path: str = ""):
        if path:
            np.save(path+"_w2d.npy", self.weights[0].numpy())
            np.save(path+"_b2d.npy", self.weights[1].numpy())

    def forwardPropagation(self, x):
        x = tf.nn.conv2d(
            x, self.weights[0], strides=self.strides, padding=self.padding)
        x = tf.nn.bias_add(x, self.weights[1])
        if self.showMeanStd:
            print("conv2d", tf.math.reduce_mean(x).numpy(),
                  tf.math.reduce_std(x).numpy())
        y = self.activation(x)
        return y

    def updateWeights(self, grads, args):
        for i in range(len(self.weights)):
            self.optimizer(self.weights[i], grads[i], [
                self.m1wbs[i], self.m2wbs[i]], args)


class Conv3d:
    def __init__(self, input_nchannels, output_nchannels,  kernel_size=[3, 3, 3],
                 activation='leaky_relu', strides=[1, 1, 1, 1, 1], padding='SAME',
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.ic = input_nchannels
        self.oc = output_nchannels
        if type(kernel_size) is int:
            self.ks = [kernel_size, kernel_size, kernel_size]
        else:
            self.ks = list(kernel_size)
        self.activation = activations.summary[activation]
        self.optimizer = optimizers.summary[optimizer]
        self.strides = [1, 1, 1, 1, 1]
        if type(strides) is int:
            self.strides = [1, strides, strides, strides, 1]
        elif len(strides) == 1:
            self.strides = [1, strides[0], strides[0], strides[0], 1]
        elif len(strides) == 3:
            self.strides = [1]+list(strides)+[1]
        elif len(strides) >= 5:
            self.strides = strides[:5]
        self.padding = padding
        self.weights = []
        self.m1wbs = []
        self.m2wbs = []
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        if path:
            w = tf.Variable(tf.convert_to_tensor(
                np.load(path+"_w3d.npy"), dtype=tf.float32))
            b = tf.Variable(tf.convert_to_tensor(
                np.load(path+"_b3d.npy"), dtype=tf.float32))
        else:
            shape = self.ks+[self.ic, self.oc]
            w = tf.Variable(tf.random.truncated_normal(
                shape=shape, mean=0, stddev=np.math.sqrt(1.31/np.prod(shape[: -1]))))
            b = tf.Variable(tf.zeros(self.oc))
        self.weights = [w, b]

        m1w, m1b = tf.zeros_like(w), tf.zeros_like(b)
        m2w, m2b = tf.zeros_like(w), tf.zeros_like(b)
        self.m1wbs = [m1w, m1b]
        self.m2wbs = [m2w, m2b]

        global countsize
        countsize += (sys.getsizeof(w)+sys.getsizeof(b))*2

    def storeWeights(self, path: str = ""):
        if path:
            np.save(path+"_w3d.npy", self.weights[0].numpy())
            np.save(path+"_b3d.npy", self.weights[1].numpy())

    def forwardPropagation(self, x):
        x = tf.nn.conv3d(
            x, self.weights[0], strides=self.strides, padding=self.padding)
        x = tf.nn.bias_add(x, self.weights[1])
        if self.showMeanStd:
            print("conv3d", tf.math.reduce_mean(x).numpy(),
                  tf.math.reduce_std(x).numpy())
        y = self.activation(x)
        return y

    def updateWeights(self, grads, args):
        for i in range(len(self.weights)):
            self.optimizer(self.weights[i], grads[i], [
                self.m1wbs[i], self.m2wbs[i]], args)


class LambdaConv1d:
    def __init__(self, input_nchannels, output_nchannels, intermediate_nchannels,
                 head_nchannels=16, context_size=[21],
                 K_nlayers=1, K_kernel_sizes=[[17]], K_activations=["leaky_relu"],
                 K_strides=[1], K_paddings=["SAME"], K_out_nchannels: list = [],
                 V_nlayers=1, V_kernel_sizes=[[17]], V_activations=["leaky_relu"],
                 V_strides=[1], V_paddings=["SAME"], V_out_nchannels: list = [],
                 Q_nlayers=1, Q_kernel_sizes=[[17]], Q_activations=["leaky_relu"],
                 Q_strides=[1], Q_paddings=["SAME"], Q_out_nchannels: list = [],
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.d = input_nchannels
        self.v = output_nchannels
        self.k = intermediate_nchannels
        self.h = head_nchannels
        self.K_layers = []
        K_out_nchannels.insert(0, self.d)
        K_out_nchannels.append(self.k)
        for i in range(K_nlayers):
            if debug:
                print("K", K_kernel_sizes[i])
            layer = Conv1d(K_out_nchannels[i], K_out_nchannels[i+1],
                           K_kernel_sizes[i], K_activations[i],  K_strides[i], K_paddings[i],
                           optimizer, debug=debug, showMeanStd=showMeanStd)
            self.K_layers.append(layer)
        self.V_layers = []
        V_out_nchannels.insert(0, self.d)
        V_out_nchannels.append(self.v)
        for i in range(V_nlayers):
            if debug:
                print("V", V_kernel_sizes[i])
            layer = Conv1d(V_out_nchannels[i], V_out_nchannels[i+1],
                           V_kernel_sizes[i], V_activations[i],  V_strides[i], V_paddings[i],
                           optimizer, debug=debug, showMeanStd=showMeanStd)
            self.V_layers.append(layer)
        self.Q_layers = []
        Q_out_nchannels.insert(0, self.d)
        Q_out_nchannels.append(self.k)
        for i in range(Q_nlayers):
            if debug:
                print("Q", Q_kernel_sizes[i])
            layer = Conv1d(Q_out_nchannels[i], Q_out_nchannels[i+1],
                           Q_kernel_sizes[i], Q_activations[i],  Q_strides[i], Q_paddings[i],
                           optimizer, debug=debug, showMeanStd=showMeanStd)
            self.Q_layers.append(layer)
        if type(context_size) is int:
            context_size = [context_size]
        else:
            context_size = list(context_size)
        self.Epos_layers = []
        for i in range(int(self.k/self.h)):
            layer = Conv2d(1, self.h, context_size+[1], "",
                           optimizer=optimizer, debug=debug, showMeanStd=showMeanStd)
            self.Epos_layers.append(layer)
        if self.k % self.h > 0:
            layer = Conv2d(1, self.k-len(self.Epos_layers)*self.h, context_size+[1], "",
                           optimizer=optimizer, debug=debug, showMeanStd=showMeanStd)
            self.Epos_layers.append(layer)
        self.weights = []
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        K_weights, V_weights, Q_weights = [], [], []
        Epos_weights = []
        if path:
            for i, layer in enumerate(self.K_layers):
                layer.config(path+"_K"+str(i))
                K_weights.append(layer.weights)
            for i, layer in enumerate(self.V_layers):
                layer.config(path+"_V"+str(i))
                V_weights.append(layer.weights)
            for i, layer in enumerate(self.Q_layers):
                layer.config(path+"_Q"+str(i))
                Q_weights.append(layer.weights)
            for i, layer in enumerate(self.Epos_layers):
                layer.config(path+"_Epos"+str(i))
                Epos_weights.append(layer.weights)
        else:
            for i, layer in enumerate(self.K_layers):
                layer.config()
                K_weights.append(layer.weights)
            for i, layer in enumerate(self.V_layers):
                layer.config()
                V_weights.append(layer.weights)
            for i, layer in enumerate(self.Q_layers):
                layer.config()
                Q_weights.append(layer.weights)
            for i, layer in enumerate(self.Epos_layers):
                layer.config()
                Epos_weights.append(layer.weights)
        self.weights = [K_weights, V_weights, Q_weights, Epos_weights]

    def storeWeights(self, path: str = ""):
        if path:
            for i, layer in enumerate(self.K_layers):
                layer.storeWeights(path + "_K"+str(i))
            for i, layer in enumerate(self.V_layers):
                layer.storeWeights(path + "_V"+str(i))
            for i, layer in enumerate(self.Q_layers):
                layer.storeWeights(path + "_Q"+str(i))
            for i, layer in enumerate(self.Epos_layers):
                layer.storeWeights(path+"_Epos"+str(i))

    def forwardPropagation(self, x: tf.Tensor):
        k, v, q = x, x, x  # key value, query
        for i, layer in enumerate(self.K_layers):
            k = layer.forwardPropagation(k)
            if self.showMeanStd:
                print("k"+str(i), tf.math.reduce_mean(k).numpy(),
                      tf.math.reduce_std(k).numpy())
        for i, layer in enumerate(self.V_layers):
            v = layer.forwardPropagation(v)
            if self.showMeanStd:
                print("v"+str(i), tf.math.reduce_mean(v).numpy(),
                      tf.math.reduce_std(v).numpy())
        for i, layer in enumerate(self.Q_layers):
            q = layer.forwardPropagation(q)
            if self.showMeanStd:
                print("q"+str(i), tf.math.reduce_mean(q).numpy(),
                      tf.math.reduce_std(q).numpy())
        L = k.shape[1]
        k = tf.transpose(tf.nn.softmax(k[0], axis=0))
        content_lambda = tf.matmul(k, v[0])*np.math.sqrt(L/self.k/np.math.e)
        y1 = tf.nn.conv1d(q, content_lambda[tf.newaxis, ...], stride=[
                          1, 1, 1], padding="VALID")
        if self.showMeanStd:
            print("y1", tf.math.reduce_mean(y1).numpy(),
                  tf.math.reduce_std(y1).numpy())
        y2s = []
        nEpos_layers = len(self.Epos_layers)
        interval = int(self.v/nEpos_layers)
        for i, layer in enumerate(self.Epos_layers):
            if i < nEpos_layers-1:
                position_lambdas = layer.forwardPropagation(
                    v[:, :, interval*i:interval*(i+1), tf.newaxis])
                y2 = tf.squeeze(tf.matmul(
                    position_lambdas, q[:, :, self.h*i:self.h*(i+1), tf.newaxis]), axis=[3])
            else:
                position_lambdas = layer.forwardPropagation(
                    v[:, :, interval*i:, tf.newaxis])
                y2 = tf.squeeze(tf.matmul(position_lambdas,
                                q[:, :, self.h*i:, tf.newaxis]), axis=[3])
            y2s.append(y2)
        y2 = tf.concat(y2s, axis=2)/np.math.sqrt(self.h)
        if self.showMeanStd:
            print("y2", tf.math.reduce_mean(y2).numpy(),
                  tf.math.reduce_std(y2).numpy())
        y = (y1+y2)/1.32
        if self.showMeanStd:
            print("y", tf.math.reduce_mean(y).numpy(),
                  tf.math.reduce_std(y).numpy())
        return y

    def updateWeights(self, grads, args):
        for i, layer in enumerate(self.K_layers):
            layer.updateWeights(grads[0][i], args)
        for i, layer in enumerate(self.V_layers):
            layer.updateWeights(grads[1][i], args)
        for i, layer in enumerate(self.Q_layers):
            layer.updateWeights(grads[2][i], args)
        for i, layer in enumerate(self.Epos_layers):
            layer.updateWeights(grads[3][i], args)


class LambdaConv2d:
    def __init__(self, input_nchannels, output_nchannels, intermediate_nchannels,
                 head_nchannels=16, context_size=[9, 9],
                 K_nlayers=1, K_kernel_sizes=[[3, 3]], K_activations=["leaky_relu"],
                 K_strides=[1], K_paddings=["SAME"], K_out_nchannels: list = [],
                 V_nlayers=1, V_kernel_sizes=[[3, 3]], V_activations=["leaky_relu"],
                 V_strides=[1], V_paddings=["SAME"], V_out_nchannels: list = [],
                 Q_nlayers=1, Q_kernel_sizes=[[3, 3]], Q_activations=["leaky_relu"],
                 Q_strides=[1], Q_paddings=["SAME"], Q_out_nchannels: list = [],
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.d = input_nchannels
        self.v = output_nchannels
        self.k = intermediate_nchannels
        self.h = head_nchannels
        self.K_layers = []
        K_out_nchannels = [self.d]+K_out_nchannels+[self.k]
        for i in range(K_nlayers):
            layer = Conv2d(K_out_nchannels[i], K_out_nchannels[i+1],
                           K_kernel_sizes[i], K_activations[i],  K_strides[i], K_paddings[i],
                           optimizer, debug=debug, showMeanStd=showMeanStd)
            self.K_layers.append(layer)
        self.V_layers = []
        V_out_nchannels = [self.d]+V_out_nchannels+[self.v]
        for i in range(V_nlayers):
            layer = Conv2d(V_out_nchannels[i], V_out_nchannels[i+1],
                           V_kernel_sizes[i], V_activations[i],  V_strides[i], V_paddings[i],
                           optimizer, debug=debug, showMeanStd=showMeanStd)
            self.V_layers.append(layer)
        self.Q_layers = []
        Q_out_nchannels = [self.d]+Q_out_nchannels+[self.k]
        for i in range(Q_nlayers):
            layer = Conv2d(Q_out_nchannels[i], Q_out_nchannels[i+1],
                           Q_kernel_sizes[i], Q_activations[i],  Q_strides[i], Q_paddings[i],
                           optimizer, debug=debug, showMeanStd=showMeanStd)
            self.Q_layers.append(layer)
        if type(context_size) is int:
            context_size = [context_size, context_size]
        else:
            context_size = list(context_size)
        self.Epos_layers = []
        for i in range(int(self.k/self.h)):
            layer = Conv3d(1, self.h, context_size+[1], "",
                           optimizer=optimizer, debug=debug, showMeanStd=showMeanStd)
            self.Epos_layers.append(layer)
        if self.k % self.h > 0:
            layer = Conv3d(1, self.k-self.h*len(self.Epos_layers), context_size+[1], "",
                           optimizer=optimizer, debug=debug, showMeanStd=showMeanStd)
            self.Epos_layers.append(layer)
        self.weights = []
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        K_weights, V_weights, Q_weights = [], [], []
        Epos_weights = []
        if path:
            for i, layer in enumerate(self.K_layers):
                layer.config(path+"_K"+str(i))
                K_weights.append(layer.weights)
            for i, layer in enumerate(self.V_layers):
                layer.config(path+"_V"+str(i))
                V_weights.append(layer.weights)
            for i, layer in enumerate(self.Q_layers):
                layer.config(path+"_Q"+str(i))
                Q_weights.append(layer.weights)
            for i, layer in enumerate(self.Epos_layers):
                layer.config(path+"_Epos"+str(i))
                Epos_weights.append(layer.weights)
        else:
            for i, layer in enumerate(self.K_layers):
                layer.config()
                K_weights.append(layer.weights)
            for i, layer in enumerate(self.V_layers):
                layer.config()
                V_weights.append(layer.weights)
            for i, layer in enumerate(self.Q_layers):
                layer.config()
                Q_weights.append(layer.weights)
            for i, layer in enumerate(self.Epos_layers):
                layer.config()
                Epos_weights.append(layer.weights)
        self.weights = [K_weights, V_weights, Q_weights, Epos_weights]

    def storeWeights(self, path: str = ""):
        if path:
            for i, layer in enumerate(self.K_layers):
                layer.storeWeights(path + "_K"+str(i))
            for i, layer in enumerate(self.V_layers):
                layer.storeWeights(path + "_V"+str(i))
            for i, layer in enumerate(self.Q_layers):
                layer.storeWeights(path + "_Q"+str(i))
            for i, layer in enumerate(self.Epos_layers):
                layer.storeWeights(path + "_Epos"+str(i))

    def forwardPropagation(self, x: tf.Tensor):
        k, v, q = x, x, x
        for i, layer in enumerate(self.K_layers):
            k = layer.forwardPropagation(k)
            if self.showMeanStd:
                print("k"+str(i), tf.math.reduce_mean(k).numpy(),
                      tf.math.reduce_std(k).numpy())
        for i, layer in enumerate(self.V_layers):
            v = layer.forwardPropagation(v)
            if self.showMeanStd:
                print("v"+str(i), tf.math.reduce_mean(v).numpy(),
                      tf.math.reduce_std(v).numpy())
        for i, layer in enumerate(self.Q_layers):
            q = layer.forwardPropagation(q)
            if self.showMeanStd:
                print("q"+str(i), tf.math.reduce_mean(q).numpy(),
                      tf.math.reduce_std(q).numpy())
        L = k.shape[1]
        k = tf.nn.softmax(tf.reshape(tf.transpose(
            k[0], [2, 0, 1]), [self.k, -1]), axis=1)
        content_lambda = tf.matmul(k, tf.transpose(tf.reshape(tf.transpose(v[0], [2, 0, 1]), [
                                   self.v, -1])))*(L**0.1)/0.06
        y1 = tf.nn.conv2d(q, content_lambda[tf.newaxis, tf.newaxis, ...], strides=[
                          1, 1, 1, 1], padding="VALID")/np.math.sqrt(self.k)
        if self.showMeanStd:
            print("y1", tf.math.reduce_mean(y1).numpy(),
                  tf.math.reduce_std(y1).numpy())
        y2s = []
        nEpos_layers = len(self.Epos_layers)
        interval = int(self.v/nEpos_layers)
        for i, layer in enumerate(self.Epos_layers):
            if i < nEpos_layers-1:
                position_lambdas = layer.forwardPropagation(
                    v[:, :, :, interval*i:interval*(i+1), tf.newaxis])
                y2 = tf.squeeze(tf.matmul(
                    position_lambdas, q[:, :, :, self.h*i:self.h*(i+1), tf.newaxis]), axis=[4])
            else:
                position_lambdas = layer.forwardPropagation(
                    v[:, :, :, interval*i:, tf.newaxis])
                y2 = tf.squeeze(tf.matmul(position_lambdas,
                                q[:, :, :, self.h*i:, tf.newaxis]), axis=[4])
            y2s.append(y2)
        y2 = tf.concat(y2s, axis=3)/np.math.sqrt(self.h)
        if self.showMeanStd:
            print("y2", tf.math.reduce_mean(y2).numpy(),
                  tf.math.reduce_std(y2).numpy())
        y = (y1+y2)/1.41
        if self.showMeanStd:
            print("y", tf.math.reduce_mean(y).numpy(),
                  tf.math.reduce_std(y).numpy())
        return y

    def updateWeights(self, grads, args):
        for i, layer in enumerate(self.K_layers):
            layer.updateWeights(grads[0][i], args)
        for i, layer in enumerate(self.V_layers):
            layer.updateWeights(grads[1][i], args)
        for i, layer in enumerate(self.Q_layers):
            layer.updateWeights(grads[2][i], args)
        for i, layer in enumerate(self.Epos_layers):
            layer.updateWeights(grads[3][i], args)


class LambdaResnetBlock1d:
    def __init__(self, nLambdaLayers: int, arch_br: list, activation="leaky_relu",
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.blkic = arch_br[0]["dkvhCon"][0] if (
            "K" in arch_br[0].keys()) else arch_br[0]["io"][0]
        self.blkoc = arch_br[-1]["dkvhCon"][2] if (
            "K" in arch_br[-1].keys()) else arch_br[-1]["io"][1]
        self.convert_layer = Conv1d(self.blkic, self.blkoc, [1], "", [1, 1, 1], "SAME",
                                    optimizer, debug=debug, showMeanStd=showMeanStd)
        self.branch_layers = []
        for i in range(nLambdaLayers):
            if "K" in arch_br[i].keys():
                d, k, v, h, context_size = arch_br[i]["dkvhCon"]
                archK = arch_br[i]["K"]
                archQ = arch_br[i]["Q"]
                archV = arch_br[i]["V"]
                branch_layer = LambdaConv1d(d, v, k, h, context_size,
                                            archK["nlayers"], archK["kernel_size"], archK["activations"],
                                            archK["strides"], archK["paddings"], archK["out_nchannels"],
                                            archV["nlayers"], archV["kernel_size"], archV["activations"],
                                            archV["strides"], archV["paddings"], archV["out_nchannels"],
                                            archQ["nlayers"], archQ["kernel_size"], archQ["activations"],
                                            archQ["strides"], archQ["paddings"], archQ["out_nchannels"],
                                            optimizer, debug=debug, showMeanStd=showMeanStd)
            else:
                ic, oc = arch_br[i]["io"]
                activation = arch_br[i]["activation"] if (
                    i < nLambdaLayers-1) else ""
                branch_layer = Conv1d(ic, oc, arch_br[i]["kernel_size"],
                                      activation, arch_br[i]["strides"], arch_br[i]["padding"],
                                      optimizer, debug=debug, showMeanStd=showMeanStd)
            self.branch_layers.append(branch_layer)
        self.weights = []
        self.m1brsc, self.m2brsc = 0., 0.
        self.activation = activations.summary[activation]
        self.optimizer = optimizers.summary[optimizer]
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        branch_weights = []
        if path:
            self.convert_layer.config(path+"_id")
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.config(path+"_br"+str(i))
                branch_weights.append(branch_layer.weights)
            branch_scale = tf.Variable(tf.convert_to_tensor(
                np.load(path+"_brsc.npy"), dtype=tf.float32))
        else:
            self.convert_layer.config()
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.config()
                branch_weights.append(branch_layer.weights)
            branch_scale = tf.Variable(
                tf.convert_to_tensor(0., dtype=tf.float32))
        self.weights = [self.convert_layer.weights,
                        branch_weights, branch_scale]

        self.m1brsc, self.m2brsc = tf.zeros_like(
            branch_scale), tf.zeros_like(branch_scale)

    def storeWeights(self, path: str = ""):
        if path:
            self.convert_layer.storeWeights(path+"_id")
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.storeWeights(path+"_br"+str(i))
            np.save(path+"_brsc.npy", self.weights[2].numpy())

    def forwardPropagation(self, x):
        if self.blkic != self.blkoc:
            residual = self.convert_layer.forwardPropagation(x)
        else:
            residual = x
        for i, branch_layer in enumerate(self.branch_layers):
            x = branch_layer.forwardPropagation(x)
            if self.showMeanStd:
                print("br"+str(i), tf.math.reduce_mean(x).numpy(),
                      tf.math.reduce_std(x).numpy())
        y = (x+self.weights[2]*residual)/tf.sqrt(tf.square(self.weights[2])+1)
        y = self.activation(y)
        if self.showMeanStd:
            print("br", tf.math.reduce_mean(y).numpy(),
                  tf.math.reduce_std(y).numpy())
        return y

    def updateWeights(self, grads, args):
        if self.blkic != self.blkoc:
            self.convert_layer.updateWeights(grads[0], args)
        for i, branch_layer in enumerate(self.branch_layers):
            branch_layer.updateWeights(grads[1][i], args)
        # branch scale update here
        self.optimizer(self.weights[2], grads[2], [
                       self.m1brsc, self.m2brsc], args)


class LambdaResnetBlock2d:
    def __init__(self, nLambdaLayers: int, arch_br: list, activation="leaky_relu",
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.blkic = arch_br[0]["dkvhCon"][0] if (
            "K" in arch_br[0].keys()) else arch_br[0]["io"][0]
        self.blkoc = arch_br[-1]["dkvhCon"][2] if (
            "K" in arch_br[-1].keys()) else arch_br[-1]["io"][1]
        self.convert_layer = Conv2d(self.blkic, self.blkoc, [1, 1], "", [1, 1, 1, 1], "SAME",
                                    optimizer, debug=debug, showMeanStd=showMeanStd)
        self.branch_layers = []
        for i in range(nLambdaLayers):
            if "K" in arch_br[i].keys():
                d, k, v, h, context_size = arch_br[i]["dkvhCon"]
                archK = arch_br[i]["K"]
                archQ = arch_br[i]["Q"]
                archV = arch_br[i]["V"]
                branch_layer = LambdaConv2d(d, v, k, h, context_size,
                                            archK["nlayers"], archK["kernel_size"], archK["activations"],
                                            archK["strides"], archK["paddings"], archK["out_nchannels"],
                                            archV["nlayers"], archV["kernel_size"], archV["activations"],
                                            archV["strides"], archV["paddings"], archV["out_nchannels"],
                                            archQ["nlayers"], archQ["kernel_size"], archQ["activations"],
                                            archQ["strides"], archQ["paddings"], archQ["out_nchannels"],
                                            optimizer, debug=debug, showMeanStd=showMeanStd)
            else:
                ic, oc = arch_br[i]["io"]
                activation = arch_br[i]["activation"] if (
                    i < nLambdaLayers-1) else ""
                branch_layer = Conv2d(ic, oc, arch_br[i]["kernel_size"],
                                      activation, arch_br[i]["strides"], arch_br[i]["padding"],
                                      optimizer, debug=debug, showMeanStd=showMeanStd)
            self.branch_layers.append(branch_layer)
        self.weights = []
        self.m1brsc, self.m2brsc = 0., 0.
        self.activation = activations.summary[activation]
        self.optimizer = optimizers.summary[optimizer]
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        branch_weights = []
        if path:
            self.convert_layer.config(path+"_id")
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.config(path+"_br"+str(i))
                branch_weights.append(branch_layer.weights)
            branch_scale = tf.Variable(tf.convert_to_tensor(
                np.load(path+"_brsc.npy"), dtype=tf.float32))
        else:
            self.convert_layer.config()
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.config()
                branch_weights.append(branch_layer.weights)
            branch_scale = tf.Variable(
                tf.convert_to_tensor(0., dtype=tf.float32))
        self.weights = [self.convert_layer.weights,
                        branch_weights, branch_scale]

        self.m1brsc, self.m2brsc = tf.zeros_like(
            branch_scale), tf.zeros_like(branch_scale)

    def storeWeights(self, path: str = ""):
        if path:
            self.convert_layer.storeWeights(path+"_id")
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.storeWeights(path+"_br"+str(i))
            np.save(path+"_brsc.npy", self.weights[2].numpy())

    def forwardPropagation(self, x):
        if self.blkic != self.blkoc:
            residual = self.convert_layer.forwardPropagation(x)
        else:
            residual = x
        for i, branch_layer in enumerate(self.branch_layers):
            x = branch_layer.forwardPropagation(x)
            if self.showMeanStd:
                print("br"+str(i), tf.math.reduce_mean(x).numpy(),
                      tf.math.reduce_std(x).numpy())
        y = (x+self.weights[2]*residual)/tf.sqrt(tf.square(self.weights[2])+1)
        y = self.activation(y)
        if self.showMeanStd:
            print("br", tf.math.reduce_mean(y).numpy(),
                  tf.math.reduce_std(y).numpy())
        return y

    def updateWeights(self, grads, args):
        if self.blkic != self.blkoc:
            self.convert_layer.updateWeights(grads[0], args)
        for i, branch_layer in enumerate(self.branch_layers):
            branch_layer.updateWeights(grads[1][i], args)
        # branch scale update here
        self.optimizer(self.weights[2], grads[2], [
                       self.m1brsc, self.m2brsc], args)


# without branch rescale
class LambdaResBlk1d:
    def __init__(self, nLambdaLayers: int, arch_br: list, activation="leaky_relu",
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.blkic = arch_br[0]["dkvhCon"][0] if (
            "K" in arch_br[0].keys()) else arch_br[0]["io"][0]
        self.blkoc = arch_br[-1]["dkvhCon"][2] if (
            "K" in arch_br[-1].keys()) else arch_br[-1]["io"][1]
        self.convert_layer = Conv1d(self.blkic, self.blkoc, [1], "", [1, 1, 1], "SAME",
                                    optimizer, debug=debug, showMeanStd=showMeanStd)
        self.branch_layers = []
        for i in range(nLambdaLayers):
            if "K" in arch_br[i].keys():
                d, k, v, h, context_size = arch_br[i]["dkvhCon"]
                archK = arch_br[i]["K"]
                archQ = arch_br[i]["Q"]
                archV = arch_br[i]["V"]
                branch_layer = LambdaConv1d(d, v, k, h, context_size,
                                            archK["nlayers"], archK["kernel_size"], archK["activations"],
                                            archK["strides"], archK["paddings"], archK["out_nchannels"],
                                            archV["nlayers"], archV["kernel_size"], archV["activations"],
                                            archV["strides"], archV["paddings"], archV["out_nchannels"],
                                            archQ["nlayers"], archQ["kernel_size"], archQ["activations"],
                                            archQ["strides"], archQ["paddings"], archQ["out_nchannels"],
                                            optimizer, debug=debug, showMeanStd=showMeanStd)
            else:
                ic, oc = arch_br[i]["io"]
                activation = arch_br[i]["activation"] if (
                    i < nLambdaLayers-1) else ""
                branch_layer = Conv1d(ic, oc, arch_br[i]["kernel_size"],
                                      activation, arch_br[i]["strides"], arch_br[i]["padding"],
                                      optimizer, debug=debug, showMeanStd=showMeanStd)
            self.branch_layers.append(branch_layer)
        self.weights = []
        self.activation = activations.summary[activation]
        self.optimizer = optimizers.summary[optimizer]
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        branch_weights = []
        if path:
            self.convert_layer.config(path+"_id")
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.config(path+"_br"+str(i))
                branch_weights.append(branch_layer.weights)
        else:
            self.convert_layer.config()
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.config()
                branch_weights.append(branch_layer.weights)
        self.weights = [self.convert_layer.weights, branch_weights]

    def storeWeights(self, path: str = ""):
        if path:
            self.convert_layer.storeWeights(path+"_id")
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.storeWeights(path+"_br"+str(i))

    def forwardPropagation(self, x):
        if self.blkic != self.blkoc:
            residual = self.convert_layer.forwardPropagation(x)
        else:
            residual = x
        for i, branch_layer in enumerate(self.branch_layers):
            x = branch_layer.forwardPropagation(x)
            if self.showMeanStd:
                print("br"+str(i), tf.math.reduce_mean(x).numpy(),
                      tf.math.reduce_std(x).numpy())
        y = x+residual
        y = self.activation(y)
        if self.showMeanStd:
            print("br", tf.math.reduce_mean(y).numpy(),
                  tf.math.reduce_std(y).numpy())
        return y

    def updateWeights(self, grads, args):
        if self.blkic != self.blkoc:
            self.convert_layer.updateWeights(grads[0], args)
        for i, branch_layer in enumerate(self.branch_layers):
            branch_layer.updateWeights(grads[1][i], args)


# without branch rescale
class LambdaResBlk2d:
    def __init__(self, nLambdaLayers: int, arch_br: list, activation="leaky_relu",
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.blkic = arch_br[0]["dkvhCon"][0] if (
            "K" in arch_br[0].keys()) else arch_br[0]["io"][0]
        self.blkoc = arch_br[-1]["dkvhCon"][2] if (
            "K" in arch_br[-1].keys()) else arch_br[-1]["io"][1]
        self.convert_layer = Conv2d(self.blkic, self.blkoc, [1, 1], "", [1, 1, 1, 1], "SAME",
                                    optimizer, debug=debug, showMeanStd=showMeanStd)
        self.branch_layers = []
        for i in range(nLambdaLayers):
            if "K" in arch_br[i].keys():
                d, k, v, h, context_size = arch_br[i]["dkvhCon"]
                archK = arch_br[i]["K"]
                archQ = arch_br[i]["Q"]
                archV = arch_br[i]["V"]
                branch_layer = LambdaConv2d(d, v, k, h, context_size,
                                            archK["nlayers"], archK["kernel_size"], archK["activations"],
                                            archK["strides"], archK["paddings"], archK["out_nchannels"],
                                            archV["nlayers"], archV["kernel_size"], archV["activations"],
                                            archV["strides"], archV["paddings"], archV["out_nchannels"],
                                            archQ["nlayers"], archQ["kernel_size"], archQ["activations"],
                                            archQ["strides"], archQ["paddings"], archQ["out_nchannels"],
                                            optimizer, debug=debug, showMeanStd=showMeanStd)
            else:
                ic, oc = arch_br[i]["io"]
                activation = arch_br[i]["activation"] if (
                    i < nLambdaLayers-1) else ""
                branch_layer = Conv2d(ic, oc, arch_br[i]["kernel_size"],
                                      activation, arch_br[i]["strides"], arch_br[i]["padding"],
                                      optimizer, debug=debug, showMeanStd=showMeanStd)
            self.branch_layers.append(branch_layer)
        self.weights = []
        self.activation = activations.summary[activation]
        self.optimizer = optimizers.summary[optimizer]
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        branch_weights = []
        if path:
            self.convert_layer.config(path+"_id")
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.config(path+"_br"+str(i))
                branch_weights.append(branch_layer.weights)
        else:
            self.convert_layer.config()
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.config()
                branch_weights.append(branch_layer.weights)
        self.weights = [self.convert_layer.weights, branch_weights]

    def storeWeights(self, path: str = ""):
        if path:
            self.convert_layer.storeWeights(path+"_id")
            for i, branch_layer in enumerate(self.branch_layers):
                branch_layer.storeWeights(path+"_br"+str(i))

    def forwardPropagation(self, x):
        if self.blkic != self.blkoc:
            residual = self.convert_layer.forwardPropagation(x)
        else:
            residual = x
        for i, branch_layer in enumerate(self.branch_layers):
            x = branch_layer.forwardPropagation(x)
            if self.showMeanStd:
                print("br"+str(i), tf.math.reduce_mean(x).numpy(),
                      tf.math.reduce_std(x).numpy())
        y = x+residual
        y = self.activation(y)
        if self.showMeanStd:
            print("br", tf.math.reduce_mean(y).numpy(),
                  tf.math.reduce_std(y).numpy())
        return y

    def updateWeights(self, grads, args):
        if self.blkic != self.blkoc:
            self.convert_layer.updateWeights(grads[0], args)
        for i, branch_layer in enumerate(self.branch_layers):
            branch_layer.updateWeights(grads[1][i], args)


class NLambdaResnetBlock12d:
    def __init__(self, nblocks1d: int, arch1d: list, norm2d_mean: float, norm2d_std: float,
                 nblocks2d: int, arch2d: list, output_nchannels: int,
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.blocks1d = []
        for i in range(nblocks1d):
            block = LambdaResnetBlock1d(len(arch1d[i]), arch1d[i], arch1d[-1]["activations"][i],
                                        optimizer, debug=debug, showMeanStd=showMeanStd)
            self.blocks1d.append(block)
        self.blocks2d = []
        self.x2dNorm = GlobalNorm(norm2d_mean, norm2d_std)
        for i in range(nblocks2d):
            block = LambdaResnetBlock2d(len(arch2d[i]), arch2d[i], arch2d[-1]["activations"][i],
                                        optimizer, debug=debug, showMeanStd=showMeanStd)
            self.blocks2d.append(block)
        blksoc = arch2d[nblocks2d-1][-1]["dkvhCon"][2] if (
            "K" in arch2d[nblocks2d-1][-1].keys()) else arch2d[nblocks2d-1][-1]["io"][1]
        self.outlayer = Conv2d(blksoc, output_nchannels, [1, 1], "", [1, 1, 1, 1], "SAME",
                               optimizer, debug=debug, showMeanStd=showMeanStd)
        self.weights = []
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        w1d, w2d = [], []
        if path:
            if path[-1] != "/":
                path += "/"
            for i, block1d in enumerate(self.blocks1d):
                block1d.config(path+"_blk1d"+str(i))
                w1d.append(block1d.weights)
            self.x2dNorm.config(path+"_gn2d")
            for i, block2d in enumerate(self.blocks2d):
                block2d.config(path+"_blk2d"+str(i))
                w2d.append(block2d.weights)
            self.outlayer.config(path+"_output")
        else:
            for i, block1d in enumerate(self.blocks1d):
                block1d.config()
                w1d.append(block1d.weights)
            self.x2dNorm.config()
            for i, block2d in enumerate(self.blocks2d):
                block2d.config()
                w2d.append(block2d.weights)
            self.outlayer.config()
        self.weights = [w1d, w2d, self.outlayer.weights]

    def storeWeights(self, path: str = ""):
        if path:
            for i, block1d in enumerate(self.blocks1d):
                block1d.storeWeights(path+"_blk1d"+str(i))
            self.x2dNorm.storeWeights(path+"_gn2d")
            for i, block2d in enumerate(self.blocks2d):
                block2d.storeWeights(path+"_blk2d"+str(i))
            self.outlayer.storeWeights(path+"_output")

    def forwardPropagation(self, x1d, x2d):
        x1d = tf.one_hot(x1d, 20, axis=2)
        for i, block1d in enumerate(self.blocks1d):
            x1d = block1d.forwardPropagation(x1d)
            if self.showMeanStd:
                print("blk1d"+str(i), tf.math.reduce_mean(x1d).numpy(),
                      tf.math.reduce_std(x1d).numpy())
        L = x1d.shape[1]
        x1dmat = tf.tile(x1d, [L, 1, 1])[tf.newaxis, ...]
        x2d = self.x2dNorm.forwardPropagation(x2d)
        x = tf.concat([x2d, x1dmat, tf.transpose(
            x1dmat, [0, 2, 1, 3])], axis=3)
        for i, block2d in enumerate(self.blocks2d):
            x = block2d.forwardPropagation(x)
            if self.showMeanStd:
                print("blk2d"+str(i), tf.math.reduce_mean(x).numpy(),
                      tf.math.reduce_std(x).numpy())
        x = self.outlayer.forwardPropagation(x)
        y = (x+tf.transpose(x))/2
        if self.showMeanStd:
            print("output", tf.math.reduce_mean(y).numpy(),
                  tf.math.reduce_std(y).numpy())
        return y

    def updateWeights(self, grads, args: dict):
        for i, block1d in enumerate(self.blocks1d):
            block1d.updateWeights(grads[0][i], args)
        for i, block2d in enumerate(self.blocks2d):
            block2d.updateWeights(grads[1][i], args)
        self.outlayer.updateWeights(grads[2], args)


# without branch rescale
class NLambdaResBlk12d:
    def __init__(self,  input_nchannels: list, nblocks1d: int, arch1d: list, norm2d_mean: float, norm2d_std: float,
                 nblocks2d: int, arch2d: list, output_nchannels: int,
                 optimizer: str = "adam_adaclip", debug=False, showMeanStd=False):
        self.blocks1d = []
        for i in range(nblocks1d):
            block = LambdaResBlk1d(len(arch1d[i]), arch1d[i], arch1d[-1]["activations"][i],
                                   optimizer, debug=debug, showMeanStd=showMeanStd)
            self.blocks1d.append(block)

        self.x2dNorm = GlobalNorm(norm2d_mean, norm2d_std)
        blksic = arch2d[0][0]["dkvhCon"][0] if (
            "K" in arch2d[0][0].keys()) else arch2d[0][0]["io"][0]
        self.inlayers2d = []
        layer = Conv2d(input_nchannels, blksic, [3, 3], "leaky_relu", [1, 1, 1, 1], "SAME",
                       optimizer, debug=debug, showMeanStd=showMeanStd)
        self.inlayers2d.append(layer)
        self.blocks2d = []
        for i in range(nblocks2d):
            block = LambdaResBlk2d(len(arch2d[i]), arch2d[i], arch2d[-1]["activations"][i],
                                   optimizer, debug=debug, showMeanStd=showMeanStd)
            self.blocks2d.append(block)
        blksoc = arch2d[nblocks2d-1][-1]["dkvhCon"][2] if (
            "K" in arch2d[nblocks2d-1][-1].keys()) else arch2d[nblocks2d-1][-1]["io"][1]
        self.outlayers2d = []
        layer = Conv2d(blksoc, output_nchannels, [1, 1], "leaky_relu", [1, 1, 1, 1], "SAME",
                       optimizer, debug=debug, showMeanStd=showMeanStd)
        self.outlayers2d.append(layer)
        self.weights = []
        self.debug = debug
        self.showMeanStd = showMeanStd

    def config(self, path: str = ""):
        win, w1d, w2d, wout = [], [], [], []
        if path:
            if path[-1] != "/":
                path += "/"
            for i, block1d in enumerate(self.blocks1d):
                block1d.config(path+"_blk1d"+str(i))
                w1d.append(block1d.weights)
            self.x2dNorm.config(path+"_gn2d")
            for i, layer in enumerate(self.inlayers2d):
                layer.config(path+"_in"+str(i))
                win.append(layer.weights)
            for i, block2d in enumerate(self.blocks2d):
                block2d.config(path+"_blk2d"+str(i))
                w2d.append(block2d.weights)
            for i, layer in enumerate(self.outlayers2d):
                layer.config(path+"_out"+str(i))
                wout.append(layer.weights)
        else:
            for i, block1d in enumerate(self.blocks1d):
                block1d.config()
                w1d.append(block1d.weights)
            self.x2dNorm.config()
            for i, layer in enumerate(self.inlayers2d):
                layer.config()
                win.append(layer.weights)
            for i, block2d in enumerate(self.blocks2d):
                block2d.config()
                w2d.append(block2d.weights)
            for i, layer in enumerate(self.outlayers2d):
                layer.config()
                wout.append(layer.weights)
        self.weights = [win, w1d, w2d, wout]

    def storeWeights(self, path: str = ""):
        if path:
            for i, block1d in enumerate(self.blocks1d):
                block1d.storeWeights(path+"_blk1d"+str(i))
            self.x2dNorm.storeWeights(path+"_gn2d")
            for i, layer in enumerate(self.inlayers2d):
                layer.storeWeights(path+"_in"+str(i))
            for i, block2d in enumerate(self.blocks2d):
                block2d.storeWeights(path+"_blk2d"+str(i))
            for i, layer in enumerate(self.outlayers2d):
                layer.storeWeights(path+"_out"+str(i))

    def forwardPropagation(self, x1d, x2d):
        x1d = tf.one_hot(x1d, 20, axis=2)
        for i, block1d in enumerate(self.blocks1d):
            x1d = block1d.forwardPropagation(x1d)
            if self.showMeanStd:
                print("blk1d"+str(i), tf.math.reduce_mean(x1d).numpy(),
                      tf.math.reduce_std(x1d).numpy())
        L = x1d.shape[1]
        x1dmat = tf.tile(x1d, [L, 1, 1])[tf.newaxis, ...]
        x2d = self.x2dNorm.forwardPropagation(x2d)
        x = tf.concat([x2d, x1dmat, tf.transpose(
            x1dmat, [0, 2, 1, 3])], axis=3)
        for i, layer in enumerate(self.inlayers2d):
            x = layer.forwardPropagation(x)
        for i, block2d in enumerate(self.blocks2d):
            x = block2d.forwardPropagation(x)
            if self.showMeanStd:
                print("blk2d"+str(i), tf.math.reduce_mean(x).numpy(),
                      tf.math.reduce_std(x).numpy())
        for i, layer in enumerate(self.outlayers2d):
            x = layer.forwardPropagation(x)
        y = (x+tf.transpose(x))/2
        if self.showMeanStd:
            print("output", tf.math.reduce_mean(y).numpy(),
                  tf.math.reduce_std(y).numpy())
        return y

    def updateWeights(self, grads, args: dict):
        for i, layer in enumerate(self.inlayers2d):
            layer.updateWeights(grads[0][i], args)
        for i, block1d in enumerate(self.blocks1d):
            block1d.updateWeights(grads[1][i], args)
        for i, block2d in enumerate(self.blocks2d):
            block2d.updateWeights(grads[2][i], args)
        for i, layer in enumerate(self.outlayers2d):
            layer.updateWeights(grads[3][i], args)
