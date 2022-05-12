import numpy as np
import tensorflow as tf
import layers
import os
import shutil
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


class Model(layers.NLambdaResBlk12d):
    def __init__(self, input_nchannels: int, nblocks1d: int, arch1d: list, nblocks2d: int, arch2d: list, output_nchannels: int,
                 optimizer: str = "adam_adaclip", debug: bool = False, showMeanStd: bool = False,
                 nepochs=500, patience=60, lr: list = [1e-3, 1e-2, 20], batch_size: list = [1, 512],
                 moment: list = [0.9, 0.999, 0.1], clip_grad=1e16, clip_lambda=0.01, verbose=1):
        super(Model, self).__init__(input_nchannels, nblocks1d, arch1d, nblocks2d, arch2d, output_nchannels,
                                    optimizer, debug=debug, showMeanStd=showMeanStd)
        self.epochs = nepochs
        # early stopping patience, negative means no early stopping
        self.patience = patience
        # learning rate decrease with epoch, with a periodic fluctuation
        self.lr_min = lr[0]
        self.lr_max = lr[1]
        self.lr_cycle = lr[2]
        # batch size increase with epoch
        self.batch_size_min = batch_size[0]
        self.batch_size_max = batch_size[1]
        # forget factor and smooth factor for moments
        self.args = {
            "beta1": moment[0],
            "beta2": moment[1],
            "smoothFactor": moment[2],
            "clip_grad": clip_grad,
            "clip_lambda": clip_lambda
        }
        # 0, silent; 1, report every epoch; 2 report every batch
        self.verbose = verbose

    def __dy_batch_size(self, iepoch):
        ran = np.math.log(self.batch_size_max/self.batch_size_min, 2)
        ret = self.batch_size_min*(1 << int(ran*iepoch/self.epochs))
        return ret

    def __dy_lr_cycle(self, iepoch, ibatch):
        ran = np.math.log(self.lr_max/self.lr_min)
        lr_max_bound = self.lr_max * \
            np.math.pow(np.math.e, -ran*iepoch/self.epochs)
        iphase = 2*(ibatch % self.lr_cycle)/self.lr_cycle
        if iphase > 1:
            iphase = 2-iphase
        ran = lr_max_bound-self.lr_min
        ret = ran*np.math.pow(iphase, 3)+self.lr_min
        return ret

    # train
    def fit(self, train_data, validation_data, saveParamTo: str = "", saveHistoryTo: str = "",
            saveFilmsTo: str = "", iProFilm: int = 0, sProFilm: str = "pro", startEpoch=0):
        # ---dataset
        x1d_train, x2d_train, y_train = train_data
        x1d_valid, x2d_valid, y_valid = validation_data
        # ---variables
        train_lossEpochL, valid_lossEpochL = [], []  # train, loss history
        train_lossEpoch, valid_lossEpoch = 0., 0.  # train, loss in epoch
        train_lossBatch = 0.  # train, loss in batch
        train_lossPoint, valid_lossPoint = 0., 0.  # train, loss for datapoint
        param_savePathL = {}  # train, manage save params
        param_savePath = ""  # train
        valid_loss_min, valid_loss_min_index = -1., 0  # train, early stopping
        npatience = 0  # train, early stopping
        ibatch, istep = 0, 0  # train, batch/datapoint index
        # ---train
        for iepoch in range(startEpoch, self.epochs):
            train_lossEpoch = 0.
            batch_size = self.__dy_batch_size(iepoch)
            istep = 0
            while(istep < len(y_train)):
                lr = self.__dy_lr_cycle(iepoch, ibatch)
                with tf.GradientTape() as tape:
                    train_lossBatch = 0.
                    while(istep < len(y_train)):
                        x1d, x2d, y = x1d_train[istep], x2d_train[istep], y_train[istep]
                        yp = self.forwardPropagation(x1d, x2d)
                        train_lossPoint = tf.reduce_mean(tf.square(yp-y))
                        istep += 1
                        train_lossBatch += train_lossPoint
                        if istep % batch_size == 0:
                            break
                    train_lossBatch /= ((istep-1) % batch_size+1)
                ibatch += 1
                grads = tape.gradient(train_lossBatch, self.weights)
                if self.debug:
                    print(self.weights[0][0])
                    print(grads[0][0])
                args: dict = self.args
                args["lr"] = lr
                self.updateWeights(grads, self.args)
                if self.verbose > 1:
                    print("batch %d, train loss %f. current batchsize is %d, learning rate is %f" % (
                        ibatch+1, train_lossBatch, batch_size, lr))
                train_lossEpoch += train_lossBatch*((istep-1) % batch_size+1)
            train_lossEpoch /= len(y_train)
            train_lossEpochL.append(train_lossEpoch)
            # store weights
            if saveParamTo:
                param_savePath = saveParamTo+str(iepoch)
                if os.path.exists(param_savePath):
                    print("warning: potential path conflicts.", file=sys.stderr)
                    shutil.rmtree(param_savePath)
                os.mkdir(param_savePath)
                self.storeWeights(param_savePath+'/')
                param_savePathL[iepoch] = param_savePath

            # validation
            valid_lossEpoch = 0.
            for istep, (x1d, x2d, y) in enumerate(zip(x1d_valid, x2d_valid, y_valid)):
                yp = self.forwardPropagation(x1d, x2d)
                valid_lossPoint = tf.reduce_mean(tf.square(yp-y))
                valid_lossEpoch += valid_lossPoint
            valid_lossEpoch /= len(y_valid)
            valid_lossEpochL.append(valid_lossEpoch)
            # summary this epoch
            if self.verbose > 0:
                print("epoch %d: train loss %f, valid loss %f. current batchsize is %d" %
                      (iepoch+1, train_lossEpoch, valid_lossEpoch, batch_size))
            # save films of i-th valid datapoint
            if saveFilmsTo:
                if iProFilm >= len(y_valid):
                    print(
                        "error: protein index to generate film exceeds size of valid set", file=sys.stderr)
                    quit()
                if not os.path.exists(saveFilmsTo):
                    os.mkdir(saveFilmsTo)
                self.visualization(
                    (x1d_valid[iProFilm], x2d_valid[iProFilm], y_valid[iProFilm]), sProFilm, saveto=saveFilmsTo)
            # earlystopping
            if self.patience > 0:
                if valid_loss_min >= valid_lossEpoch or valid_loss_min < 0:
                    valid_loss_min = valid_lossEpoch
                    valid_loss_min_index = iepoch
                    npatience = 0
                else:
                    npatience += 1
                    if npatience > self.patience:
                        break
                if self.debug:
                    print("npatience %d" % npatience)

        # after training, only save the best parameters
        if saveParamTo:
            if os.path.exists(saveParamTo):
                print("warning: potential path conflicts.", file=sys.stderr)
                shutil.rmtree(saveParamTo)
            for icp, cp in param_savePathL.items():
                if icp == valid_loss_min_index:
                    os.rename(cp, saveParamTo)
                else:
                    shutil.rmtree(cp)

        # save training history
        history = np.array([train_lossEpochL, valid_lossEpochL])
        if saveHistoryTo:
            timestr = time.strftime("%Y-%m-%d_%H-%M-%S",
                                    time.localtime(time.time()))
            if not(os.path.exists(saveHistoryTo)):
                os.mkdir(saveHistoryTo)
            np.save(saveHistoryTo+"/"+timestr+".npy", history)
            fig = plt.figure(figsize=(8, 6))
            plt.title("train-valid history", fontsize=24)
            plt.plot(history[0], label="train")
            plt.plot(history[1], label="valid")
            plt.ylabel("loss(mse)", fontsize=24)
            plt.xlabel("epoch", fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=24)
            if saveHistoryTo:
                plt.savefig(saveHistoryTo+"/"+timestr+".png")
                plt.close(fig)
            else:
                plt.show()
        return history

    # evaluate
    def evaluate(self, data, showPearson=True, showDistribution=True,
                 saveHistogramTo: str = "", label: str = "test"):
        x1d_eval, x2d_eval, y_eval = data
        loss = 0.
        yp1d, y1d = [], []
        for istep, (x1d, x2d, y) in enumerate(zip(x1d_eval, x2d_eval, y_eval)):
            yp = self.forwardPropagation(x1d, x2d)
            loss += tf.reduce_mean(tf.square(yp-y))
            if showPearson:
                yp1d.append(np.reshape(yp, -1))
                y1d.append(np.reshape(y, -1))
        loss /= len(y_eval)
        print(label+" evaluated. loss(mse) is %f" % loss)
        if showPearson:
            yp1d = np.concatenate(yp1d, axis=0)
            y1d = np.concatenate(y1d, axis=0)
            print("on"+label+", prediction std dev %.4f, true std dev %.4f, pearson corrcoef is %.4f" %
                  (np.std(yp1d), np.std(y1d), np.corrcoef(yp1d, y1d)[1, 0]))

        if showDistribution:
            if not showPearson:
                yp1d = np.concatenate(yp1d, axis=0)
                y1d = np.concatenate(y1d, axis=0)
            fig = plt.figure(figsize=(8, 12))
            plt.title(label+"true/pred Econ histogram", fontsize=24)
            plt.xlabel("density", fontsize=24)
            plt.ylabel("contact energy (kcal/mol)", fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.hist([y1d, yp1d], bins=20, label=[
                "true", "pred"], log=True, density=True, orientation="horizontal")
            plt.legend(fontsize=24)
            if saveHistogramTo:
                timestr = time.strftime(
                    "%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
                plt.savefig(saveHistogramTo+'/'+label+timestr+'.jpg')
                plt.close(fig)
            else:
                plt.show()

    # visualization
    def visualization(self, datapoint, slabel: str = "pro", clip_min: float = -10., clip_max: float = 0., saveto=""):
        x1d, x2d, y = datapoint
        yp = self.forwardPropagation(x1d, x2d)
        imy, imyp = y[0, :, :, 0], yp[0, :, :, 0]
        # analysis
        pcorr = np.corrcoef(np.reshape(imy, -1), np.reshape(imyp, -1))[0, 1]
        pcorr_clipped = np.corrcoef(
            np.clip(np.reshape(imy, -1), clip_min, clip_max),
            np.clip(np.reshape(imyp, -1), clip_min, clip_max)
        )[0, 1]
        # show econmaps
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        fig.suptitle(slabel, fontsize=20)
        norm = mpl.colors.Normalize(vmin=clip_min, vmax=clip_max, clip=True)
        axes[0].set_title('true', fontsize=16)
        imy = axes[0].imshow(imy, cmap='viridis', norm=norm)
        axes[1].set_title('pred', fontsize=16)
        imyp = axes[1].imshow(imyp, cmap='viridis', norm=norm)
        axes[2].set_title('econ-histogram', fontsize=16)
        histyyp = axes[2].hist([np.reshape(y, -1), np.reshape(yp, -1)], bins=20,
                               label=["true", "pred"], log=True, density=True, orientation="horizontal")
        axes[2].legend()
        fig.colorbar(imyp, ax=[axes[0], axes[1]], orientation='horizontal')
        fig.text(0.05, 0.05, "Pearson corrcoef  is {:.4f}, if clipped to [{:.2f}, {:.2f}], Pearson corrcoef is {:4f}".format(
            pcorr, clip_min, clip_max, pcorr_clipped))
        if saveto:
            timestr = time.strftime(
                "%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
            plt.savefig(saveto+"/"+slabel+"_"+timestr+'.jpg')
            plt.close(fig)
        else:
            plt.show()
