import torch
from torch import cuda
from torch import nn
import layers
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

device = torch.device("cuda:0" if cuda.is_available() else "cpu")


def weightInit(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        nn.init.constant_(m.bias, 0)


class Model(layers.EnvMPNN):
    def __init__(self, archSeqEnc, archGraphEnc, archRO):
        super(Model, self).__init__(archSeqEnc, archGraphEnc, archRO)
        self.kmax = max([archSC["others"]["nearest_num_k"]
                        for archSC in archGraphEnc]) if (len(archGraphEnc) > 0) else 0

    def config(self, path=""):
        if not path:
            self.apply(weightInit)
        else:
            self.load_state_dict(torch.load(path+".pt", map_location=device))

    def fit(self, train_xSeqN, train_kListSeqN, train_envmatSeqN, train_dismatN, train_econmatN,
            valid_xSeqN, valid_kListSeqN, valid_envmatSeqN, valid_dismatN, valid_econmatN,
            batch_size=1, nepoch=500, patience=60, lr_max=1e-3, lr_min=1e-4, grad_norm_cutoff=1e10, loss_fn=nn.MSELoss(), optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
            paramsSavePath="cp", historySaveDir="history",
            filmsSaveDir="film", iFilm=0, sFilm="protein", verbose=2):
        optim = optimizer(self.parameters(), lr_max, [0.9, 0.999])
        schedu = scheduler(optim, nepoch, lr_min, verbose=True)
        train_N, valid_N = len(train_xSeqN), len(valid_xSeqN)

        filmSample = (valid_xSeqN[iFilm], valid_kListSeqN[iFilm],
                      valid_envmatSeqN[iFilm], valid_dismatN[iFilm], valid_econmatN[iFilm])

        train_loss_epoch_l, valid_loss_epoch_l = [], []
        valid_loss_min, iepoch_min, npatience = 10000., 0, patience
        for iepoch in range(nepoch):
            # train
            train_loss_total = 0.
            for i, (xSeq, kListSeq, envmatSeq, dismat, econmat) in enumerate(zip(train_xSeqN, train_kListSeqN, train_envmatSeqN, train_dismatN, train_econmatN)):
                econmat = econmat.to(device)
                optim.zero_grad(set_to_none=True)
                econPred = self.forward(xSeq, kListSeq, envmatSeq, dismat)
                train_loss_step = loss_fn(econPred, econmat)
                train_loss_step.backward()
                if (i % batch_size == batch_size-1):
                    nn.utils.clip_grad.clip_grad_norm_(
                        self.parameters(), grad_norm_cutoff)
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                with torch.no_grad():
                    train_loss_total += train_loss_step
            train_loss_epoch = train_loss_total/train_N
            train_loss_epoch_l.append(train_loss_epoch)
            # valid
            with torch.no_grad():
                valid_loss_total = 0.
                for i, (xSeq, kListSeq, envmatSeq, dismat, econmat) in enumerate(zip(valid_xSeqN, valid_kListSeqN, valid_envmatSeqN, valid_dismatN, valid_econmatN)):
                    econmat = econmat.to(device)
                    econPred = self.forward(xSeq, kListSeq, envmatSeq, dismat)
                    valid_loss_step: torch.Tensor = loss_fn(econPred, econmat)
                    valid_loss_total += valid_loss_step
                valid_loss_epoch = valid_loss_total/valid_N
                valid_loss_epoch_l.append(valid_loss_epoch)
            # adjust lr and report
            schedu.step()
            if verbose > 0:
                print("epoch %d: train loss is %.4f, valid loss is %.4f." %
                      (iepoch, train_loss_epoch, valid_loss_epoch))
            # save params
            if paramsSavePath:
                torch.save(self.state_dict(), paramsSavePath+str(iepoch)+".pt")
            # save film sample
            if filmsSaveDir:
                if not os.path.exists(filmsSaveDir):
                    os.makedirs(filmsSaveDir)
                self.visualize(filmSample[0], filmSample[3], filmSample[4],
                               filmSample[1], filmSample[2], label=sFilm, savetoDir=filmsSaveDir)

            # earlystopping
            if valid_loss_epoch < valid_loss_min:
                valid_loss_min = valid_loss_epoch
                iepoch_min = iepoch
            else:
                npatience -= 1
            if npatience <= 0:
                break

        # only save the best params
        if paramsSavePath:
            for iepoch in range(nepoch):
                pt_path = paramsSavePath+str(iepoch)+".pt"
                if os.path.exists(pt_path) and iepoch != iepoch_min:
                    os.remove(pt_path)
            os.rename(paramsSavePath+str(iepoch_min) +
                      ".pt", paramsSavePath+".pt")
        history = torch.Tensor([train_loss_epoch_l, valid_loss_epoch_l])
        if historySaveDir:
            if not os.path.exists(historySaveDir):
                os.makedirs(historySaveDir, mode=0o754)
            timestr = time.strftime(
                "%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
            torch.save(history, historySaveDir+"/"+timestr+".hi")
        return history

    @torch.no_grad()
    def predict(self, xSeq, dismat):
        posSeq = xSeq[-3:, :]
        L = xSeq.shape[1]
        k = min(self.kmax, L-1)
        kListSeq = []
        envmatSeq = []
        for i in range(L):
            disarr = dismat[i]
            disarr_sorted = sorted([(dis, seq) for dis, seq in zip(
                disarr, torch.arange(L))], key=lambda x: x[0])[1:k+1]
            kList = [unit[1] for unit in disarr_sorted]
            kListSeq.append(kList)
            kPos = posSeq[kList]-posSeq[i]
            kPos_reduced = torch.Tensor(
                [pos/torch.sum(torch.square(pos)) for pos in kPos])
            envmat = torch.matmul(kPos_reduced, kPos_reduced.transpose(0, 1))
            envmatSeq.append(envmat)
        kListSeq = torch.tensor(kListSeq, dtype=torch.long)
        envmatSeq = torch.tensor(envmatSeq, dtype=torch.float32)

        econPred = self.forward(xSeq, kListSeq, envmatSeq, dismat)
        return econPred

    @torch.no_grad()
    def evaluate(self, xSeqN, dismatN, econmatN, loss_fn=nn.MSELoss(), kListSeqN=None, envmatSeqN=None, label="test", drawHist: bool = True, savetoDir: str = ""):
        N = len(xSeqN)
        loss_total = 0.
        econPred_collect, econTrue_collect = [], []

        if (kListSeqN is not None) and (envmatSeqN is not None):
            for xSeq, kListSeq, envmatSeq, dismat, econmat in zip(
                    xSeqN, kListSeqN, envmatSeqN, dismatN, econmatN):
                econmatPred = self.forward(
                    xSeq, kListSeq, envmatSeq, dismat).to("cpu")
                loss_step = loss_fn(econmatPred, econmat)
                loss_total += loss_step
                econmatPred = econmatPred.detach().numpy()
                econmat = econmat.detach().numpy()
                econPred_collect.append(np.reshape(econmatPred, -1))
                econTrue_collect.append(np.reshape(econmat, -1))
        else:
            for xSeq, dismat, econmat in zip(xSeqN, dismatN, econmatN):
                econmatPred = self.predict(xSeq, dismat).to("cpu")
                loss_step = loss_fn(econmatPred, econmat)
                loss_total += loss_step
                econmatPred = econmatPred.detach().numpy()
                econmat = econmat.detach().numpy()
                econPred_collect.append(np.reshape(econmatPred, -1))
                econTrue_collect.append(np.reshape(econmat, -1))

        loss = loss_total/N
        econPred_collect = np.concatenate(econPred_collect)
        econTrue_collect = np.concatenate(econTrue_collect)
        corrcoef = np.corrcoef(econPred_collect, econTrue_collect)[0, 1]
        k, b = np.polyfit(econPred_collect, econTrue_collect, 1)
        print("on %s dataset, loss is %.4f, pearson corrcoef is %.4f, \
            linReg (k, b): %.4f, %.4f" % (label, loss, corrcoef, k, b))

        if drawHist:
            y1d, yp1d = econTrue_collect, econPred_collect
            fig = plt.figure(figsize=(8, 12))
            plt.title("true/pred Econ histogram on dataset " +
                      label, fontsize=24)
            plt.xlabel("density", fontsize=24)
            plt.ylabel("contact energy score (REU)", fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.hist([y1d, yp1d], bins=20, label=[
                "true", "pred"], log=True, density=True, orientation="horizontal")
            plt.legend(fontsize=24)
            if savetoDir:
                if not os.path.exists(savetoDir):
                    os.makedirs(savetoDir, mode=0o754)
                timestr = time.strftime(
                    "%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
                plt.savefig(savetoDir+"/"+label+timestr+'.jpg')
                plt.close(fig)
            else:
                plt.show()

    def visualize(self, xSeq, dismat, econmat, kListSeq=None, envmatSeq=None, clip_min=-10., clip_max=0., label="protein", savetoDir=""):
        if (kListSeq is not None) and (envmatSeq is not None):
            econmatPred = self.forward(
                xSeq, kListSeq, envmatSeq, dismat).to("cpu")
        else:
            econmatPred = self.predict(xSeq, dismat).to("cpu")
        imy = econmat[0, 0, :, :].detach().numpy()
        imyp = econmatPred[0, 0, :, :].detach().numpy()
        # analysis
        y1d, yp1d = np.reshape(imy, -1), np.reshape(imyp, -1)
        pcorr = np.corrcoef(y1d, yp1d)[0, 1]
        pcorr_clipped = np.corrcoef(
            np.clip(y1d, clip_min, clip_max),
            np.clip(yp1d, clip_min, clip_max)
        )[0, 1]
        k, b = np.polyfit(y1d, yp1d, 1)
        # show econmaps
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        fig.suptitle(label, fontsize=20)
        norm = mpl.colors.Normalize(vmin=clip_min, vmax=clip_max, clip=True)
        axes[0].set_title('true', fontsize=16)
        imy = axes[0].imshow(imy, cmap='viridis', norm=norm)
        axes[1].set_title('pred', fontsize=16)
        imyp = axes[1].imshow(imyp, cmap='viridis', norm=norm)
        axes[2].set_title('econ-histogram', fontsize=16)
        histyyp = axes[2].hist([y1d, yp1d], bins=20, label=[
                               "true", "pred"], log=True, density=True, orientation="horizontal")
        axes[2].legend()
        fig.colorbar(imyp, ax=[axes[0], axes[1]], orientation='horizontal')
        fig.text(0.05, 0.05, "PCorr {:.4f}, linReg k {:.4f} - b {:.4f}, clipped to [{:.2f}, {:.2f}] PCorr {:4f}".format(
            pcorr, k, b, clip_min, clip_max, pcorr_clipped))
        if savetoDir:
            if not os.path.exists(savetoDir):
                os.makedirs(savetoDir, mode=0o754)
            timestr = time.strftime(
                "%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
            plt.savefig(savetoDir+"/"+label+"_"+timestr+'.jpg')
            plt.close(fig)
        else:
            plt.show()
