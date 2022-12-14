import torch
from torch import nn
from torch import cuda

device = torch.device("cuda:0" if cuda.is_available() else "cpu")

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
    "": lambda x: x,
}


class resBlock1d(nn.Module):  # simple resnet encoder, no down sampling
    def __init__(self, archBlock1d, do_down_sample=False, down_sample_ks=2):
        super(resBlock1d, self).__init__()
        layers = []
        inc, outc = archBlock1d["branch"][0]["ioc"][0], archBlock1d["branch"][-1]["ioc"][-1]
        layers.append(nn.Conv1d(inc, outc, 1, 1, "same"))
        if do_down_sample:
            layers.append(nn.AvgPool1d(down_sample_ks))
        self.shortcut = nn.Sequential(*layers)

        layers = []
        for i, archLayer1d in enumerate(archBlock1d["branch"]):
            layer = nn.Conv1d(archLayer1d["ioc"][0], archLayer1d["ioc"][1],
                              archLayer1d["ks"], archLayer1d["st"], archLayer1d["pd"])
            layers.append(layer)
            if i < len(archBlock1d)-1:
                layer_act = ACTIVATIONS[archLayer1d["act"]]
                layers.append(layer_act)
        if do_down_sample:
            layers.append(nn.MaxPool1d(down_sample_ks))
        self.branch = nn.Sequential(*layers)
        self.bw = nn.Parameter(torch.FloatTensor(
            [archBlock1d["others"]["bw_init"]]), requires_grad=True)
        self.act = ACTIVATIONS[archBlock1d["others"]["act"]]

    def forward(self, x: torch.Tensor):
        x_sc = self.shortcut(x)
        x_br = self.branch(x)
        y = (x_sc+self.bw*x_br)/torch.sqrt(1+self.bw**2)
        y = self.act(y)
        return y


class resBlock2d(nn.Module):
    def __init__(self, archBlock2d, do_down_sample=False, down_sample_ks=(2, 2)):
        super(resBlock2d, self).__init__()
        layers = []
        inc, outc = archBlock2d["branch"][0]["ioc"][0], archBlock2d["branch"][-1]["ioc"][-1]
        layers.append(nn.Conv2d(inc, outc, (1, 1), 1, "same"))
        if do_down_sample:
            layers.append(nn.AvgPool2d(down_sample_ks))
        self.shortcut = nn.Sequential(*layers)
        layers = []
        for i, archLayer2d in enumerate(archBlock2d["branch"]):
            layer = nn.Conv2d(archLayer2d["ioc"][0], archLayer2d["ioc"][1],
                              archLayer2d["ks"], archLayer2d["st"], archLayer2d["pd"])
            layers.append(layer)
            if i < len(archBlock2d)-1:
                layer_act = ACTIVATIONS[archLayer2d["act"]]
                layers.append(layer_act)
        if do_down_sample:
            layers.append(nn.MaxPool2d(down_sample_ks))
        self.branch = nn.Sequential(*layers)
        self.bw = nn.Parameter(torch.FloatTensor(
            [archBlock2d["others"]["bw_init"]]), requires_grad=True)
        self.act = ACTIVATIONS[archBlock2d["others"]["act"]]

    def forward(self, x: torch.Tensor):
        x_sc = self.shortcut(x)
        x_br = self.branch(x)
        y = (x_sc+self.bw*x_br)/torch.sqrt(1+self.bw**2)
        y = self.act(y)
        return y


class SeqEncoder(nn.Module):
    def __init__(self, archSeqEnc):
        super(SeqEncoder, self).__init__()
        blocks = []
        for archBlock in archSeqEnc:
            blocks.append(resBlock1d(archBlock))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        y = self.blocks(x)
        return y


class SpatialConcentrator(nn.Module):
    def __init__(self, archSC):
        super(SpatialConcentrator, self).__init__()
        din = int((archSC["branch"][0]["branch"][0]["ioc"][0]-1)/2)
        dmed = archSC["branch"][-1]["branch"][-1]["ioc"][-1]
        dout = archSC["others"]["dout"]
        self.shortcut = nn.Linear(din, dout)
        blocks = []
        for archBlock in archSC["branch"]:
            if "ds_ks" in archBlock["others"].keys():
                block = resBlock2d(archBlock, do_down_sample=True,
                                   down_sample_ks=archBlock["others"]["ds_ks"])
            else:
                block = resBlock2d(archBlock)
            blocks.append(block)
        blocks.append(nn.AdaptiveMaxPool2d((1, 1)))
        blocks.append(nn.Flatten())
        blocks.append(nn.Linear(dmed, dout))
        self.dsbranch = nn.Sequential(*blocks)
        self.bw = nn.Parameter(torch.FloatTensor(
            [archSC["others"]["bw_init"]]), requires_grad=True)
        self.act = ACTIVATIONS[archSC["others"]["act"]]

    def forward(self, xSeq, x):
        x_sc = self.shortcut(xSeq)
        x_br = self.dsbranch(x)
        y = (x_sc+self.bw*x_br)/torch.sqrt(1+self.bw**2)
        y = self.act(y)
        return y


class GraphEncoder(nn.Module):
    def __init__(self, archSC):
        super(GraphEncoder, self).__init__()
        self.k = archSC["others"]["nearest_num_k"]
        self.d = int((archSC["branch"][0]["branch"][0]["ioc"][0]-1)/2)
        self.spaConc = SpatialConcentrator(archSC)

    def forward(self, x):
        xSeq, kListSeq, envmatSeq = x
        (L, d), k = xSeq.shape, self.k
        x1d = torch.zeros([L, k, d])
        for i in range(L):
            x1d[i] = xSeq[kListSeq[i, :k], :]
        x1d = x1d.transpose(1, 2)
        x1dmat = torch.tile(x1d.unsqueeze(3), dims=(1, 1, 1, k)).to(device)
        x = torch.concat([envmatSeq[:, :k, :k].unsqueeze(1), x1dmat,
                         x1dmat.transpose(2, 3)], dim=1)
        y = self.spaConc(xSeq, x)
        return y, kListSeq, envmatSeq


class ReadOut(nn.Module):
    def __init__(self, archRO):
        super(ReadOut, self).__init__()
        blocks = []
        for archBlock2d in archRO:
            block = resBlock2d(archBlock2d)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        y = self.blocks(x)
        return y


class EnvMPNN(nn.Module):
    def __init__(self, archSeqEnc, archGraphEnc, archRO):
        super(EnvMPNN, self).__init__()
        seq_oc = archSeqEnc[-1]["branch"][-1]["ioc"][-1] if (
            len(archSeqEnc) > 0) else 0
        graph_ic = int((archGraphEnc[0]["branch"][0]["branch"][0]
                       ["ioc"][0]-1)/2) if (len(archGraphEnc) > 0) else 0
        graph_oc = archGraphEnc[-1]["others"]["dout"] if (
            len(archGraphEnc) > 0) else 0
        ro_ic = archRO[0]["branch"][0]["ioc"][0] if (len(archRO) > 0) else 0
        ro_oc = archRO[-1]["branch"][-1]["ioc"][-1] if (len(archRO) > 0) else 0
        if graph_ic == 0:
            graph_ic = ro_ic
            graph_oc = ro_ic

        self.seqEnc = SeqEncoder(archSeqEnc)
        self.seq2graph = nn.Conv1d(seq_oc, graph_ic, 1, 1, padding="same")
        graphEncs = []
        for archSC in archGraphEnc:
            graphEnc = GraphEncoder(archSC)
            graphEncs.append(graphEnc)
        self.graphEnc = nn.Sequential(*graphEncs)
        self.graph2ro = nn.Conv2d(
            graph_oc*2+1, ro_ic, (1, 1), 1, padding="same")
        self.dismatRO = ReadOut(archRO)
        self.finalout = nn.Conv2d(ro_oc, 1, [3, 3], 1, "same")

    def forward(self, xSeq, kListSeq, envmatSeq, dismat):
        xSeq, kListSeq, envmatSeq, dismat = xSeq.to(device), kListSeq.to(device), envmatSeq.to(device), dismat.to(device)
        seq1d = self.seqEnc(xSeq)  # [1, d, L]
        graph1d = self.seq2graph(seq1d)  # [1, din, L]
        graph1d = graph1d[0, :, :].t()  # [L, din]
        # kListSeq [L, kmax]; envmatSeq [L, kmax, kmax]
        graph1d = self.graphEnc([graph1d, kListSeq, envmatSeq])[0]  # [L, dout]
        L = graph1d.shape[0]
        graph1d = graph1d.t().unsqueeze(2).unsqueeze(0)  # [1, dout, L, 1]
        x1dmat = torch.tile(graph1d, (1, 1, 1, L))  # [1, dout, L, L]
        x2d = torch.concat([dismat, x1dmat,
                            x1dmat.transpose(2, 3)], dim=1)  # [1, 1+2*dout, L, L]
        x2d = self.graph2ro(x2d)  # [1, inc, L, L]
        y = self.dismatRO(x2d)  # [1, outc, L, L]
        y = self.finalout(y)  # [1, 1, L, L]
        return y
