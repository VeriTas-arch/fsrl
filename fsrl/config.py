from dataclasses import dataclass

import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADDINPUT = 4
NUMRESPONSESTEP = 1


@dataclass
class TrainConfig:
    rngseed: int = -1
    rew: float = 1.0
    wp: float = 0.0
    bent: float = 0.1
    blossv: float = 0.1
    gr: float = 0.9
    hs: int = 200
    bs: int = 32
    gc: float = 2.0
    eps: float = 1e-6
    nbiter: int = 5000
    save_every: int = 200
    pe: int = 101
    cs: int = 15
    triallen: int = 4
    nbtraintrials: int = 32
    nbtesttrials: int = 28
    testlmult: float = 3.0
    l2: float = 0.0
    lr: float = 1e-4
    lpw: float = 1e-4
    nbcues_min: int = 6
    nbcues_max: int = 10

    @property
    def nbcuesrange(self):
        return range(self.nbcues_min, self.nbcues_max + 1)

    @property
    def nbtrials(self):
        return self.nbtraintrials + self.nbtesttrials

    @property
    def eplen(self):
        return self.nbtrials * self.triallen

    @property
    def nbstimbits(self):
        return 2 * self.cs + 1

    @property
    def outputsize(self):
        return 2

    @property
    def inputsize(self):
        return self.nbstimbits + ADDINPUT + self.outputsize

    def to_model_dict(self):
        """Return the dict shape expected by the original model code."""
        return {
            "rngseed": self.rngseed,
            "rew": self.rew,
            "wp": self.wp,
            "bent": self.bent,
            "blossv": self.blossv,
            "gr": self.gr,
            "hs": self.hs,
            "bs": self.bs,
            "gc": self.gc,
            "eps": self.eps,
            "nbiter": self.nbiter,
            "save_every": self.save_every,
            "pe": self.pe,
            "nbcuesrange": self.nbcuesrange,
            "cs": self.cs,
            "triallen": self.triallen,
            "nbtraintrials": self.nbtraintrials,
            "nbtesttrials": self.nbtesttrials,
            "nbtrials": self.nbtrials,
            "eplen": self.eplen,
            "testlmult": self.testlmult,
            "l2": self.l2,
            "lr": self.lr,
            "lpw": self.lpw,
            "outputsize": self.outputsize,
            "inputsize": self.inputsize,
        }
