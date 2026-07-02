# %% [markdown]
# ## HOW TO USE THIS NOTEBOOK
#
# This is the simpler vesion of the code, without any experimental or evaluation code. It just meta-trains a network (over 30000 iterations) and stores the optimized network in `net.dat`. You can then use this file to run the EVAL mode of the main code (i.e. run `main.ipynb` with EVAL=True) and produce figures.
#
# If you want to understand how the system works, it is highly recommended to look at this code rather than the main code.
#
# This system uses the exact same training process as the main code, except for the fact that plastic weights are reset at every episode and no data from previous episodes is used (no attempt at continual meta-learning, unlike the main code where the network keeps memory of up to 3 sequences). However, the resulting network work just as well on all experiments from the main code, including list-linking.

# %%
# Based on the code for the Stimulus-response task as described in Miconi et al. ICLR 2019.

import torch
import torch.nn as nn
import numpy as np
from numpy import random
import torch.nn.functional as F
import time
import platform
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

myseed = -1


# If running this code on a cluster, uncomment the following, and pass a RNG seed as the --seed parameter on the command line
# parser  = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=-1)
# args = parser.parse_args()
# myseed =  args.seed


np.set_printoptions(precision=5)
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'


# fmt: off
# Global parameters
GG                  = {}
GG["rngseed"]       = myseed    # RNG seed, or -1 for no seed
GG["rew"]           = 1.0       # reward amount
GG["wp"]            = 0.0       # penalty for taking action 1 (not used here)
GG["bent"]          = 0.1       #  entropy incentive (actually sum-of-squares)
GG["blossv"]        = 0.1       # value prediction loss coefficient
GG["gr"]            = 0.9       # Gamma for temporal reward discounting

GG["hs"]            = 200       # Size of the RNN's hidden layer
GG["bs"]            = 32        # Batch size
GG["gc"]            = 2.0       # Gradient clipping
GG["eps"]           = 1e-6      # A parameter for Adam
GG["nbiter"]        = 30000     # 60000
GG["save_every"]    = 200
GG["pe"]            = 101       # "print every"


GG["nbcuesrange"]   = range(4, 9)  # The total number of cues varies from one episode to the next

GG["cs"] = (
    15  # 10     # Cue size -  number of binary elements in each cue vector (not including the 'go' bit and additional inputs, see below)
)

GG["triallen"] = 4  # Number of time steps in each trial
NUMRESPONSESTEP = 1
GG["nbtraintrials"] = (
    20  #  The first  nbtraintrials are the "train" trials. This  is included in nbtrials.
)
GG["nbtesttrials"] = (
    10  #  The last nbtesttrials are the "test" trials. This  is included in nbtrials.
)
GG["nbtrials"] = (
    GG["nbtraintrials"] + GG["nbtesttrials"]
)  # Number of trials per episode
GG["eplen"] = GG["nbtrials"] * GG["triallen"]  # eplen = episode length
GG["testlmult"] = 3.0  # multiplier for the loss during the test trials
GG["l2"] = 0  # 1e-5 # L2 penalty
GG["lr"] = 1e-4
GG["lpw"] = 1e-4  #  3    # plastic weight loss
# fmt: on


# RNN with plastic connections and neuromodulation ("DA").
# Plasticity only in the recurrent connections.


class RetroModulRNN(nn.Module):
    def __init__(self, GG):
        super(RetroModulRNN, self).__init__()
        # NOTE: 'outputsize' excludes the value and neuromodulator outputs!
        for paramname in ["outputsize", "inputsize", "hs", "bs"]:
            if paramname not in GG.keys():
                raise KeyError(
                    "Must provide missing key in argument 'GG': " + paramname
                )
        NBDA = 2  # 2 DA neurons, we  take the difference  - see below
        self.GG = GG
        self.activ = torch.tanh
        self.i2h = torch.nn.Linear(self.GG["inputsize"], GG["hs"]).to(device)
        self.w = torch.nn.Parameter(
            (
                (1.0 / np.sqrt(GG["hs"])) * (2.0 * torch.rand(GG["hs"], GG["hs"]) - 1.0)
            ).to(device),
            requires_grad=True,
        )
        self.alpha = 0.01 * (2.0 * torch.rand(GG["hs"], GG["hs"]) - 1.0).to(device)
        self.alpha = torch.nn.Parameter(self.alpha, requires_grad=True)
        self.etaet = torch.nn.Parameter(
            (0.7 * torch.ones(1)).to(device), requires_grad=True
        )  # Everyone has the same etaet
        self.DAmult = torch.nn.Parameter(
            (1.0 * torch.ones(1)).to(device), requires_grad=True
        )  # Everyone has the same DAmult
        self.h2DA = torch.nn.Linear(GG["hs"], NBDA).to(device)  # DA output
        self.h2o = torch.nn.Linear(GG["hs"], self.GG["outputsize"]).to(
            device
        )  # Actual output
        self.h2v = torch.nn.Linear(GG["hs"], 1).to(device)  # V prediction

    def forward(self, inputs, hidden, et, pw):
        BATCHSIZE = inputs.shape[0]  #  self.GG['bs']
        HS = self.GG["hs"]
        assert pw.shape[0] == hidden.shape[0] == et.shape[0] == BATCHSIZE

        # Multiplying inputs (i.e. current hidden  values) by the total recurrent weights, w + alpha  * plastic_weights
        hactiv = self.activ(
            self.i2h(inputs).view(BATCHSIZE, HS, 1)
            + torch.matmul(
                (self.w + torch.mul(self.alpha, pw)), hidden.view(BATCHSIZE, HS, 1)
            )
        ).view(BATCHSIZE, HS)
        activout = self.h2o(
            hactiv
        )  # Output layer. Pure linear, raw scores - will be softmaxed later
        valueout = self.h2v(hactiv)  # Value prediction

        # Now computing the Hebbian updates...

        # With batching, DAout is a matrix of size BS x 1
        DAout2 = torch.tanh(self.h2DA(hactiv))
        DAout = (
            self.DAmult * (DAout2[:, 0] - DAout2[:, 1])[:, None]
        )  # DA output is the difference between two tanh neurons - allows negative, positive and easy stable 0 output (by jamming both neurons to max or min)

        # Eligibility trace gets stamped into the plastic weights  - gated by DAout
        deltapw = DAout.view(BATCHSIZE, 1, 1) * et
        pw = pw + deltapw

        torch.clip_(pw, min=-50.0, max=50.0)

        # Updating the eligibility trace - Hebbbian update with a simple decay
        # NOTE: the decay is for the eligibility trace, NOT the plastic weights (which never decay during a lifetime, i.e. an episode)
        deltaet = torch.bmm(
            hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)
        )  # batched outer product; at this point 'hactiv' is the output and 'hidden' is the input  (i.e. ativities from previous time step)
        deltaet = torch.tanh(deltaet)
        et = (1 - self.etaet) * et + self.etaet * deltaet

        hidden = hactiv
        return activout, valueout, DAout, hidden, et, pw

    def initialZeroET(self, mybs):
        # return torch.zeros(self.GG['bs'], self.GG['hs'], self.GG['hs'], requires_grad=False).to(device)
        return torch.zeros(mybs, self.GG["hs"], self.GG["hs"], requires_grad=False).to(
            device
        )

    def initialZeroPlasticWeights(self, mybs):
        return torch.zeros(mybs, self.GG["hs"], self.GG["hs"], requires_grad=False).to(
            device
        )

    def initialZeroState(self, mybs):
        return torch.zeros(mybs, self.GG["hs"], requires_grad=False).to(device)


print("Starting...")

print("Passed GG: ", GG)
print(platform.uname())
suffix = (
    "_"
    + "".join(
        [
            (
                str(kk) + str(vv) + "_"
                if kk != "pe"
                and kk != "nbsteps"
                and kk != "rngseed"
                and kk != "save_every"
                and kk != "test_every"
                else ""
            )
            for kk, vv in sorted(zip(GG.keys(), GG.values()))
        ]
    )
    + "_rng"
    + str(GG["rngseed"])
)  # Turning the parameters into a nice suffix for filenames
print(suffix)


# Total input size = cue size +  one 'go' bit + 4 additional inputs
ADDINPUT = 4  # Additional inputs: 1 inputs for the previous reward, 1 inputs for numstep, 1 unused,  1 "Bias" inputs
NBSTIMBITS = (
    2 * GG["cs"] + 1
)  # The additional bit is for the response cue (i.e. the "Go" cue)
GG["outputsize"] = 2  # "response" and "no response"
GG["inputsize"] = (
    NBSTIMBITS + ADDINPUT + GG["outputsize"]
)  # The total number of input bits is the size of cues, plus the "response cue" binary input, plus the number of additional inputs, plus the number of actions


# Initialize random seeds, unless rngseed is -1 (first two redundant?)
if GG["rngseed"] > -1:
    print("Setting random seed", GG["rngseed"])
    np.random.seed(GG["rngseed"])
    random.seed(GG["rngseed"])
    torch.manual_seed(GG["rngseed"])
else:
    print("No random seed.")


BS = GG["bs"]  # Batch size


print("Initializing network")
net = RetroModulRNN(GG)


print("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
print("Size (numel) of all optimized elements:", allsizes)
print("Total size (numel) of all optimized elements:", sum(allsizes))

print("Initializing optimizer")
optimizer = torch.optim.Adam(
    net.parameters(), lr=1.0 * GG["lr"], eps=GG["eps"], weight_decay=GG["l2"]
)


lossbetweensaves = 0
nowtime = time.time()

nbtrials = [0] * BS
totalnbtrials = 0
nbtrialswithcc = 0
all_mean_testrewards_ep = []


print("Starting episodes!")

for numepisode in range(GG["nbiter"]):

    PRINTTRACE = False
    if (numepisode) % (GG["pe"]) == 0:
        PRINTTRACE = True

    optimizer.zero_grad()
    loss = 0
    lossv = 0
    GG["nbcues"] = random.choice(GG["nbcuesrange"])
    hidden = net.initialZeroState(BS)
    et = net.initialZeroET(BS)  #  The Hebbian eligibility trace

    # In this simplified version we just reset the plastic weights at every episode (the main version only resets it every 3rd episode and remembers previous lists)
    pw = net.initialZeroPlasticWeights(BS)

    numstep_ep = 0
    iscorrect_thisep = np.zeros((BS, GG["nbtrials"]))
    istest_thisep = np.zeros((BS, GG["nbtrials"]))
    isadjacent_thisep = np.zeros((BS, GG["nbtrials"]))
    # isolddata_thisep  = np.zeros((BS, GG['nbtrials']))
    resps_thisep = np.zeros((BS, GG["nbtrials"]))
    cuepairs_thisep = []
    numactionschosen_alltrialsandsteps_thisep = np.zeros(
        (BS, GG["nbtrials"], GG["triallen"])
    ).astype(int)

    # Generate the bitstring for each cue number for this episode. Make sure they're all different (important when using very small cues for debugging, e.g. cs=2, ni=2)
    cuedata = []
    for nb in range(BS):
        cuedata.append([])
        for ncue in range(GG["nbcues"]):
            assert len(cuedata[nb]) == ncue
            foundsame = 1
            cpt = 0
            while foundsame > 0:
                cpt += 1
                if cpt > 10000:
                    # This should only occur with very weird parameters, e.g. cs=2, ni>4
                    raise ValueError("Could not generate a full list of different cues")
                foundsame = 0
                candidate = np.random.randint(2, size=GG["cs"]) * 2 - 1
                for backtrace in range(ncue):
                    # if np.array_equal(cuedata[nb][backtrace], candidate):
                    if np.mean(cuedata[nb][backtrace] == candidate) > 0.66:
                        foundsame = 1

            cuedata[nb].append(candidate)

    reward = np.zeros(BS)
    sumreward = np.zeros(BS)
    sumrewardtest = np.zeros(BS)
    rewards = []
    vs = []
    logprobs = []
    cues = []
    for nb in range(BS):
        cues.append([])
    dist = 0
    numactionschosen = np.zeros(BS, dtype="int32")

    nbtrials = np.zeros(BS)
    nbtesttrials = nbtesttrials_correct = nbtesttrials_adjcues = (
        nbtesttrials_adjcues_correct
    ) = nbtesttrials_nonadjcues = nbtesttrials_nonadjcues_correct = 0
    nbrewardabletrials = np.zeros(BS)
    thistrialhascorrectorder = np.zeros(BS)
    thistrialhasadjacentcues = np.zeros(BS)
    thistrialhascorrectanswer = np.zeros(BS)

    # 2 steps of blank input between episodes. Not sure if it helps.
    inputs = np.zeros((BS, GG["inputsize"]), dtype="float32")
    inputsC = torch.from_numpy(inputs).detach().to(device)
    for _ in range(2):
        y, v, DAout, hidden, et, pw = net(
            inputsC, hidden, et, pw
        )  # y  should output raw scores, not probas

    for numtrial in range(GG["nbtrials"]):

        # To simplify dynamics as much as possible, we reset hidden activations and eligibility traces (but not plastic weights) between trials.
        hidden = net.initialZeroState(BS)
        et = net.initialZeroET(BS)

        # First, we prepare the specific sequence of inputs for this trial
        # The inputs can be a pair of cue numbers, or -1 (empty stimulus), or a single number equal to GG['nbcues'], which indicates the 'response' cue.
        # These will be translated into actual network inputs (using the actual bitstrings) later.
        # Remember that the actual data for each cue  (i.e. its actual bitstring) is randomly generated for each episode, above

        cuepairs_thistrial = []
        for nb in range(BS):
            thistrialhascorrectorder[nb] = 0
            cuerange = range(GG["nbcues"])
            # # In any trial, we show exactly two cues (randomly chosen), simultaneously:
            cuepair = list(np.random.choice(cuerange, 2, replace=False))

            # If the trial is NOT a test trial, these two cues should be adjacent
            if nbtrials[nb] < GG["nbtraintrials"]:
                while abs(cuepair[0] - cuepair[1]) > 1:
                    cuepair = list(np.random.choice(cuerange, 2, replace=False))
            else:
                assert nbtrials[nb] >= GG["nbtraintrials"]

            assert nbtrials[nb] == numtrial

            thistrialhascorrectorder[nb] = 1 if cuepair[0] < cuepair[1] else 0
            thistrialhasadjacentcues[nb] = (
                1 if (abs(cuepair[0] - cuepair[1]) == 1) else 0
            )
            isadjacent_thisep[nb, numtrial] = thistrialhasadjacentcues[nb]
            istest_thisep[nb, numtrial] = 1 if numtrial >= GG["nbtraintrials"] else 0

            # mycues = [cuepair,cuepair]
            mycues = [cuepair]
            cuepairs_thistrial.append(cuepair)

            mycues.append(
                GG["nbcues"]
            )  # The 'go' cue, instructing response from the network
            mycues.append(
                -1
            )  # One empty  step.During the first empty step, reward (computed on the previous step) is seen by the network.
            mycues.append(-1)
            # mycues.append(-1)
            assert len(mycues) == GG["triallen"]
            assert (
                mycues[NUMRESPONSESTEP] == GG["nbcues"]
            )  # The 'response' step is signalled by the 'go' cue, whose number is GG['nbcues'].
            cues[nb] = mycues

        cuepairs_thisep.append(cuepairs_thistrial)

        # Now we are ready to actually  run  the trial:

        for numstep in range(GG["triallen"]):

            # Preparing inputs
            inputs = np.zeros((BS, GG["inputsize"]), dtype="float32")
            for nb in range(BS):
                # Turning the cue number for this time step into actual (signed) bitstring inputs, using the cue  data generated at the beginning of the episode
                inputs[nb, :NBSTIMBITS] = 0
                if cues[nb][numstep] != -1 and cues[nb][numstep] != GG["nbcues"]:
                    assert len(cues[nb][numstep]) == 2
                    inputs[nb, : NBSTIMBITS - 1] = np.concatenate(
                        (
                            cuedata[nb][cues[nb][numstep][0]][:],
                            cuedata[nb][cues[nb][numstep][1]][:],
                        )
                    )
                if cues[nb][numstep] == GG["nbcues"]:
                    inputs[nb, NBSTIMBITS - 1] = 1  # "Go" cue

                inputs[nb, NBSTIMBITS + 0] = 1.0  # Bias neuron, probably not necessary
                inputs[nb, NBSTIMBITS + 1] = (
                    numstep_ep / GG["eplen"]
                )  # Time passed in this episode. Should it be the trial? Doesn't matter much anyway.
                inputs[nb, NBSTIMBITS + 2] = (
                    1.0 * reward[nb]
                )  # Reward from previous time step

                assert (
                    NUMRESPONSESTEP + 1 < GG["triallen"]
                )  # If that is not the case, we must provide the action signal in the next trial (this works)
                if numstep == NUMRESPONSESTEP + 1:
                    inputs[nb, NBSTIMBITS + ADDINPUT + numactionschosen[nb]] = (
                        1  # Previously chosen action, folowing standard meta-RL practice
                    )

            inputsC = torch.from_numpy(inputs).detach().to(device)

            ## Running the network
            y, v, DAout, hidden, et, pw = net(
                inputsC, hidden, et, pw
            )  # y  should output raw scores, not probas

            # Choosing the action from the outputs
            y = F.softmax(y, dim=1)
            # Must convert y to probas to use this !
            distrib = torch.distributions.Categorical(y)
            actionschosen = distrib.sample()
            logprobs.append(
                distrib.log_prob(actionschosen)
            )  # To be used later for the A2C algorithm
            # Alternatively: only record logprobs just after the response step (the only step where it matters). Better performance, but not used for the paper.
            # if numstep == NUMRESPONSESTEP:
            #     logprobs.append(distrib.log_prob(actionschosen))    # To be used later for the A2C algorithm
            # else:
            #     logprobs.append(0)
            numactionschosen = (
                actionschosen.data.cpu().numpy()
            )  # Store as scalars (for the whole batch)

            if PRINTTRACE:
                print(
                    "Tr",
                    numtrial,
                    "Step ",
                    numstep,
                    ", Cue 1  (0):",
                    inputs[0, : GG["cs"]],
                    "Cue 2 (0):",
                    inputs[0, GG["cs"] : 2 * GG["cs"]],
                    "Other inputs:",
                    inputs[0, 2 * GG["cs"] :],
                    "\n - Outputs(0): ",
                    y.data.cpu().numpy()[0, :],
                    " - action chosen(0): ",
                    numactionschosen[0],
                    "TrialLen:",
                    GG["triallen"],
                    "numstep",
                    numstep,
                    "TTHCC(0): ",
                    thistrialhascorrectorder[0],
                    "Reward (based on prev step): ",
                    reward[0],
                    ", DAout:",
                    float(DAout[0]),
                    ", cues(0):",
                    cues[0],
                )  # , ", cc(0):", correctcue[0])

            # Computing the rewards. This is done for each time step.
            reward = np.zeros(BS, dtype="float32")
            for nb in range(BS):

                numactionschosen_alltrialsandsteps_thisep[nb, numtrial, numstep] = (
                    numactionschosen[nb]
                )

                if numstep == NUMRESPONSESTEP:  # 2: # 4: #3: #  2:
                    # This is the 'response' step of the trial (and we showed the response signal
                    assert cues[nb][numstep] == GG["nbcues"]
                    resps_thisep[nb, numtrial] = (
                        numactionschosen[nb] * 2 - 1
                    )  # Store the response in this timestep as the response for the whole trial, for logging/analysis purposes
                    # We must deliver reward (which will be perceived by the agent at the next step), positive or negative, depending on response
                    thistrialhascorrectanswer[nb] = 1
                    if thistrialhascorrectorder[nb] and numactionschosen[nb] == 1:
                        reward[nb] += GG["rew"]
                    elif (not thistrialhascorrectorder[nb]) and numactionschosen[
                        nb
                    ] == 0:
                        reward[nb] += GG["rew"]
                    else:
                        reward[nb] -= GG["rew"]
                        thistrialhascorrectanswer[nb] = 0
                    iscorrect_thisep[nb, numtrial] = thistrialhascorrectanswer[nb]

                if numstep == GG["triallen"] - 1:
                    # This was the last step of the trial
                    nbtrials[nb] += 1
                    totalnbtrials += 1
                    if thistrialhascorrectorder[nb]:
                        nbtrialswithcc += 1

            rewards.append(reward)
            vs.append(v)
            sumreward += reward
            if numtrial >= GG["nbtrials"] - GG["nbtesttrials"]:
                sumrewardtest += reward

            loss += (
                GG["bent"] * y.pow(2).sum() / BS
            )  # In real A2c, this is an entropy incentive. Our original version of PyTorch did not have an entropy() function for Distribution, so we use sum-of-squares instead.

            numstep_ep += 1

        # All steps done for this trial
        if numtrial >= GG["nbtrials"] - GG["nbtesttrials"]:
            sumrewardtest += reward
            nbtesttrials += BS
            nbtesttrials_correct += np.sum(thistrialhascorrectanswer)
            nbtesttrials_adjcues += np.sum(thistrialhasadjacentcues)
            nbtesttrials_adjcues_correct += np.sum(
                thistrialhasadjacentcues * thistrialhascorrectanswer
            )
            nbtesttrials_nonadjcues += np.sum(1 - thistrialhasadjacentcues)
            nbtesttrials_nonadjcues_correct += np.sum(
                (1 - thistrialhasadjacentcues) * thistrialhascorrectanswer
            )

    # All trials done for this episode

    # Computing the various losses for A2C (outer-loop training)

    R = torch.zeros(BS, requires_grad=False).to(device)
    gammaR = GG["gr"]
    for numstepb in reversed(range(GG["eplen"])):
        R = gammaR * R + torch.from_numpy(rewards[numstepb]).detach().to(device)
        ctrR = R - vs[numstepb][:, 0]  # I think this is right...
        lossv += ctrR.pow(2).sum() / BS
        LOSSMULT = (
            GG["testlmult"]
            if numstepb > GG["eplen"] - GG["triallen"] * GG["nbtesttrials"]
            else 1.0
        )
        loss -= (
            LOSSMULT * (logprobs[numstepb] * ctrR.detach()).sum() / BS
        )  # Action policy loss

    lossobj = float(loss)
    loss += (
        GG["blossv"] * lossv
    )  # lossmult is not applied to value-prediction loss; is it right?...
    loss /= GG["eplen"]
    losspw = (
        torch.mean(pw**2) * GG["lpw"]
    )  # loss on squared final plastic weights is not divided by episode length
    loss += losspw

    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), GG["gc"])
    if numepisode > 100:  # Burn-in period
        optimizer.step()

    lossnum = float(loss)
    lossbetweensaves += lossnum
    all_mean_testrewards_ep.append(sumrewardtest.mean())

    if PRINTTRACE:

        print("Episode", numepisode, "====")
        previoustime = nowtime
        nowtime = time.time()
        print("Time spent on last", GG["pe"], "iters: ", nowtime - previoustime)

        print(
            " etaet: ",
            net.etaet.data.cpu().numpy(),
            " DAmult: ",
            float(net.DAmult),
            " mean-abs pw: ",
            np.mean(np.abs(pw.data.cpu().numpy())),
        )
        print("min/max/med-abs w, alpha, pw")
        print(
            float(torch.min(net.w)),
            float(torch.max(net.w)),
            float(torch.median(torch.abs(net.w))),
        )
        print(
            float(torch.min(net.alpha)),
            float(torch.max(net.alpha)),
            float(torch.median(torch.abs(net.alpha))),
        )
        print(
            float(torch.min(pw)),
            float(torch.max(pw)),
            float(torch.median(torch.abs(pw))),
        )

        # print("lossobj (with coeff):", lossobj / GG['eplen'], ", lossv (with coeff): ", GG['blossv'] * float(lossv) / GG['eplen'],
        # ", losspw:", float(losspw))
        # print ("Total reward for this episode(0):", sumreward[0], "Prop. of trials w/ rewarded cue:", (nbtrialswithcc / totalnbtrials),  " Total Nb of trials:", totalnbtrials)
        print(
            "Nb Test Trials:",
            nbtesttrials,
            ", Nb Test Trials AdjCues:",
            nbtesttrials_adjcues,
            ", Nb Test Trials NonAdjCues:",
            nbtesttrials_nonadjcues,
        )
        if nbtesttrials > 0:
            # Should always be the  case except for LinkedListsEval
            print(
                ">>>> Test Performance (both methods):",
                np.array(
                    [
                        nbtesttrials_correct / nbtesttrials,
                        np.sum(iscorrect_thisep * istest_thisep)
                        / np.sum(istest_thisep),
                    ]
                ),
                "Test Perf AdjCues:",
                (
                    np.array([nbtesttrials_adjcues_correct / nbtesttrials_adjcues])
                    if nbtesttrials_adjcues > 0
                    else "N/A"
                ),
                "Test Perf NonAdjCues:",
                (
                    np.array(
                        [nbtesttrials_nonadjcues_correct / nbtesttrials_nonadjcues]
                    )
                    if nbtesttrials_nonadjcues > 0
                    else "N/A"
                ),
            )

    if (numepisode) % GG["save_every"] == 0 and numepisode > 0:
        print("Saving local files...")

        if numepisode > 0:
            # print("Saving model parameters...")
            # torch.save(net.state_dict(), 'net_'+suffix+'.dat')
            torch.save(net.state_dict(), ROOT_DIR / ("netAE" + str(GG["rngseed"]) + ".dat"))
            torch.save(net.state_dict(), ROOT_DIR / "net.dat")

        # with open('rewards_'+suffix+'.txt', 'w') as thefile:
        #     for item in all_mean_rewards_ep[::10]:
        #             thefile.write("%s\n" % item)
        # with open('testrew_'+suffix+'.txt', 'w') as thefile:
        #     for item in all_mean_testrewards_ep[::10]:
        #             thefile.write("%s\n" % item)
        with open(ROOT_DIR / ("tAE" + str(GG["rngseed"]) + ".txt"), "w") as thefile:
            for item in all_mean_testrewards_ep[::10]:
                thefile.write("%s\n" % item)
