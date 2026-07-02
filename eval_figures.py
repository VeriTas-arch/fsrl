import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA


ROOT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADDINPUT = 4
NUMRESPONSESTEP = 1
ALPHABET = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def default_params(batch_size):
    params = {}
    params["rngseed"] = -1
    params["rew"] = 1.0
    params["wp"] = 0.0
    params["bent"] = 0.1
    params["blossv"] = 0.1
    params["gr"] = 0.9
    params["hs"] = 200
    params["bs"] = batch_size
    params["gc"] = 2.0
    params["eps"] = 1e-6
    params["nbiter"] = 1
    params["save_every"] = 200
    params["pe"] = 101
    params["nbcuesrange"] = range(4, 9)
    params["cs"] = 15
    params["triallen"] = 4
    params["nbtraintrials"] = 20
    params["nbtesttrials"] = 10
    params["nbtrials"] = params["nbtraintrials"] + params["nbtesttrials"]
    params["eplen"] = params["nbtrials"] * params["triallen"]
    params["testlmult"] = 3.0
    params["l2"] = 0
    params["lr"] = 1e-4
    params["lpw"] = 1e-4
    params["lda"] = 0
    params["lhl1"] = 0
    params["nbepsbwresets"] = 1
    params["nbcues"] = 8

    nbstimbits = 2 * params["cs"] + 1
    params["outputsize"] = 2
    params["inputsize"] = nbstimbits + ADDINPUT + params["outputsize"]
    return params


class RetroModulRNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        for paramname in ["outputsize", "inputsize", "hs", "bs"]:
            if paramname not in params:
                raise KeyError("Must provide missing key in params: " + paramname)

        nbda = 2
        self.params = params
        self.activ = torch.tanh
        self.i2h = torch.nn.Linear(params["inputsize"], params["hs"]).to(DEVICE)
        self.w = torch.nn.Parameter(
            ((1.0 / np.sqrt(params["hs"])) * (2.0 * torch.rand(params["hs"], params["hs"]) - 1.0)).to(DEVICE),
            requires_grad=True,
        )
        self.alpha = torch.nn.Parameter(
            (0.01 * (2.0 * torch.rand(params["hs"], params["hs"]) - 1.0)).to(DEVICE),
            requires_grad=True,
        )
        self.etaet = torch.nn.Parameter((0.7 * torch.ones(1)).to(DEVICE), requires_grad=True)
        self.DAmult = torch.nn.Parameter((1.0 * torch.ones(1)).to(DEVICE), requires_grad=True)
        self.h2DA = torch.nn.Linear(params["hs"], nbda).to(DEVICE)
        self.h2o = torch.nn.Linear(params["hs"], params["outputsize"]).to(DEVICE)
        self.h2v = torch.nn.Linear(params["hs"], 1).to(DEVICE)

    def forward(self, inputs, hidden, et, pw):
        batch_size = inputs.shape[0]
        hs = self.params["hs"]
        assert pw.shape[0] == hidden.shape[0] == et.shape[0] == batch_size

        hactiv = self.activ(
            self.i2h(inputs).view(batch_size, hs, 1)
            + torch.matmul((self.w + torch.mul(self.alpha, pw)), hidden.view(batch_size, hs, 1))
        ).view(batch_size, hs)
        activout = self.h2o(hactiv)
        valueout = self.h2v(hactiv)

        daout2 = torch.tanh(self.h2DA(hactiv))
        daout = self.DAmult * (daout2[:, 0] - daout2[:, 1])[:, None]

        pw = pw + daout.view(batch_size, 1, 1) * et
        torch.clip_(pw, min=-50.0, max=50.0)

        deltaet = torch.bmm(hactiv.view(batch_size, hs, 1), hidden.view(batch_size, 1, hs))
        deltaet = torch.tanh(deltaet)
        et = (1 - self.etaet) * et + self.etaet * deltaet

        return activout, valueout, daout, hactiv, et, pw

    def initialZeroET(self, batch_size):
        return torch.zeros(batch_size, self.params["hs"], self.params["hs"], requires_grad=False).to(DEVICE)

    def initialZeroPlasticWeights(self, batch_size):
        return torch.zeros(batch_size, self.params["hs"], self.params["hs"], requires_grad=False).to(DEVICE)

    def initialZeroState(self, batch_size):
        return torch.zeros(batch_size, self.params["hs"], requires_grad=False).to(DEVICE)


@dataclass
class EpisodeResult:
    correct: np.ndarray
    adjacent: np.ndarray
    cue_pairs: np.ndarray
    responses: np.ndarray
    actions: np.ndarray
    rates: np.ndarray
    cue_data: list
    pw_trial20_step1: np.ndarray | None


def resolve_model_path(model_path):
    if model_path:
        path = Path(model_path)
        return path if path.is_absolute() else ROOT_DIR / path
    for candidate in ("net.dat", "net_active.dat"):
        path = ROOT_DIR / candidate
        if path.exists():
            return path
    raise FileNotFoundError("Expected net.dat or net_active.dat in the repository root, or pass --model-path.")


def load_model_state(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_network(params, model_path):
    net = RetroModulRNN(params)
    net.load_state_dict(load_model_state(model_path))
    net.eval()
    return net


def set_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_cue_data(params, batch_size):
    cue_data = []
    for nb in range(batch_size):
        cue_data.append([])
        for ncue in range(params["nbcues"]):
            foundsame = 1
            attempts = 0
            while foundsame > 0:
                attempts += 1
                if attempts > 10000:
                    raise ValueError("Could not generate a full list of different cues")
                foundsame = 0
                candidate = np.random.randint(2, size=params["cs"]) * 2 - 1
                for backtrace in range(ncue):
                    if np.mean(cue_data[nb][backtrace] == candidate) > 0.66:
                        foundsame = 1
            cue_data[nb].append(candidate)
    return cue_data


def sample_cue_pair(params, nbtrials, batch_index, linked_lists, linking_is_sham, episode_index, trial_index):
    cue_range = range(params["nbcues"])
    if linked_lists:
        show_first_half_first = 1
        if show_first_half_first:
            if episode_index == 0:
                cue_range = range(params["nbcues"] // 2)
            elif episode_index == 1:
                cue_range = range(params["nbcues"] // 2, params["nbcues"])
        else:
            if episode_index == 0:
                cue_range = range(params["nbcues"] // 2, params["nbcues"])
            elif episode_index == 1:
                cue_range = range(params["nbcues"] // 2)

    cue_pair = list(np.random.choice(cue_range, 2, replace=False))

    if nbtrials[batch_index] < params["nbtraintrials"]:
        while abs(cue_pair[0] - cue_pair[1]) > 1:
            cue_pair = list(np.random.choice(cue_range, 2, replace=False))

    if linked_lists and episode_index == 2 and trial_index < params["nbtraintrials"]:
        if linking_is_sham:
            cue_pair = [params["nbcues"] // 2 - 3, params["nbcues"] // 2 - 2]
        else:
            cue_pair = (
                [params["nbcues"] // 2 - 1, params["nbcues"] // 2]
                if np.random.randint(2)
                else [params["nbcues"] // 2, params["nbcues"] // 2 - 1]
            )

    return cue_pair


def run_eval(params, net, linked_lists=False, linking_is_sham=False, keep_rates=True):
    torch.set_grad_enabled(False)
    batch_size = params["bs"]
    nbstimbits = 2 * params["cs"] + 1
    nb_episodes = 3 if linked_lists else 1
    if linked_lists:
        params["nbepsbwresets"] = 3
        params["nbiter"] = 3
        params["nbcues"] = 8
        params["nbtraintrials"] = 10
        params["nbtesttrials"] = 0
        params["nbtrials"] = params["nbtraintrials"] + params["nbtesttrials"]
        params["eplen"] = params["nbtrials"] * params["triallen"]

    old_cue_data = []
    cue_data = None
    results = []

    for episode_index in range(nb_episodes):
        if linked_lists and episode_index == 2:
            params["nbtraintrials"] = 1 if linking_is_sham else 4
            params["nbtesttrials"] = 1
            params["nbtrials"] = params["nbtraintrials"] + params["nbtesttrials"]
            params["eplen"] = params["nbtrials"] * params["triallen"]

        if episode_index % params["nbepsbwresets"] == 0:
            old_cue_data = []
            hidden = net.initialZeroState(batch_size)
            et = net.initialZeroET(batch_size)
            pw = net.initialZeroPlasticWeights(batch_size)
        else:
            hidden = hidden.detach()
            et = et.detach()
            pw = pw.detach()

        if not linked_lists or episode_index == 0:
            cue_data = generate_cue_data(params, batch_size)

        iscorrect = np.zeros((batch_size, params["nbtrials"]))
        isadjacent = np.zeros((batch_size, params["nbtrials"]))
        responses = np.zeros((batch_size, params["nbtrials"]))
        cue_pairs = []
        actions = np.zeros((batch_size, params["nbtrials"], params["triallen"])).astype(int)
        rates = np.zeros((batch_size, params["hs"], params["eplen"]), dtype="float32") if keep_rates else None
        pw_trial20_step1 = None

        reward = np.zeros(batch_size, dtype="float32")
        nbtrials = np.zeros(batch_size)
        numactionschosen = np.zeros(batch_size, dtype="int32")

        inputs = np.zeros((batch_size, params["inputsize"]), dtype="float32")
        inputs_t = torch.from_numpy(inputs).detach().to(DEVICE)
        for _ in range(2):
            _, _, _, hidden, et, pw = net(inputs_t, hidden, et, pw)

        numstep_ep = 0
        for trial_index in range(params["nbtrials"]):
            hidden = net.initialZeroState(batch_size)
            et = net.initialZeroET(batch_size)

            cues = []
            cue_pairs_thistrial = []
            correct_order = np.zeros(batch_size)
            adjacent = np.zeros(batch_size)
            correct_answer = np.zeros(batch_size)

            for nb in range(batch_size):
                cue_pair = sample_cue_pair(params, nbtrials, nb, linked_lists, linking_is_sham, episode_index, trial_index)
                correct_order[nb] = 1 if cue_pair[0] < cue_pair[1] else 0
                adjacent[nb] = 1 if abs(cue_pair[0] - cue_pair[1]) == 1 else 0
                isadjacent[nb, trial_index] = adjacent[nb]
                cue_pairs_thistrial.append(cue_pair)
                cues.append([cue_pair, params["nbcues"], -1, -1])

            cue_pairs.append(cue_pairs_thistrial)

            for numstep in range(params["triallen"]):
                inputs = np.zeros((batch_size, params["inputsize"]), dtype="float32")

                for nb in range(batch_size):
                    inputs[nb, :nbstimbits] = 0
                    cue = cues[nb][numstep]
                    if cue != -1 and cue != params["nbcues"]:
                        inputs[nb, : nbstimbits - 1] = np.concatenate(
                            (cue_data[nb][cue[0]][:], cue_data[nb][cue[1]][:])
                        )
                    if cue == params["nbcues"]:
                        inputs[nb, nbstimbits - 1] = 1

                    inputs[nb, nbstimbits + 0] = 1.0
                    inputs[nb, nbstimbits + 1] = numstep_ep / params["eplen"]
                    inputs[nb, nbstimbits + 2] = reward[nb]
                    if numstep == NUMRESPONSESTEP + 1:
                        inputs[nb, nbstimbits + ADDINPUT + numactionschosen[nb]] = 1

                inputs_t = torch.from_numpy(inputs).detach().to(DEVICE)
                y, _, _, hidden, et, pw = net(inputs_t, hidden, et, pw)

                if keep_rates:
                    rates[:, :, numstep_ep] = hidden.cpu().numpy()
                if trial_index == 19 and numstep == 0:
                    pw_trial20_step1 = pw.detach().cpu().numpy().astype("float32")

                y = F.softmax(y, dim=1)
                actionschosen = torch.distributions.Categorical(y).sample()
                numactionschosen = actionschosen.data.cpu().numpy()
                actions[:, trial_index, numstep] = numactionschosen

                reward = np.zeros(batch_size, dtype="float32")
                for nb in range(batch_size):
                    if numactionschosen[nb] == 1:
                        reward[nb] -= params["wp"]
                    if numstep == NUMRESPONSESTEP:
                        responses[nb, trial_index] = numactionschosen[nb] * 2 - 1
                        correct_answer[nb] = 1
                        if correct_order[nb] and numactionschosen[nb] == 1:
                            reward[nb] += params["rew"]
                        elif (not correct_order[nb]) and numactionschosen[nb] == 0:
                            reward[nb] += params["rew"]
                        else:
                            reward[nb] -= params["rew"]
                            correct_answer[nb] = 0
                        iscorrect[nb, trial_index] = correct_answer[nb]
                    if numstep == params["triallen"] - 1:
                        nbtrials[nb] += 1

                numstep_ep += 1

        old_cue_data.append(cue_data)
        results.append(
            EpisodeResult(
                correct=iscorrect.astype(int),
                adjacent=isadjacent.astype(int),
                cue_pairs=np.moveaxis(np.array(cue_pairs), 1, 0),
                responses=responses.astype(int),
                actions=actions,
                rates=rates,
                cue_data=cue_data,
                pw_trial20_step1=pw_trial20_step1,
            )
        )

    return results[-1]


def ordered_cue_pairs(nbcues):
    pairs = []
    for distance in range(1, nbcues):
        for start in range(nbcues):
            if start + distance >= nbcues:
                break
            pairs.append([start, start + distance])
    return pairs


def per_pair_performance(result, params, n_splits=10):
    pairs = ordered_cue_pairs(params["nbcues"])
    split_size = max(1, params["bs"] // n_splits)
    n_splits = max(1, params["bs"] // split_size)
    allperfs = []

    for split_index in range(n_splits):
        start = split_index * split_size
        stop = min(params["bs"], start + split_size)
        perfs = np.full(len(pairs), np.nan)
        for pos, pair in enumerate(pairs):
            values = []
            for nb in range(start, stop):
                for nt in range(result.cue_pairs.shape[1]):
                    if nt < params["nbtrials"] - params["nbtesttrials"]:
                        continue
                    trial_pair = result.cue_pairs[nb, nt]
                    if set(trial_pair) == set(pair):
                        values.append(result.correct[nb, nt])
            if values:
                perfs[pos] = np.mean(values)
        allperfs.append(perfs)

    return pairs, np.array(allperfs)


def nan_column_stat(values, reducer):
    stats = []
    for col in range(values.shape[1]):
        column = values[:, col]
        valid = column[~np.isnan(column)]
        stats.append(np.nan if valid.size == 0 else reducer(valid))
    return np.array(stats)


def plot_pair_performance(result, params, output_path, linked_lists=False):
    pairs, allperfs = per_pair_performance(result, params)

    fig, ax = plt.subplots(figsize=(5, 3))
    start = 0
    offset = 0
    group_len = params["nbcues"] - 1
    tick_positions = []
    tick_labels = []

    for distance_index in range(params["nbcues"] - 1):
        xs = list(range(start + offset, start + group_len + offset))
        group = allperfs[:, start : start + group_len]
        medians = nan_column_stat(group, np.median)
        q25 = nan_column_stat(group, lambda x: np.quantile(x, 0.25))
        q75 = nan_column_stat(group, lambda x: np.quantile(x, 0.75))
        ax.plot(xs, medians)
        ax.fill_between(xs, q25, q75, alpha=0.3)

        if group_len == 1:
            y = medians[0]
            ax.plot([start + offset], [y], ".")
            if not np.isnan(y):
                ax.errorbar([start + offset], [y], [[y - q25[0]], [q75[0] - y]])

        tick_positions.extend(xs)
        for numx, pair in enumerate(pairs[start : start + group_len]):
            tick_labels.append(("\n" if numx % 2 == 1 else "") + ALPHABET[pair[0]] + ALPHABET[pair[1]])

        start += group_len
        group_len -= 1
        offset += 2

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    if linked_lists:
        ax.axhline(y=0.5, color="k", linestyle="--")
        ax.set_ylabel("% correct (last test trial)")
    else:
        ax.set_ylabel("% correct (last " + str(params["nbtesttrials"]) + " trials)")
    fig.tight_layout()
    save_figure(fig, output_path)


def trial_labels(result, params):
    first_iscuenum = np.zeros((params["nbcues"], params["bs"], params["nbtrials"]))
    ordered = np.zeros((params["bs"], params["nbtrials"]))
    for nb in range(params["bs"]):
        for nt in range(params["nbtrials"]):
            first = result.cue_pairs[nb, nt, 0]
            second = result.cue_pairs[nb, nt, 1]
            first_iscuenum[first, nb, nt] = 1
            ordered[nb, nt] = 1 if first < second else 0
    return first_iscuenum, ordered


def pca_for_trial20(result, params):
    trial_index = 19
    pos_in_trial = 1
    rates = result.rates[:, :, pos_in_trial :: params["triallen"]]
    mx = rates[:, :, trial_index]
    n_components = min(50, mx.shape[0], mx.shape[1])
    pca = PCA(n_components=n_components)
    mx_pca = pca.fit_transform(mx)
    return pca, mx_pca


def plot_fig4a(result, params, net, output_path):
    pca, mx_pca = pca_for_trial20(result, params)
    first_iscuenum, ordered = trial_labels(result, params)
    trial_index = 19

    wo = net.h2o.weight.detach().cpu().numpy()
    wo = wo[1, :] - wo[0, :]
    wo_pca = pca.transform(wo[None, :])[0, :]

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.ravel()

    resp_pos = mx_pca[result.responses[:, trial_index] == 1]
    resp_neg = mx_pca[result.responses[:, trial_index] == -1]
    axes[0].plot(resp_pos[:, 0], resp_pos[:, 1], "+c", alpha=0.3, label="Choose Stim1")
    axes[0].plot(resp_neg[:, 0], resp_neg[:, 1], ".r", alpha=0.2, label="Choose Stim2")
    axes[0].arrow(0, 0, 1.3 * wo_pca[0], 1.3 * wo_pca[1], color="k", zorder=10, width=0.1, head_width=0.5)
    axes[0].text(0, -0.75, r"$\mathbf{W_{out}}$", fontsize=15)
    axes[0].legend()

    ordered_points = mx_pca[ordered[:, trial_index] == 1]
    unordered_points = mx_pca[ordered[:, trial_index] == 0]
    axes[1].plot(ordered_points[:, 0], ordered_points[:, 1], "+c", alpha=0.3, label="Stim1>Stim2")
    axes[1].plot(unordered_points[:, 0], unordered_points[:, 1], ".r", alpha=0.2, label="Stim2>Stim1")
    axes[1].legend()

    correct_points = mx_pca[result.correct[:, trial_index] == 1]
    wrong_points = mx_pca[result.correct[:, trial_index] == 0]
    axes[2].plot(correct_points[:, 0], correct_points[:, 1], "+c", alpha=0.3, label="Correct")
    axes[2].plot(wrong_points[:, 0], wrong_points[:, 1], ".r", alpha=0.2, label="Error")
    axes[2].legend()

    colors = ["g", "r", "b", "y"]
    cue_indices = [0, 2, 3, 4]
    cue_labels = ["Cue1:A", "Cue1:B", "Cue1:C", "Cue1:D"]
    for cue_index, color, label in zip(cue_indices, colors, cue_labels):
        points = mx_pca[first_iscuenum[cue_index, :, trial_index] == 1]
        axes[3].plot(points[:, 0], points[:, 1], ".", color=color, alpha=0.3, label=label)
    axes[3].legend(ncol=2)

    for ax in axes:
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    fig.tight_layout()
    save_figure(fig, output_path)


def single_item_alignment(result, params, net):
    if result.pw_trial20_step1 is None:
        raise ValueError("Trial 20 plastic weights were not captured; run standard eval with at least 20 trials.")

    batch_size = params["bs"]
    nbstimbits = 2 * params["cs"] + 1
    inputs = np.zeros((batch_size, params["inputsize"]), dtype="float32")
    cue_data = np.array(result.cue_data)
    pwtest = torch.from_numpy(result.pw_trial20_step1).to(DEVICE)

    wo = net.h2o.weight.detach().cpu().numpy()
    wo = wo[1, :] - wo[0, :]

    allcorrs_s1 = []
    allcorrs_s2 = []

    for cue_index in range(params["nbcues"]):
        inputs[:, :nbstimbits] = 0
        inputs[:, : params["cs"]] = cue_data[:, cue_index, :]
        inputs[:, nbstimbits + 0] = 1.0
        inputs[:, nbstimbits + 1] = 0
        inputs[:, nbstimbits + 2] = 0

        inputs0 = torch.from_numpy(inputs).detach().to(DEVICE)
        inputs1 = inputs0.clone()
        inputs1[:, :nbstimbits] = 0

        hidden = net.initialZeroState(batch_size)
        et = net.initialZeroET(batch_size)

        _, _, _, hidden, et, pw = net(inputs0, hidden, et, pwtest)
        _, _, _, hiddenout, _, _ = net(inputs1, hidden, et, pw)

        z2 = np.corrcoef(hiddenout.detach().cpu().numpy(), wo)[:-1, -1]
        z1 = np.corrcoef(hidden.detach().cpu().numpy(), wo)[:-1, -1]
        allcorrs_s2.append(z2)
        allcorrs_s1.append(z1)

    return np.array(allcorrs_s1), np.array(allcorrs_s2)


def plot_fig4b(result, params, net, output_path):
    _, allcorrs_s2 = single_item_alignment(result, params, net)

    fig, ax = plt.subplots(figsize=(3.33, 3))
    ax.set_xticks(np.arange(params["nbcues"]))
    ax.set_xticklabels(ALPHABET[: params["nbcues"]])
    ax.plot(np.mean(allcorrs_s2, axis=1), color="orange")
    ax.errorbar(
        np.arange(params["nbcues"]),
        np.mean(allcorrs_s2, axis=1),
        yerr=np.std(allcorrs_s2, axis=1),
        color="b",
        marker="o",
        linestyle="none",
    )
    ax.set_xlabel("Single item (X)")
    ax.set_ylabel(r"Representation alignment" + "\n" + r"Corr($\psi_{t2}(X), \mathbf{w}_{out}$)")
    fig.tight_layout()
    save_figure(fig, output_path)


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Teaching eval code for NN.pdf Fig. 2a, 3a, 4a, and 4b.")
    parser.add_argument("--figures", nargs="+", default=["all"], choices=["all", "fig2a", "fig3a", "fig4a", "fig4b"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    figures = {"fig2a", "fig3a", "fig4a", "fig4b"} if "all" in args.figures else set(args.figures)
    output_dir = ROOT_DIR / args.output_dir
    model_path = resolve_model_path(args.model_path)
    set_seed(args.seed)

    if figures & {"fig2a", "fig4a", "fig4b"}:
        params = default_params(args.batch_size)
        net = load_network(params, model_path)
        result = run_eval(params, net, keep_rates=bool(figures & {"fig4a"}))
        if "fig2a" in figures:
            plot_pair_performance(result, params, output_dir / "fig2a_sde", linked_lists=False)
        if "fig4a" in figures:
            plot_fig4a(result, params, net, output_dir / "fig4a_pca")
        if "fig4b" in figures:
            plot_fig4b(result, params, net, output_dir / "fig4b_single_item_alignment")

    if "fig3a" in figures:
        params = default_params(args.batch_size)
        net = load_network(params, model_path)
        result = run_eval(params, net, linked_lists=True, linking_is_sham=False, keep_rates=False)
        plot_pair_performance(result, params, output_dir / "fig3a_linked_sde", linked_lists=True)

    print("Saved requested figures to", output_dir)


if __name__ == "__main__":
    main()
