"""Generate figures, kernel probes and Grad-CAM from actual checkpoints."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from models import PlainCNN
from run_experiments import ROOT, RESULTS, CHECKPOINTS, preprocess, augment_view, write_csv, save_json

FIG = RESULTS / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140, "savefig.dpi": 160})


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def data_figures(config, classes):
    dataset = CIFAR10(str(ROOT / "data"), train=True, download=False)
    ids = np.load(RESULTS / "split_indices.npz")["train"]
    y = np.array(dataset.targets)[ids]
    x = torch.from_numpy(dataset.data[ids].copy()).permute(0, 3, 1, 2).float()/255
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.5), layout="constrained")
    for c in range(10):
        for row, i in enumerate(np.flatnonzero(y == c)[:2]):
            axes[row, c].imshow(x[i].permute(1, 2, 0))
            axes[row, c].axis("off")
            if row == 0:
                axes[row, c].set_title(classes[c])
    fig.suptitle("CIFAR-10: training examples only")
    save(fig, "dataset_samples.png")
    views = [x] + [augment_view(x, config["seed"]+i+1) for i in range(config["augmentation_views"])]
    fig, axes = plt.subplots(3, 6, figsize=(10, 5.2), layout="constrained")
    for row in range(3):
        for col in range(6):
            axes[row, col].imshow(views[row][col].permute(1, 2, 0))
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(["Original", "View 1", "View 2"][row])
            if row == 0:
                axes[row, col].set_title(classes[y[col]])
    fig.suptitle("Training augmentation: random crop + horizontal flip")
    save(fig, "augmentation_examples.png")


def metric_figures(metrics, selected, classes):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    colors = ["#2563eb", "#ea580c", "#16a34a"]
    for row, family in enumerate(["plain", "resnet"]):
        subset = [r for r in metrics if r["family"] == family]
        for color, run in zip(colors, subset):
            history = read_csv(RESULTS / "history" / f"{run['run']}.csv")
            epochs = [int(r["epoch"]) for r in history]
            label = f"lr={run['lr']:g}, {'aug' if run['augmentation'] else 'base'}"
            axes[row, 0].plot(epochs, [float(r["train_loss"]) for r in history], color=color, linestyle="--", alpha=.65)
            axes[row, 0].plot(epochs, [float(r["val_loss"]) for r in history], color=color, label=label)
            axes[row, 1].plot(epochs, [float(r["val_top1"])*100 for r in history], color=color, label=label)
            axes[row, 1].scatter(run["best_epoch"], run["val_top1"]*100, color=color)
        axes[row, 0].set_title(f"{family}: loss (solid=validation, dashed=train)")
        axes[row, 1].set_title(f"{family}: validation Top-1")
        for ax in axes[row]:
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        axes[row, 0].set_ylabel("Cross entropy")
        axes[row, 1].set_ylabel("Accuracy (%)")
    save(fig, "learning_curves.png")
    # Per-class One-vs-Rest ROC for every completed run.
    for run in metrics:
        bundle = np.load(RESULTS / "predictions" / f"{run['run']}.npz")
        p, y = bundle["probabilities"], bundle["labels"]
        fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
        curve_rows = []
        for c, cls in enumerate(classes):
            fpr, tpr, thresholds = roc_curve(y == c, p[:, c])
            ax.plot(fpr, tpr, lw=1.3, label=f"{cls}: {auc(fpr, tpr):.3f}")
            curve_rows.extend({"class": cls, "fpr": float(f), "tpr": float(t), "threshold": float(s)}
                              for f, t, s in zip(fpr, tpr, thresholds))
        write_csv(RESULTS / "roc" / f"{run['run']}.csv", curve_rows)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.015),
               title=f"{run['run']}\nTest One-vs-Rest ROC | Macro-AUC={run['test_macro_auc']:.4f}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=.2)
        save(fig, f"roc_{run['run']}.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), layout="constrained")
    for ax, family in zip(axes, ["plain", "resnet"]):
        name = selected[family]["final"]
        cm = np.loadtxt(RESULTS / "per_class" / f"{name}_confusion.csv", delimiter=",", skiprows=1)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1000)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 500 else "black")
        ax.set_xticks(range(10), classes, rotation=50, ha="right")
        ax.set_yticks(range(10), classes)
        ax.set(title=f"{family}: selected model", xlabel="Predicted class", ylabel="True class")
    save(fig, "confusion_matrices.png")
    comparison = []
    for family in ["plain", "resnet"]:
        comparison.extend(next(r for r in metrics if r["run"] == selected[family][kind]) for kind in ["baseline", "augmented"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    xx = np.arange(4)
    for offset, key, label, color in [(-.18, "test_top1", "Top-1", "#2563eb"), (.18, "test_top5", "Top-5", "#93c5fd")]:
        bars = axes[0].bar(xx+offset, [r[key]*100 for r in comparison], .35, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8)
    labels = [f"{r['family']}\n{'augmentation' if r['augmentation'] else 'baseline'}" for r in comparison]
    axes[0].set_xticks(xx, labels)
    axes[0].set(ylabel="Test accuracy (%)", ylim=(0, 110), title="Controlled augmentation comparison")
    axes[0].legend(loc="lower right")
    final = [next(r for r in metrics if r["run"] == selected[f]["final"]) for f in ["plain", "resnet"]]
    for offset, key, label in [(-.18, "parameters_total", "Total"), (.18, "parameters_trainable", "Trainable")]:
        bars = axes[1].bar(np.arange(2)+offset, [r[key] for r in final], .35, label=label)
        axes[1].bar_label(bars, labels=[f"{r[key]:,}" for r in final], fontsize=8)
    axes[1].set_xticks([0, 1], ["Plain CNN", "ResNet-18"])
    axes[1].set(yscale="log", ylim=(1000, 5e7), ylabel="Parameter count (log scale)", title="Final model parameters")
    axes[1].legend()
    save(fig, "model_comparison.png")


def kernel_figures(model):
    weights = model.features[0].weight.detach().numpy()[:2]
    np.save(RESULTS / "kernel_weights.npy", weights)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), layout="constrained")
    bound = abs(weights).max()
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            ax.imshow(weights[row, col], cmap="RdBu_r", vmin=-bound, vmax=bound)
            ax.set_title(f"Kernel {row} / {'RGB'[col]} channel")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{weights[row,col,i,j]:.2f}", ha="center", va="center", fontsize=8)
        rgb = weights[row].transpose(1, 2, 0)
        axes[row, 3].imshow((rgb + bound)/(2*bound))
        axes[row, 3].set_title(f"Kernel {row} / RGB weights")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("First convolution: two trained 3 x 3 x 3 kernels\nShared symmetric scale; RGB view is a weight visualization")
    save(fig, "two_kernels.png")
    # Controlled, synthetic probes: raw linear convolution, with no bias/ReLU.
    grid = torch.arange(9)
    vertical = ((grid[None, :] >= 4).float()*2-1).expand(9, 9)
    horizontal = vertical.T
    patterns = {"vertical edge": vertical.expand(3, 9, 9), "horizontal edge": horizontal.expand(3, 9, 9),
                "diagonal edge": ((grid[:, None] > grid[None, :]).float()*2-1).expand(3, 9, 9),
                "red field": torch.tensor([1., 0., 0.])[:, None, None].expand(3, 9, 9),
                "green field": torch.tensor([0., 1., 0.])[:, None, None].expand(3, 9, 9),
                "blue field": torch.tensor([0., 0., 1.])[:, None, None].expand(3, 9, 9)}
    probes = torch.stack(list(patterns.values()))
    response = F.conv2d(probes, torch.from_numpy(weights)).abs().mean((2, 3)).T.numpy()
    write_csv(RESULTS / "kernel_probe_responses.csv", [{"kernel": k, **{key: float(response[k, i]) for i, key in enumerate(patterns)}} for k in range(2)])
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for i in range(2):
        ax.bar(np.arange(6) + (i-.5)*.36, response[i], .36, label=f"Kernel {i}")
    ax.set_xticks(np.arange(6), list(patterns), rotation=15)
    ax.set(ylabel="Mean absolute linear response", title="Kernel probes (normalized-coordinate synthetic inputs)")
    ax.legend()
    save(fig, "kernel_probes.png")


def grad_cam(model, x, target):
    activations = []
    handle = model.features[6].register_forward_hook(lambda module, inputs, output: activations.append(output))
    try:
        logits = model(x)
        gradients = torch.autograd.grad(logits[0, target], activations[0])[0]
        alpha = gradients.mean((2, 3), keepdim=True)
        heat = (alpha * activations[0]).sum(1, keepdim=True).relu()
        coarse = heat.detach().numpy()[0, 0]
        heat = F.interpolate(heat, (32, 32), mode="bilinear", align_corners=False)[0, 0].detach()
        heat = heat / heat.max().clamp_min(1e-12)
        return heat.numpy(), coarse
    finally:
        handle.remove()


def xai_figures(model, name, classes):
    dataset = CIFAR10(str(ROOT / "data"), train=False, download=False)
    bundle = np.load(RESULTS / "predictions" / f"{name}.npz")
    p, y = bundle["probabilities"], bundle["labels"]
    predicted = p.argmax(1)
    chosen = []
    # First two correct and first two errors, distinct true classes in each group.
    for correct in [True, False]:
        used = set()
        for i in np.flatnonzero((predicted == y) == correct):
            if int(y[i]) not in used:
                chosen.append(int(i))
                used.add(int(y[i]))
                if len(used) == 2:
                    break
    fig, axes = plt.subplots(4, 3, figsize=(9, 11), layout="constrained")
    rows, maps, coarse_maps = [], [], []
    for row, idx in enumerate(chosen):
        image = dataset.data[idx]/255.
        x = preprocess(torch.from_numpy(dataset.data[idx].copy()).permute(2, 0, 1)[None].float()/255)
        target = int(predicted[idx])
        heat, coarse = grad_cam(model, x, target)
        maps.append(heat)
        coarse_maps.append(coarse)
        order = np.argsort(heat.ravel())
        masked_probabilities = []
        for pixel_ids in [order[-256:], order[:256]]:
            masked = x.clone().view(1, 3, -1)
            masked[:, :, torch.from_numpy(pixel_ids.copy())] = 0  # ImageNet channel means.
            with torch.no_grad():
                masked_probabilities.append(float(model(masked.view_as(x)).softmax(1)[0, target]))
        original_probability = float(p[idx, target])
        mass = heat.sum()
        yy, xx = np.indices((32, 32))
        rows.append({"test_index": idx, "true_class": classes[y[idx]], "target_predicted_class": classes[target],
                     "correct": bool(target == y[idx]), "probability": original_probability,
                     "top_quarter_masked_probability": masked_probabilities[0],
                     "bottom_quarter_masked_probability": masked_probabilities[1],
                     "top_quarter_drop": original_probability-masked_probabilities[0],
                     "bottom_quarter_drop": original_probability-masked_probabilities[1],
                     "cam_center_x": float((heat*xx).sum()/mass) if mass else None,
                     "cam_center_y": float((heat*yy).sum()/mass) if mass else None,
                     "cam_nonzero": bool(mass > 0)})
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Test #{idx} | true: {classes[y[idx]]}\npred: {classes[target]} ({original_probability:.1%})")
        axes[row, 1].imshow(heat, cmap="inferno", vmin=0, vmax=1)
        axes[row, 1].set_title("Grad-CAM: predicted class")
        axes[row, 2].imshow(np.clip(.6*image + .4*plt.cm.inferno(heat)[:, :, :3], 0, 1))
        axes[row, 2].set_title("Overlay | final convolution (8 x 8)")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"{name}: gradient-based explanations\nFirst two correct / first two errors; no cherry-picking by confidence")
    save(fig, "gradcam_examples.png")
    write_csv(RESULTS / "xai_cases.csv", rows)
    np.savez_compressed(RESULTS / "gradcam_maps.npz", test_indices=chosen, normalized_maps=maps, raw_maps=coarse_maps)
    save_json(RESULTS / "xai_metadata.json", {"model": name, "target": "predicted class logit", "layer": "features[6]",
              "spatial_resolution": [8, 8], "selection": "first two correct and first two errors with distinct true classes per group",
              "occlusion": "mask 256/1024 highest or lowest CAM pixels to ImageNet channel means"})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(config["threads"])
    classes = json.loads((RESULTS / "classes.json").read_text())
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    selected = json.loads((RESULTS / "selection.json").read_text())["selected"]
    data_figures(config, classes)
    metric_figures(metrics, selected, classes)
    name = selected["plain"]["final"]
    model = PlainCNN()
    model.load_state_dict(torch.load(CHECKPOINTS / f"{name}.pt", weights_only=True, map_location="cpu")["state_dict"])
    model.eval()
    kernel_figures(model)
    xai_figures(model, name, classes)
    print("All figures and explanations generated from saved results.")


if __name__ == "__main__":
    main()
