"""Independent integrity checks for the submitted experimental evidence."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torchvision.datasets import CIFAR10

from models import PlainCNN, make_resnet, parameter_counts
from predict import load_selected
from run_experiments import ROOT, RESULTS, CHECKPOINTS, preprocess, save_json


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    checks = []
    config = json.loads((ROOT / "experiment_config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(config["threads"])
    split = np.load(RESULTS / "split_indices.npz")
    tr, va, unused, te = [split[k] for k in ["train", "validation", "unused", "test"]]
    assert len(tr) == config["train_per_class"]*10 and len(va) == config["val_per_class"]*10 and len(te) == 10000
    assert len(set(tr) & set(va)) == 0
    assert np.array_equal(np.sort(np.concatenate([tr, va, unused])), np.arange(50000))
    train = CIFAR10(str(ROOT / "data"), train=True, download=False)
    test = CIFAR10(str(ROOT / "data"), train=False, download=False)
    for ids, n in [(tr, config["train_per_class"]), (va, config["val_per_class"])]:
        assert np.array_equal(np.bincount(np.array(train.targets)[ids]), np.full(10, n))
    checks.append("Disjoint, class-balanced train/validation; official test indices; complete source accounting")
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    selected = json.loads((RESULTS / "selection.json").read_text())["selected"]
    assert len(metrics) == 6
    for run in metrics:
        name = run["run"]
        bundle = np.load(RESULTS / "predictions" / f"{name}.npz")
        p, y = bundle["probabilities"], bundle["labels"]
        assert p.shape == (10000, 10) and np.isfinite(p).all() and (p >= 0).all()
        assert np.array_equal(y, test.targets)
        np.testing.assert_allclose(p.sum(1), 1, atol=2e-6)
        rank = np.argsort(-p, axis=1)
        assert abs((rank[:, 0] == y).mean() - run["test_top1"]) < 1e-7
        assert abs((rank[:, :5] == y[:, None]).any(1).mean() - run["test_top5"]) < 1e-7
        auc = np.mean([roc_auc_score(y == c, p[:, c]) for c in range(10)])
        assert abs(auc-run["test_macro_auc"]) < 1e-12
        rows = read_csv(RESULTS / "predictions" / f"{name}.csv")
        assert len(rows) == 10000
        classes = json.loads((RESULTS / "classes.json").read_text())
        csv_probabilities = np.array([[float(row[f"p_{c}"]) for c in classes] for row in rows])
        np.testing.assert_array_equal(csv_probabilities, p)
        history = read_csv(RESULTS / "history" / f"{name}.csv")
        assert len(history) == run["epochs"]
        val = [float(r["val_top1"]) for r in history]
        assert np.argmax(val)+1 == run["best_epoch"]
        assert max(val) == run["val_top1"]
        checkpoint = torch.load(CHECKPOINTS / f"{name}.pt", weights_only=True, map_location="cpu")
        assert checkpoint["epoch"] == run["best_epoch"] and checkpoint["config"] == config
    checks.append("All six runs: 10,000 probability rows, normalized scores, recomputed Top-1/Top-5/Macro-AUC, CSV/NPZ equality")
    checks.append("Every checkpoint is the earliest maximum validation Top-1 epoch; complete training histories")
    for family in ["plain", "resnet"]:
        candidates = [r for r in metrics if r["family"] == family and not r["augmentation"]]
        baseline = max(candidates, key=lambda r: r["val_top1"])
        augmented = next(r for r in metrics if r["family"] == family and r["augmentation"])
        assert selected[family]["baseline"] == baseline["run"]
        assert baseline["lr"] == augmented["lr"] and baseline["epochs"] == augmented["epochs"]
        assert selected[family]["final"] == max([baseline, augmented], key=lambda r: r["val_top1"])["run"]
        model, _, name = load_selected(family)
        count = parameter_counts(model)
        row = next(r for r in metrics if r["run"] == name)
        assert all(count[k] == row[k] for k in count)
        images = torch.from_numpy(test.data[:16].copy()).permute(0, 3, 1, 2).float()/255
        size = 32 if family == "plain" else config["resnet_input_size"]
        with torch.no_grad():
            actual = model(preprocess(images, size)).softmax(1).numpy()
        saved = np.load(RESULTS / "predictions" / f"{name}.npz")["probabilities"][:16]
        np.testing.assert_allclose(actual, saved, atol=2e-5, rtol=2e-4)
    checks.append("Validation-only family selection, controlled augmentation settings, correct total/trainable parameter counts")
    checks.append("Reloaded end-to-end models reproduce saved probabilities, including cached ResNet features")
    cams = np.load(RESULTS / "gradcam_maps.npz")
    assert cams["normalized_maps"].shape == (4, 32, 32)
    assert cams["raw_maps"].shape == (4, 8, 8)
    assert np.isfinite(cams["normalized_maps"]).all()
    assert (cams["normalized_maps"].max((1, 2)) > 0).all()
    kernels = np.load(RESULTS / "kernel_weights.npy")
    plain, _, _ = load_selected("plain")
    np.testing.assert_array_equal(kernels, plain.features[0].weight.detach().numpy()[:2])
    checks.append("Finite, nonzero Grad-CAM maps; two kernels exactly match the trained checkpoint")
    save_json(RESULTS / "verification.json", {"status": "passed", "checks": checks, "experiments": len(metrics),
              "prediction_rows": len(metrics)*10000})
    print(json.dumps({"status": "passed", "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
