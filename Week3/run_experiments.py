"""Reproducible CIFAR-10 experiments / 可重現的完整 CNN 實驗。

Run from this directory: python run_experiments.py
Test inference starts only AFTER validation-based selection is saved.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import sklearn
import torch
import torchvision
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

from models import PlainCNN, make_resnet, parameter_counts

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
CACHE = ROOT / "cache"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(config):
    train = CIFAR10(str(ROOT / "data"), train=True, download=True)
    test = CIFAR10(str(ROOT / "data"), train=False, download=True)
    targets = np.asarray(train.targets)
    rng = np.random.default_rng(config["seed"])
    train_ids, val_ids, unused_ids = [], [], []
    for cls in range(10):
        ids = rng.permutation(np.flatnonzero(targets == cls))
        n, v = config["train_per_class"], config["val_per_class"]
        assert n > 0 and v > 0 and n + v <= len(ids)
        train_ids.extend(ids[:n])
        val_ids.extend(ids[n:n+v])
        unused_ids.extend(ids[n+v:])
    train_ids, val_ids = np.sort(train_ids), np.sort(val_ids)
    assert not set(train_ids) & set(val_ids)
    assert len(train_ids) + len(val_ids) + len(unused_ids) == 50000
    np.savez_compressed(RESULTS / "split_indices.npz", train=train_ids, validation=val_ids,
                        unused=np.sort(unused_ids), test=np.arange(10000))
    split_rows = []
    for name, ids in [("train", train_ids), ("validation", val_ids), ("unused", sorted(unused_ids))]:
        split_rows.extend({"source": "official_train", "index": int(i), "split": name,
                           "label": int(targets[i])} for i in ids)
    split_rows.extend({"source": "official_test", "index": i, "split": "test", "label": int(y)}
                      for i, y in enumerate(test.targets))
    write_csv(RESULTS / "split_manifest.csv", split_rows)
    def tensor(images):
        return torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float().div_(255)
    return {"train_x": tensor(train.data[train_ids]), "train_y": torch.tensor(targets[train_ids]),
            "val_x": tensor(train.data[val_ids]), "val_y": torch.tensor(targets[val_ids]),
            "test_x": tensor(test.data), "test_y": torch.tensor(test.targets),
            "classes": train.classes}


def augment_view(images, seed):
    """Per-image reflection-padded random crop and horizontal flip.

    Two independently generated views are reused in alternating epochs;
    every epoch still has exactly one view per training image.
    """
    rng = torch.Generator().manual_seed(seed)
    shifts = torch.randint(0, 9, (len(images), 2), generator=rng)
    flips = torch.rand(len(images), generator=rng) < 0.5
    padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
    result = torch.empty_like(images)
    for i, (row, col) in enumerate(shifts.tolist()):
        crop = padded[i, :, row:row+32, col:col+32]
        result[i] = crop.flip(-1) if flips[i] else crop
    return result


def preprocess(x, size=32):
    if size != 32:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return (x - MEAN) / STD


def accuracy(logits, y):
    top = logits.topk(5, dim=1).indices
    return {"top1": float((top[:, 0] == y).float().mean()),
            "top5": float((top == y[:, None]).any(1).float().mean())}


@torch.no_grad()
def predict(model, x, batch_size):
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(x[i:i+batch_size].to(device)).cpu() for i in range(0, len(x), batch_size)])


def extract_features(backbone, x, name, config):
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"{name}_{fingerprint}.pt"
    if path.exists():
        features = torch.load(path, weights_only=True)
        assert features.shape == (len(x), 512)
        return features
    backbone.eval()
    parts = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(x), config["batch_size"]):
            images = preprocess(x[i:i+config["batch_size"]], config["resnet_input_size"])
            parts.append(backbone(images.to(DEVICE)).cpu())
            if i % (config["batch_size"] * 20) == 0:
                print(f"Features {name}: {i}/{len(x)}", flush=True)
    features = torch.cat(parts).clone()
    torch.save(features, path)
    print(f"Features {name}: complete in {time.perf_counter()-start:.1f}s", flush=True)
    return features


def train_one(family, lr, augmented, views, val_x, y, val_y, config, counts):
    name = f"{family}_lr{lr:g}_{'aug' if augmented else 'base'}"
    path = CHECKPOINTS / f"{name}.pt"
    metadata_path = RESULTS / "runs" / f"{name}.json"
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["config"] == config:
            print(f"Reuse completed run: {name}", flush=True)
            return metadata
    seed_all(config["seed"])
    model = PlainCNN() if family == "plain" else nn.Linear(512, 10)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
    epochs = config[f"{family}_epochs"]
    sampler = torch.Generator().manual_seed(config["seed"] + 100)
    best_score, best_epoch, history = -1, None, []
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        x = views[(epoch-1) % len(views)]
        order = torch.randperm(len(y), generator=sampler)
        total_loss, correct, seen = 0.0, 0, 0
        for ids in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[ids].to(DEVICE))
            labels = y[ids].to(DEVICE)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(ids)
        val_logits = predict(model, val_x, config["batch_size"])
        scores = accuracy(val_logits, val_y)
        row = {"epoch": epoch, "train_loss": total_loss/seen, "train_top1": correct/seen,
               "val_loss": float(F.cross_entropy(val_logits, val_y)),
               "val_top1": scores["top1"], "val_top5": scores["top5"]}
        history.append(row)
        if scores["top1"] > best_score:
            best_score, best_epoch = scores["top1"], epoch
            torch.save({"state_dict": model.state_dict(), "family": family, "run": name,
                        "epoch": epoch, "config": config}, path)
        write_csv(RESULTS / "history" / f"{name}.csv", history)
        print(f"{name} epoch {epoch:02d}/{epochs}: train={correct/seen:.4f} val={scores['top1']:.4f}", flush=True)
    metadata = {"run": name, "family": family, "lr": lr, "augmentation": augmented,
                "epochs": epochs, "best_epoch": best_epoch, "val_top1": best_score,
                "val_top5": history[best_epoch-1]["val_top5"],
                "training_seconds": time.perf_counter()-start, **counts, "config": config}
    save_json(metadata_path, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiment_config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for directory in [RESULTS, CHECKPOINTS, CACHE, RESULTS / "predictions", RESULTS / "per_class"]:
        directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    print(f"Device: {DEVICE}; GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    seed_all(config["seed"])
    started = datetime.now(timezone.utc).isoformat()
    data = load_data(config)
    save_json(RESULTS / "classes.json", data["classes"])
    views = [augment_view(data["train_x"], config["seed"]+i+1)
             for i in range(config["augmentation_views"])]
    runs = []
    selected = {}
    feature_seconds = {}
    for family in ["plain", "resnet"]:
        if family == "plain":
            counts = parameter_counts(PlainCNN())
            train_x = preprocess(data["train_x"])
            train_views = [preprocess(x) for x in views]
            val_x = preprocess(data["val_x"])
        else:
            backbone = make_resnet().to(DEVICE)
            counts = parameter_counts(backbone)
            backbone.fc = nn.Identity()
            for p in backbone.parameters():
                p.requires_grad_(False)
            tick = time.perf_counter()
            train_x = extract_features(backbone, data["train_x"], "train", config)
            val_x = extract_features(backbone, data["val_x"], "validation", config)
            train_views = [extract_features(backbone, x, f"train_aug{i}", config) for i, x in enumerate(views)]
            feature_seconds[family] = time.perf_counter()-tick
        candidates = [train_one(family, lr, False, [train_x], val_x, data["train_y"], data["val_y"], config, counts)
                      for lr in config[f"{family}_learning_rates"]]
        # Deterministic tie: first configured learning rate, then earliest epoch.
        baseline = max(candidates, key=lambda row: row["val_top1"])
        augmented = train_one(family, baseline["lr"], True, train_views, val_x, data["train_y"], data["val_y"], config, counts)
        runs.extend(candidates + [augmented])
        selected[family] = {"baseline": baseline["run"], "augmented": augmented["run"],
                            "final": max([baseline, augmented], key=lambda row: row["val_top1"])["run"]}
    # Lock ALL decisions before any test inference.
    save_json(RESULTS / "selection.json", {"criterion": "validation Top-1 only; first candidate wins ties",
              "selected": selected, "selected_at_utc": datetime.now(timezone.utc).isoformat()})
    test_features = extract_features(backbone, data["test_x"], "test", config)
    summary = []
    y = data["test_y"]
    for run in runs:
        family, name = run["family"], run["run"]
        model = PlainCNN() if family == "plain" else nn.Linear(512, 10)
        model.to(DEVICE)
        checkpoint = torch.load(CHECKPOINTS / f"{name}.pt", weights_only=True, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        x = preprocess(data["test_x"]) if family == "plain" else test_features
        logits = predict(model, x, config["batch_size"])
        probabilities = logits.softmax(1).numpy()
        scores = accuracy(logits, y)
        aucs = [roc_auc_score((y.numpy() == c).astype(int), probabilities[:, c]) for c in range(10)]
        np.savez_compressed(RESULTS / "predictions" / f"{name}.npz", probabilities=probabilities, labels=y.numpy())
        top5 = np.argsort(-probabilities, axis=1)[:, :5]
        rows = []
        for i, label in enumerate(y.tolist()):
            rows.append({"test_index": i, "true_class": data["classes"][label],
                         "predicted_class": data["classes"][top5[i, 0]],
                         "correct_top1": int(top5[i, 0] == label), "correct_top5": int(label in top5[i]),
                         "top5_classes": "|".join(data["classes"][j] for j in top5[i]),
                         **{f"p_{cls}": float(probabilities[i, j]) for j, cls in enumerate(data["classes"])}})
        write_csv(RESULTS / "predictions" / f"{name}.csv", rows)
        write_csv(RESULTS / "per_class" / f"{name}.csv", [
            {"class": cls, "test_support": int((y == c).sum()), "auc_ovr": aucs[c],
             "recall": float((top5[y.numpy() == c, 0] == c).mean())}
            for c, cls in enumerate(data["classes"])])
        cm = confusion_matrix(y.numpy(), top5[:, 0])
        np.savetxt(RESULTS / "per_class" / f"{name}_confusion.csv", cm, delimiter=",", fmt="%d",
                   header=",".join(data["classes"]), comments="")
        row = {k: v for k, v in run.items() if k != "config"}
        row.update({"test_top1": scores["top1"], "test_top5": scores["top5"], "test_macro_auc": float(np.mean(aucs)),
                    "selected_final": name == selected[family]["final"]})
        summary.append(row)
        print(f"TEST {name}: {scores}, Macro-AUC={np.mean(aucs):.6f}", flush=True)
    write_csv(RESULTS / "metrics.csv", summary)
    save_json(RESULTS / "metrics.json", summary)
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    save_json(RESULTS / "environment.json", {
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "scikit_learn": sklearn.__version__, "platform": platform.platform(),
        "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "threads": torch.get_num_threads(), "deterministic_algorithms": True,
        "feature_extraction_seconds": feature_seconds, "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(), "config": config})


if __name__ == "__main__":
    main()
