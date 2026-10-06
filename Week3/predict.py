"""Predict one RGB image with a validation-selected model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import PlainCNN, make_resnet
from run_experiments import ROOT, RESULTS, CHECKPOINTS, preprocess


def load_selected(family):
    selection = json.loads((RESULTS / "selection.json").read_text())["selected"]
    name = selection[family]["final"]
    state = torch.load(CHECKPOINTS / f"{name}.pt", map_location="cpu", weights_only=True)
    model = PlainCNN() if family == "plain" else make_resnet()
    if family == "plain":
        model.load_state_dict(state["state_dict"])
    else:
        model.fc.load_state_dict(state["state_dict"])
    return model.eval(), state["config"], name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["plain", "resnet"], required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    model, config, name = load_selected(args.model)
    torch.set_num_threads(config["threads"])
    with Image.open(args.image) as source:
        image = np.array(source.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR))
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float()/255
    size = 32 if args.model == "plain" else config["resnet_input_size"]
    with torch.no_grad():
        p = model(preprocess(tensor, size)).softmax(1)[0]
    classes = json.loads((RESULTS / "classes.json").read_text())
    scores, indices = p.topk(5)
    print(json.dumps({"model": name, "top5": [{"class": classes[i], "probability": float(s)}
          for s, i in zip(scores, indices)]}, indent=2))


if __name__ == "__main__":
    main()
