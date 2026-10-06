"""Build and EXECUTE three reader-friendly notebooks from real results."""
import json
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = '''# 子目標：定位作業目錄，讀取已完成的真實實驗結果。
from pathlib import Path
import csv, json
import numpy as np
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
if not (ROOT / "results" / "metrics.json").exists():
    raise FileNotFoundError("請在 Week3 作業目錄開啟 Notebook。")
R = ROOT / "results"
metrics = json.loads((R / "metrics.json").read_text())
selected = json.loads((R / "selection.json").read_text())["selected"]
classes = json.loads((R / "classes.json").read_text())
def table(rows, columns):
    text = "| " + " | ".join(columns) + " |\\n| " + " | ".join(["---"]*len(columns)) + " |\\n"
    for row in rows:
        text += "| " + " | ".join(str(row[c]) for c in columns) + " |\\n"
    display(Markdown(text))
def figure(name):
    display(Image(filename=str(R / "figures" / name)))
'''


def make(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]}})
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, timeout=180, km=manager, resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    assert all(not any(o.output_type == "error" for o in c.get("outputs", [])) for c in notebook.cells)
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / name)
    print(f"Executed and validated: {name}", flush=True)


def main():
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    q1 = report.split("## Quiz 1", 1)[1].split("## Quiz 2", 1)[0]
    q2 = report.split("## Quiz 2", 1)[1].split("## Quiz 3", 1)[0]
    q3 = report.split("## Quiz 3", 1)[1].split("## 重現與限制", 1)[0]
    # Strip linked images from prose: code cells below embed the actual image bytes.
    def prose(section):
        return "\n".join(line for line in section.splitlines() if not line.startswith("!["))
    make("Quiz1_Dataset.ipynb", [
        md("# Quiz 1：準備資料集 / Dataset\n\n**蘇芯儀 / Sharon0726**\n\n本 Notebook 附已執行的結果；完整訓練程式見 `run_experiments.py`。"),
        code(SETUP), md("## Quiz 1" + prose(q1)),
        code('''# 子目標：驗證來源索引完整涵蓋，且 train／validation 沒有交集。
split = np.load(R / "split_indices.npz")
assert not set(split["train"]) & set(split["validation"])
assert np.array_equal(np.sort(np.concatenate([split["train"], split["validation"], split["unused"]])), np.arange(50000))
with (R / "split_manifest.csv").open(encoding="utf-8") as f:
    manifest = list(csv.DictReader(f))
rows = [{"class": cls, **{s: sum(row["split"] == s and int(row["label"]) == c for row in manifest)
         for s in ["train", "validation", "test", "unused"]}} for c, cls in enumerate(classes)]
table(rows, ["class", "train", "validation", "test", "unused"])
print("Split integrity: passed; official train and test use separate index namespaces.")'''),
        code('# 子目標：展示每類的訓練影像。\nfigure("dataset_samples.png")'),
        md("### 執行整份實驗\n\n於本目錄執行 `python run_experiments.py`；資料將由 TorchVision 下載並核對 checksum。原始影像不存入 Git。")])
    make("Quiz2_CNN_Training.ipynb", [
        md("# Quiz 2：訓練 CNN / Training and Evaluation\n\n**蘇芯儀 / Sharon0726**\n\n此 Notebook 讀取實際完成的 6 組實驗，不以展示數值取代訓練。"), code(SETUP),
        md("## Quiz 2" + prose(q2)),
        code('''# 子目標：呈現可執行的自訂架構及全部實驗結果。
from models import PlainCNN
print(PlainCNN())
rows = [{"run": r["run"], "best_epoch": r["best_epoch"],
         "val_Top1": f"{r['val_top1']:.2%}", "test_Top1": f"{r['test_top1']:.2%}",
         "test_Top5": f"{r['test_top5']:.2%}", "Macro_AUC": f"{r['test_macro_auc']:.4f}",
         "total_params": r["parameters_total"], "trainable_params": r["parameters_trainable"]}
        for r in metrics]
table(rows, list(rows[0]))'''),
        code('# 子目標：觀察收斂與驗證集選模。\nfigure("learning_curves.png")\nprint(json.dumps(selected, indent=2))'),
        code('''# 子目標：用所有 test scores 獨立重算指標，並展示前五筆預測。
from sklearn.metrics import roc_auc_score
for family in ["plain", "resnet"]:
    name = selected[family]["final"]
    arrays = np.load(R / "predictions" / f"{name}.npz")
    p, y = arrays["probabilities"], arrays["labels"]
    order = np.argsort(-p, axis=1)
    print(name, "samples:", len(y))
    print("Top-1:", (order[:, 0] == y).mean(), "Top-5:", (order[:, :5] == y[:, None]).any(1).mean())
    print("Macro-AUC:", np.mean([roc_auc_score(y == c, p[:, c]) for c in range(10)]))
    table([{"test_index": i, "truth": classes[y[i]], "prediction": classes[order[i, 0]],
            "top5": ", ".join(classes[j] for j in order[i, :5])} for i in range(5)],
          ["test_index", "truth", "prediction", "top5"])
    figure(f"roc_{name}.png")'''),
        code('# 子目標：比較類別錯誤分布、準確率與參數量。\nfigure("confusion_matrices.png")\nfigure("model_comparison.png")')])
    make("Quiz3_Generalization_XAI.ipynb", [
        md("# Quiz 3：泛化與可解釋性 / Generalization and XAI\n\n**蘇芯儀 / Sharon0726**"), code(SETUP),
        md("## Quiz 3" + prose(q3)),
        code('''# 子目標：固定其他設定，重算資料擴增的實測差異。
rows = []
for family in ["plain", "resnet"]:
    base = next(r for r in metrics if r["run"] == selected[family]["baseline"])
    aug = next(r for r in metrics if r["run"] == selected[family]["augmented"])
    assert base["lr"] == aug["lr"] and base["epochs"] == aug["epochs"]
    rows.append({"model": family, "baseline_Top1": f"{base['test_top1']:.2%}",
                 "aug_Top1": f"{aug['test_top1']:.2%}",
                 "delta_pp": round(100*(aug["test_top1"]-base["test_top1"]), 2)})
table(rows, list(rows[0]))
figure("augmentation_examples.png")
figure("model_comparison.png")'''),
        code('''# 子目標：展示兩個實際訓練後的卷積權重與控制探針。
kernels = np.load(R / "kernel_weights.npy")
assert kernels.shape == (2, 3, 3, 3)
print("Kernel tensor shape [output, input, height, width]:", kernels.shape)
figure("two_kernels.png")
figure("kernel_probes.png")
with (R / "kernel_probe_responses.csv").open() as f:
    probes = list(csv.DictReader(f))
table(probes, list(probes[0]))'''),
        code('''# 子目標：展示梯度計算方法、熱圖及遮蔽驗證。
import inspect
from analyze_results import grad_cam
print(inspect.getsource(grad_cam))
figure("gradcam_examples.png")
with (R / "xai_cases.csv").open() as f:
    cases = list(csv.DictReader(f))
table(cases, ["test_index", "true_class", "target_predicted_class", "probability", "top_quarter_drop", "bottom_quarter_drop"])
print((R / "verification.json").read_text(encoding="utf-8"))'''),
        md("### 解讀範圍\n\nGrad-CAM 是特定模型、特定類別的局部梯度解釋。熱圖不等同物件分割，也不能證明模型理解語意；單一 seed 的擴增改善不能當成普遍結論。")])


if __name__ == "__main__":
    main()
