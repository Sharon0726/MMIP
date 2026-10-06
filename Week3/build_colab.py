"""Package source into a self-contained Google Colab training notebook."""
import base64
import io
import zipfile
from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parent
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in ["models.py", "run_experiments.py", "analyze_results.py", "verify_results.py", "prepare_data.py",
                 "predict.py", "experiment_config.json"]:
        archive.writestr(name, (root / name).read_bytes())
payload = base64.b64encode(buffer.getvalue()).decode("ascii")
nb = nbf.v4.new_notebook(cells=[
    nbf.v4.new_markdown_cell("# MMIP Week 3：GPU 訓練 / Colab Runner\n\n**蘇芯儀 / Sharon0726**\n\n使用完整 CIFAR-10：45,000 train、5,000 validation、官方 10,000 test。請選擇 GPU runtime，然後依序執行所有 cells。程式碼已內嵌，不需先上傳至 GitHub。"),
    nbf.v4.new_code_cell('''# 子目標：確認 GPU，解壓作業程式至獨立工作目錄。
import base64, io, os, zipfile, pathlib, subprocess, sys, torch
assert torch.cuda.is_available(), "請先在 Runtime > Change runtime type 選擇 GPU。"
print("GPU:", torch.cuda.get_device_name(0))
work = pathlib.Path("/content/mmip_week3")
work.mkdir(exist_ok=True)
payload = "''' + payload + '''"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
    archive.extractall(work)
os.chdir(work)
print("Workspace:", work)
'''),
    nbf.v4.new_code_cell('''# 主目標：訓練六組實驗，逐行顯示並儲存執行紀錄。
for script in ["prepare_data.py", "run_experiments.py"]:
    process = subprocess.Popen([sys.executable, "-u", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open("training.log", "a") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    assert process.wait() == 0, "Execution failed; inspect training.log"
'''),
    nbf.v4.new_code_cell('''# 子目標：產生 ROC、Kernel、Grad-CAM，並獨立重算指標。
import hashlib, json, importlib.metadata
source_names = ["models.py", "run_experiments.py", "analyze_results.py", "verify_results.py", "predict.py", "prepare_data.py", "experiment_config.json"]
pathlib.Path("results/source_sha256.json").write_text(json.dumps({name: hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in source_names}, indent=2))
pathlib.Path("results/package_versions.json").write_text(json.dumps({name: importlib.metadata.version(name) for name in ["torch", "torchvision", "numpy", "scikit-learn", "matplotlib", "Pillow"]}, indent=2))
for script in ["analyze_results.py", "verify_results.py"]:
    completed = subprocess.run([sys.executable, script], capture_output=True, text=True)
    print(completed.stdout)
    print(completed.stderr)
    completed.check_returncode()
'''),
    nbf.v4.new_code_cell('''# 子目標：匯出結果與權重；不包含原始資料及特徵快取。
from google.colab import files
archive_path = pathlib.Path("/content/MMIP_Week3_results.zip")
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for folder in ["results", "checkpoints"]:
        for path in sorted(pathlib.Path(folder).rglob("*")):
            if path.is_file():
                archive.write(path, path.as_posix())
print("Export:", archive_path, archive_path.stat().st_size, "bytes")
files.download(str(archive_path))
''')], metadata={"colab": {"name": "MMIP_Week3_Colab.ipynb"}, "accelerator": "GPU",
                "kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, root / "MMIP_Week3_Colab.ipynb")
print("Created MMIP_Week3_Colab.ipynb")
