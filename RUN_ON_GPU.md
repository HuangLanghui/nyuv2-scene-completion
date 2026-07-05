# Full GPU Run — Step-by-Step (Windows PowerShell)

This guide trains the model **from scratch on a GPU** and then runs the entire pipeline
(evaluate → demo → figures → PLY → ablations → report). Every command was verified to work on this
repo; only the training step needs a GPU.

> **Estimated time:** setup ~5–10 min (downloading CUDA PyTorch), training ~20–40 min on an
> RTX 4090, the rest a few minutes.

---

## Part 0 — One-time setup

### 0.1 Open PowerShell in the project and activate the venv
```powershell
cd D:\nyuv2-scene-completion\nyuv2-scene-completion1
python -m venv .venv          # skip if .venv already exists
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 0.2 Replace the CPU PyTorch with a CUDA build
The repo currently has the CPU-only PyTorch, which will *not* use the GPU. The project does **not**
use `torchvision`/`torchaudio`, so install **only `torch`** (smaller download).

> **Disk-space warning (important on this machine):** the CUDA `torch` wheel is ~2.8 GB and pip
> unpacks it in the temp folder, which by default is on **C:** — and C: here is nearly full. The two
> commands below (a) purge pip's cache and (b) point pip's temp + cache to **D:** (which has room), so
> the big download never touches C:.

```powershell
# free C: and route pip's big files to D:
pip cache purge
New-Item -ItemType Directory -Force D:\pip-tmp | Out-Null
$env:TMP = "D:\pip-tmp"; $env:TEMP = "D:\pip-tmp"

# remove CPU build, install ONLY the CUDA torch (cu128; older drivers: pick from https://pytorch.org)
pip uninstall -y torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
```

### 0.3 Verify the GPU is visible  ← do not proceed until this prints True
```powershell
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```
Expected: a `+cu128` version, `cuda available: True`, and your GPU name.

### 0.4 Confirm the dataset is in place
```powershell
python -c "import os; p='data/nyu_depth_v2_labeled.mat'; print(p, 'OK' if os.path.exists(p) else 'MISSING', round(os.path.getsize(p)/1e9,2),'GB') if os.path.exists(p) else print('MISSING')"
```

---

## Part 1 — Back up the current results (so you can compare / restore)

Training overwrites `outputs/checkpoints/best.pt` and `outputs/metrics/`. Keep a copy first:
```powershell
Copy-Item outputs\checkpoints outputs\checkpoints_backup_pretrained -Recurse -Force
Copy-Item outputs\metrics     outputs\metrics_backup_pretrained     -Recurse -Force
```

---

## Part 2 — Train from scratch on the GPU
```powershell
python train.py --config configs\full_train.yaml
```
This writes:
- `outputs\checkpoints\best.pt` (best validation) and `last.pt`
- `outputs\metrics\train_history.json` and `outputs\metrics\splits.json`

Watch the epoch log: training loss should fall from ~0.4 toward ~0.17 and validation F1 rise past
~0.83. The split is fixed by `seed: 42`, so your test set is the same 146 scenes as before; your final
numbers should land near IoU ≈ 0.73 (small variation from random initialisation is normal).

> **Windows note:** if the DataLoader errors or stalls, set `num_workers: 0` under `training:` in
> `configs\full_train.yaml` and rerun. (GPU speed is unaffected for this small model.)

---

## Part 3 — Evaluate on the test set
```powershell
python evaluate.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test
type outputs\metrics\test_metrics.json
```
Expect IoU / Precision / Recall / F1 / Loss printed and saved.

## Part 4 — See the completion (the part that shows it works)
```powershell
python demo.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --num 6 --export-ply
python visualize.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test --num 12
```
- `outputs\demo\` — per-scene table + `input | prediction | target` triplets + TP/FP/FN error maps + `.ply`
- `outputs\visualizations\` — `completion_test_*.png`

Every demo row should show **pred vox > input vox** and a **positive recall gain**.

## Part 5 — The key evidence: model vs non-learned baselines
```powershell
python scripts\naive_baselines.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test
```
The trained model's IoU should clearly beat `copy_input`, and `dilate_*` should be worse — i.e. it
completes rather than copies.

## Part 6 — Advanced report figures
```powershell
python scripts\run_advanced_figures.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test --num 4
```
Writes training curves, metric bars, threshold sweep, galleries, and error maps to `outputs\figures\`.

## Part 7 — Export 3D PLY point clouds
```powershell
python scripts\export_occupancy_ply.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test --indices 0,20,40,60,80,100,120,145
```
Open the `.ply` files in `outputs\exports\` with CloudCompare / MeshLab / Blender / Open3D.

## Part 8 — Run all six ablation studies
```powershell
python scripts\run_custom_ablation_study.py --config configs\full_train.yaml --checkpoint outputs\checkpoints\best.pt --split test --indices 0,20,40,60,80,100,120
```
Writes `*_ablation.json` + `*_ablation.png` (threshold, missingness, partial, point-res, voxel-res,
noise, labels).

## Part 9 — Report
Nothing to build: the finished report is provided as
`report\3D_Scene_Occupancy_Completion_Report.pdf`. The figures it embeds are the ones the steps above
regenerate under `outputs\figures\`, so after a fresh run the report reflects your own numbers.

---

## Success checklist
- [ ] `torch.cuda.is_available()` printed **True**
- [ ] `train.py` finished 20 epochs and wrote `best.pt`
- [ ] `test_metrics.json` shows IoU ≈ 0.73
- [ ] every `demo.py` row has a positive recall gain
- [ ] `naive_baselines.py`: model IoU > copy-input IoU
- [ ] figures exist in `outputs\figures\`, PLYs in `outputs\exports\`
- [ ] `report\3D_Scene_Occupancy_Completion_Report.pdf` opens and reads correctly

## If something fails
- **`cuda available: False`** → the CPU wheel is still installed; redo Part 0.2, check `nvidia-smi`.
- **Out of memory** → lower `batch_size` (e.g. 2) under `training:` in the config.
- **DataLoader hang/error on Windows** → set `num_workers: 0` in the config.
- **Want the original pretrained result back** →
  `Copy-Item outputs\checkpoints_backup_pretrained\best.pt outputs\checkpoints\best.pt -Force`

## (Optional) Linux / autodl equivalent
Same steps; only the setup differs:
```bash
cd /root/autodl-tmp/nyuv2-scene-completion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python train.py --config configs/full_train.yaml
# then evaluate.py / demo.py / scripts/*.py exactly as above (use forward slashes)
```
