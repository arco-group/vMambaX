<div align="center">
<h1>Context-Gated Cross-Modal Perception for PET-CT Lung Tumor Segmentation</h1>
<p>PyTorch Lightning implementation of CIPA on the <a href="https://arxiv.org/abs/2503.17261">PCLT20K</a> dataset</p>
</div>

## 👋 Overview
- Training and inference pipeline for cross-modal lung tumor segmentation with Mamba-based fusion.
- Tailored to the <a href="https://arxiv.org/abs/2503.17261">PCLT20K</a> benchmark with support for distributed training, mixed precision, and Weights & Biases logging.
- Modular design: VMamba encoder, context-gated decoder, dedicated dataloaders for PET-CT, and Lightning utilities.
- Ready-to-run scripts for SLURM clusters (`train.bash`) and offline inference (`pred.py`).

## 🗺️ Repository layout
- `train.py`, `train_v2.py`: Lightning entry points (local or SLURM launches).
- `pred.py`: evaluation over held-out splits with Dice/IoU/Accuracy/HD95.
- `models/`, `train_utils/`, `utils/`: backbone definitions, loss functions, PCLT20K dataloaders, helpers.
- `requirements.txt`: Python dependencies (Lightning, MedPy, optional WandB, etc.).

## ⚙️ Environment setup
1. Create a Python 3.10 environment.
   ```bash
   conda create -n cipa python=3.10
   conda activate cipa
   ```
2. Install PyTorch 2.1.2 (or any CUDA build matching your hardware).
   ```bash
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install project dependencies.
   ```bash
   pip install -r requirements.txt
   ```
4. Build the `selective_scan` CUDA operator.
   ```bash
   cd models/encoders/selective_scan
   pip install .
   cd ../../..
   ```

## 📦 PCLT20K dataset
PCLT20K contains 21,930 PET-CT pairs with expert lung tumor annotations collected from 605 patients.

### Requesting access
- Email <mailto:jiemei@hnu.edu.cn> with the subject `PCLT20K Access`.
- Include: full name, affiliation with institutional email domain, academic role, intended (non-commercial) usage.
- Refer to the paper [Multi-Modal Interactive Perception Network with Mamba for Lung Tumor Segmentation in PET-CT Images](https://arxiv.org/abs/2503.17261) for the full dataset description.

### Preparing the data
1. Place or symlink the dataset under `data/PCLT20K`.
   ```bash
   mkdir -p data
   ln -s /path/to/PCLT20K data/PCLT20K
   ```
2. Ensure `train.txt`, `test.txt`, and optionally `val.txt` are located in the dataset root.
3. When using WandB, export `WANDB_API_KEY` before starting training.

### Required directory structure
```
PCLT20K/
├── 0001/
│   ├── 0001_CT.png
│   ├── 0001_PET.png
│   └── 0001_mask.png
├── 0002/
│   ├── 0002_CT.png
│   ├── 0002_PET.png
│   └── 0002_mask.png
├── ...
├── train.txt
├── val.txt        # optional, otherwise split is derived automatically
└── test.txt
```

## 🔄 Custom datasets
To train on your own PET-CT dataset, keep the same structure:
```
<DatasetName>/
├── <sample_id>/
│   ├── <sample_id>_CT.<ext>
│   ├── <sample_id>_PET.<ext>
│   └── <sample_id>_mask.<ext>
├── train.txt
├── val.txt
└── test.txt
```
Each split file lists one `<sample_id>` per line (for example `0001_1234`). The scripts locate the PET/CT/mask files by appending `_PET`, `_CT`, and `_mask` to each identifier.

## 🚀 Training
### Single-GPU example
```bash
python train.py \
  --img_dir data/PCLT20K \
  --split_train_val_test data/PCLT20K \
  --batch_size 4 \
  --epochs 50 \
  --lr 6e-5 \
  --fusion-type context_gate
```

### Multi-GPU / Lightning DDP
- Configure `CUDA_VISIBLE_DEVICES`, `--devices`, and `--nodes` as needed.
- Enable logging with `--wandb --wandb_project cipa --wandb_run_name <run-name>`.
- Disable decoder cross-fusion via `--decoder-disable-cross-fusion`.
- Switch to the Channel Rectification Module by setting `--fusion-type channel_rectify`.

### SLURM example
Edit `train.bash` with your SLURM account, paths, and resource requirements, then submit:
```bash
sbatch train.bash
```

Lightning checkpoints are stored in `lightning_logs/`, while exported `.pth` weights reside in `checkpoints/`.

## 🧪 Inference and evaluation
1. Download or select a checkpoint (`.ckpt` from Lightning or `.pth` weights).
2. Run:
   ```bash
   python pred.py \
     --img_dir data/PCLT20K \
     --split_train_val_test data/PCLT20K \
     --checkpoint path/to/best.ckpt \
     --device cuda
   ```
3. Metrics reported: IoU, Dice, Accuracy, and HD95. Results are written to `results/`.

### Pretrained weights
- [CIPA.pth – Google Drive](https://drive.google.com/file/d/1x525pjCi4RM51Kv_zbuW7OLx7NPBLRa8/view?usp=sharing)
- [CIPA.pth – Baidu Yun (pwd: CIPA)](https://pan.baidu.com/s/14MfEaSvc-4QFOIWR7w93Tw)

## 🙏 Acknowledgements
- This project builds upon the original CIPA repository (Multi-Modal Interactive Perception Network with Mamba for Lung Tumor Segmentation in PET-CT Images).
- We gratefully acknowledge the open-source contributions of [VMamba](https://github.com/MzeroMiko/VMamba) and [Sigma](https://github.com/zifuwan/Sigma).
- The PCLT20K dataset is provided by Jie Mei et al. (CVPR 2025); we thank the authors for making their work available.
