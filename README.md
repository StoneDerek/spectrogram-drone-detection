# Spectrogram Drone Detection

A binary audio classifier that detects whether a drone is present in an audio recording. The model uses log-mel spectrograms as input to a fine-tuned EfficientNet-B0 backbone, achieving near-perfect accuracy on both held-out test data and real-world recordings.

## How It Works

Raw audio is resampled to 22,050 Hz and sliced into overlapping 3-second windows. Each window is converted to a 128-bin log-mel spectrogram, which is fed into an EfficientNet-B0 backbone pretrained on ImageNet. The backbone's 1,280-dimensional feature vector passes through a dropout layer and a linear head that outputs a single probability score. A score above the threshold (default 0.5, recommended 0.988 for high-precision use) is classified as **drone detected**.

## Model Performance

The current checkpoint (`best_model_v3.pt`) was saved at epoch 25.

| Metric | Value |
|--------|-------|
| Validation loss | 0.0167 |
| Recall | 0.999 |
| Precision | 0.980 |
| F1 | 0.989 |
| AUC | 1.000 |

**Real-world inference results (out-of-distribution audio):**

| File | True Label | Result |
|------|-----------|--------|
| `new_recording_63.wav` | Drone | ✅ 99.5% of segments flagged |
| `youtube_yes_drone.wav` | Drone | ✅ 86.7% of segments flagged |
| `aeroplane.wav` | No drone | ✅ 0 segments flagged |
| `street.wav` | No drone | ✅ 0 segments flagged |
| `traffic.wav` | No drone | ✅ 0 segments flagged |
| `leafblower.wav` | No drone | ✅ 0 segments flagged (hardest case) |

**6/6 correct on real-world audio.**

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/StoneDerek/spectrogram-drone-detection.git
cd spectrogram-drone-detection
pip install -e .
```

For development dependencies (pytest, ruff):

```bash
pip install -e ".[dev]"
```

## Project Structure

```
spectrogram-drone-detection/
├── configs/            # Hydra configuration files
├── scripts/            # Training, evaluation, and inference scripts
├── src/
│   └── dronedetection/ # Core package
├── tests/              # Unit tests
├── model_summary.txt   # Detailed model and training documentation
└── pyproject.toml      # Project metadata and dependencies
```

## Dependencies

Key dependencies (full list in `pyproject.toml`):

- `torch >= 2.2`, `torchaudio >= 2.2`, `torchvision >= 0.17`
- `librosa >= 0.10` — audio feature extraction
- `hydra-core >= 1.3` — configuration management
- `wandb >= 0.16` — experiment tracking
- `onnx >= 1.15`, `onnxruntime >= 1.17` — model export and inference
- `scikit-learn >= 1.4` — metrics and data splitting

## Training Details

Training used a two-phase approach:
- **Phase 1 (epochs 1–5):** Backbone frozen, only the classification head trained
- **Phase 2 (epoch 6+):** Full model fine-tuned end-to-end

**Optimiser:** AdamW with backbone LR 1e-4, head LR 1e-3, weight decay 1e-4  
**Scheduler:** CosineAnnealingWarmRestarts with 3-epoch linear warmup  
**Batch size:** 64 | **Max epochs:** 50 | **Early stopping patience:** 7

Augmentations included random time shift, Gaussian noise, background noise mixing, and SpecAugment.

## Training Data

The model was trained on ~99,578 files across 7 datasets:

| Dataset | Files | Label |
|---------|-------|-------|
| FSD50K | 45,255 | No-drone |
| Kaggle balanced dataset | 19,732 | Mixed |
| DADS (Drone Audio Dataset) | 17,605 | Drone |
| Al-Emadi drone audio | 14,728 | Drone |
| ESC-50 | 2,000 | No-drone |
| DroneAudioSet | 168 | Drone |
| Zenodo Drone Detection Thesis | 90 | Drone |

Overall: 36.7% drone, 63.3% no-drone. Splits are performed at the recording level to prevent data leakage (70% train / 15% val / 15% test).

## Running Tests

```bash
pytest
```
