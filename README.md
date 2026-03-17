# BEV LiDAR–Camera Fusion for 3D Object Detection

A multi-modal 3D object detection system that fuses **LiDAR point clouds** (projected to Bird's-Eye View) with **RGB camera images** using a **ResNet backbone** and **cross-attention fusion**. Trained and evaluated on the [KITTI 3D Object Detection benchmark](https://www.kaggle.com/datasets/garymk/kitti-3d-object-detection-dataset/).

---

## Architecture Overview

```
LiDAR (.bin)                          Camera (.png)
     │                                      │
     ▼                                      ▼
BEV Projection                       Resize + Normalize
(5-channel, 400×352)                 (3×384×1280)
     │                                      │
     ▼                                      ▼
BEV Encoder                          ResNet Encoder
(4-block CNN, stride 8)              (ResNet18/34, layer3 → 256ch)
     │                                      │
     └──────────────┬───────────────────────┘
                    ▼
        Cross-Attention Fusion
        (BEV queries, Image K/V)
        Pool → Attend → Upsample
                    │
                    ▼
           Detection Head
        (Cls logits + 8-DOF reg)
                    │
                    ▼
        NMS → 3D Bounding Boxes
```

### BEV Feature Map (5 channels)
| Channel | Description |
|---------|-------------|
| 0 | Max height (normalised) |
| 1 | Mean height (normalised) |
| 2 | Point density (log-normalised) |
| 3 | Max LiDAR intensity |
| 4 | Mean LiDAR intensity |

---

## Dataset

**KITTI 3D Object Detection** — 7,481 training frames, 7,518 test frames.

Each frame provides:
- `velodyne/` — LiDAR point cloud (N × 4: x, y, z, intensity)
- `image_2/` — Left RGB camera image
- `calib/` — Camera–LiDAR calibration matrices
- `label_2/` — 3D bounding box annotations

**Classes detected:** Car · Pedestrian · Cyclist

---

## Project Structure

```
├── bev_lidar_camera_fusion_kitti.ipynb   # Full Kaggle notebook
├── README.md                              # This file
└── best_model.pth                         # Saved after training (generated)
```

### Notebook Cells

| Step | Description |
|------|-------------|
| 0 | Install dependencies (`open3d`) |
| 1 | Dataset paths & configuration (`CFG` dict) |
| 2 | KITTI calibration & label parsers |
| 3 | LiDAR → BEV projection |
| 4 | `KITTIFusionDataset` + DataLoader |
| 5 | Model architecture (BEVEncoder, ImageEncoder, CrossAttentionFusion, DetectionHead) |
| 6 | Loss functions (Focal + Smooth-L1) |
| 7 | Training loop (AdamW + Cosine LR + AMP) |
| 8 | Training curve plots |
| 9 | Post-processing & NMS |
| 10 | BEV prediction visualisation |
| 11 | Evaluation (per-class AP + mAP) |
| 12 | ONNX export (optional) |

---

## Key Configuration

```python
CFG = {
    'bev_x_range':    (0.0, 70.4),     # Forward range (metres) — KITTI standard
    'bev_y_range':    (-40.0, 40.0),   # Left/right range
    'bev_resolution':  0.2,            # Metres per pixel
    'bev_channels':    5,
    'backbone':       'resnet18',      # resnet18 | resnet34 | resnet50
    'feat_channels':   256,
    'num_heads':       4,
    'num_classes':     3,
    'batch_size':      1,              # P100 constraint
    'num_epochs':      30,
    'lr':              1e-4,
}
```

---

## Quick Start (Kaggle)

1. Add dataset: `garymk/kitti-3d-object-detection-dataset`
2. Enable GPU accelerator (P100)
3. Run all cells top to bottom
4. Best checkpoint auto-saved to `/kaggle/working/best_model.pth`

---

## Known Issues & Fixes Applied

| Issue | Fix |
|-------|-----|
| OOM in cross-attention (P100) | Pool BEV/Image features to 25×25 before attention |
| GT boxes outside BEV range | Changed `bev_x_range` to `(0, 70.4)` — KITTI standard |
| Dying logits (min ≈ −101) | Balanced focal loss with separate fg/bg normalisation |

---

## Potential Improvements

- **FPN** on top of BEV encoder for multi-scale detection
- **CenterPoint-style heatmap head** instead of per-cell classification
- **Copy-paste augmentation** to increase rare class (Pedestrian, Cyclist) frequency
- **Multi-scale cross-attention** (attend at stride 8 and stride 16 simultaneously)
- **Larger backbone** (ResNet50 or EfficientNet-B4)
- **PointPillars-style voxelisation** for richer BEV encoding

---

## References

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds (Lang et al., 2019)](https://arxiv.org/abs/1812.05784)
- [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation (Liu et al., 2022)](https://arxiv.org/abs/2205.13542)
- [Focal Loss for Dense Object Detection (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
