# Official test set (624 images)

| Model | Threshold | Accuracy | Sensitivity | Specificity | F1 | AUC | ECE | Notes |
|---|---|---|---|---|---|---|---|---|
| Review 1 baseline (MobileNetV2) | 0.500 | 84.8% | 99.0% | 61.1% | 89.0% | 0.966 | 0.140 | 2.4 M params |
| RAD-DINO linear probe [10] | 0.885 | 94.1% | 98.5% | 86.8% | 95.4% | 0.988 | 0.100 | 86.6 M params, 518 px |
| RAD-DINO + MLP head (teacher) | 0.998 | 92.9% | 97.9% | 84.6% | 94.6% | 0.976 | 0.128 | 86.6 M params, 518 px |
| Student without KD (ablation) | 0.651 | 91.5% | 97.9% | 80.8% | 93.5% | 0.976 | 0.095 | 2.4 M params |
| Student, uncalibrated @0.5 | 0.500 | 87.0% | 99.2% | 66.7% | 90.5% | 0.971 | 0.126 | 2.4 M params |
| **TrustEdge student (all cases)** | 0.674 | 89.9% | 99.2% | 74.4% | 92.5% | 0.970 | 0.120 | 2.4 M params |
| TrustEdge, auto-decided cases only | 0.674 | 93.9% | 99.7% | 76.2% | – | – | – | 135 of 624 referred |
| Edge INT8 TFLite + numpy head | 0.674 | 88.6% | 98.7% | 71.8% | 91.6% | 0.971 | 0.129 | 2.7 MB backbone |
| Federated student (5 sites) | 0.603 | 88.8% | 97.9% | 73.5% | 91.6% | 0.965 | 0.113 | no DP |
