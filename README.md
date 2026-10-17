# Generalizable Deepfake Detection for Unseen AI Generators

## 1. Project Title

**Generalizable Deepfake Detection for Unseen AI Generators**

## 2. Project Overview

This research project develops and evaluates a PyTorch-based deepfake video detector
whose central evaluation criterion is **generalization to manipulation / generation
techniques that were never seen during training**.

Most published detectors report very high accuracy when the training and test data come
from the same generators. Their performance often drops sharply when they meet a new
generator. This project treats that drop as the primary thing to measure and reduce,
not as a side note.

> Status: initial project setup. No dataset has been selected or downloaded and no model
> has been trained yet.

## 3. Real-World Problem

Deepfake generators evolve quickly: face swapping, face reenactment, full-face
synthesis, diffusion-based video, and lip-sync models appear faster than labelled
detection datasets can be built. A detector deployed in practice (content moderation,
journalism, fraud prevention, identity verification) will almost always face fakes
produced by a generator it was not trained on. A detector that only recognises the
fingerprints of known generators gives a false sense of security.

## 4. Research Problem

Deepfake detectors trained with standard supervised learning tend to overfit to
**generator-specific artifacts** (particular blending boundaries, upsampling patterns,
colour mismatches, compression signatures) instead of learning cues shared across
manipulation methods. As a result:

- in-distribution performance is high, but
- cross-technique and cross-dataset performance is low and poorly calibrated.

The problem studied here is how to train and evaluate a detector so that its performance
on **unseen techniques** is high, and how to measure that honestly.

## 5. Main Research Question

> To what extent can a frame-based deepfake video detector trained on a set of known
> manipulation/generation techniques correctly detect videos produced by techniques that
> were completely excluded from training, and which training choices most improve this
> unseen-technique performance?

## 6. Research Hypothesis

**H1.** A standard ImageNet-pretrained CNN baseline will show a significant performance
drop (especially in ROC-AUC and recall on fakes) when evaluated on unseen techniques
compared with seen techniques.

**H2.** Training strategies that discourage reliance on generator-specific artifacts
(e.g. artifact-agnostic augmentation, synthetic blending/forgery simulation, or
frequency-aware features) will reduce this gap and improve unseen-technique ROC-AUC over
the baseline, without a large loss on seen techniques.

## 7. Project Objectives

1. Review candidate datasets and select one (or more) that provide multiple, clearly
   labelled manipulation techniques suitable for a technique-held-out protocol.
2. Build a reproducible preprocessing pipeline: frame sampling, face detection, face
   cropping, normalisation.
3. Define leakage-free splits by **technique** and by **source identity / source video**.
4. Implement and evaluate a baseline frame-level detector with video-level aggregation.
5. Implement at least one generalization-oriented improvement and compare it against the
   baseline on unseen techniques.
6. Run ablation studies and a structured error analysis.
7. Document results, limitations, and future work in a written report.

## 8. Why Unseen-Generator Generalization Matters

- **Realistic threat model:** attackers use the newest tools, not the ones in public
  training sets.
- **Misleading benchmarks:** random train/test splits mix the same generators (and often
  the same identities) into train and test, which inflates reported accuracy.
- **Calibration under shift:** a decision threshold tuned on seen generators can be badly
  wrong on unseen ones, even when ranking quality (AUC) is acceptable.
- **Cost of retraining:** a detector that needs new labelled data for every new
  generator does not scale.

For these reasons, **a random train/test split is not used as the main research
protocol** in this project. It may be reported only as an in-distribution reference.

## 9. Planned Pipeline

```
Video
  → Frame Extraction        (uniformly sample N frames per video)
  → Face Detection          (per frame)
  → Face Cropping           (bounding box + margin, fixed output size)
  → Preprocessing           (resize, normalise, augmentation at train time)
  → PyTorch Model           (frame-level real/fake classifier)
  → Real/Fake Prediction    (per-frame fake probability)
  → Video-Level Decision    (aggregate frame scores, e.g. mean, then threshold)
```

Implementation notes:

- Frame extraction and face cropping use OpenCV to avoid extra dependencies. The exact
  face detector will be chosen and compared once data is available; alternatives such as
  MTCNN or RetinaFace would add dependencies and will only be adopted if clearly needed.
- Preprocessed faces are cached to `data/faces/` so that training does not re-decode
  videos each epoch.
- Default settings (image size, frames per video, batch size, etc.) live in
  `src/config.py`.

## 10. Main Experiment

```
Train on known manipulation/generation techniques
  → Test on completely unseen techniques
```

Protocol principles:

- The set of techniques in the **test** split is disjoint from the set in the **train**
  split (enforced by `ExperimentConfig.validate()` in `src/config.py`).
- Real videos are split by **source identity / source video** so the same person or
  the same original clip never appears in both train and test.
- Validation for model selection and early stopping uses **seen** techniques only (or a
  separate held-out technique that is not the final test technique), so the unseen test
  set is never used for tuning.
- Where the dataset allows, a **leave-one-technique-out** scheme is used: train on all
  techniques but one, test on the held-out one, and repeat for every technique.
- A **cross-dataset** evaluation (train on dataset A, test on dataset B) is planned as a
  stronger test of generalization, depending on which datasets are selected.

The specific datasets and technique lists are **not decided yet**. Candidates to review
include FaceForensics++, Celeb-DF, DFDC, DeeperForensics-1.0, and newer multi-generator
benchmarks. Selection criteria: number of distinct techniques, label quality, licence
and access terms, size relative to available compute, and availability of identity /
source-video metadata.

## 11. Baseline Model Plan

- **Architecture:** an ImageNet-pretrained CNN from `torchvision` (candidate:
  EfficientNet-B0 or ResNet-50) with a single-logit binary head.
- **Input:** face crops at the configured image size (default 224×224).
- **Loss / optimiser:** binary cross-entropy with logits, AdamW, standard learning-rate
  schedule.
- **Augmentation:** standard only (flip, mild colour jitter, resize/crop).
- **Video-level decision:** mean of per-frame fake probabilities, thresholded at a value
  chosen on the validation split.
- **Purpose:** establish how large the seen-vs-unseen performance gap is before any
  generalization-specific method is applied.

## 12. Proposed Improvement Plan

Candidate improvements, to be narrowed down after the baseline results. Each directly
targets over-reliance on generator-specific artifacts:

1. **Forgery-simulation augmentation:** synthesise "fake-like" training samples from real
   faces (e.g. self-blending a face with a transformed copy of itself) so the model learns
   generic blending-boundary and inconsistency cues rather than one generator's
   fingerprint.
2. **Robustness augmentation:** JPEG/video compression, blur, noise, and resizing to stop
   the model relying on fragile high-frequency signatures.
3. **Frequency-aware input or branch:** add a frequency-domain representation (e.g. an
   FFT/DCT-based branch) alongside RGB.
4. **Temporal consistency (optional):** light-weight aggregation that uses variation across
   frames instead of a plain average.

Each improvement is kept only if it improves **unseen-technique** metrics; improvements
that help only seen techniques are reported but not counted as a success.

## 13. Evaluation Metrics

Reported at both **frame level** and **video level**, separately for **seen** and
**unseen** techniques:

| Metric    | Role |
|-----------|------|
| Accuracy  | Overall correctness at the chosen threshold |
| Precision | Of videos flagged fake, how many are fake |
| Recall    | Of fake videos, how many are caught (critical for unseen techniques) |
| F1-score  | Balance of precision and recall |
| ROC-AUC   | Threshold-free ranking quality; **primary metric** for generalization |

ROC-AUC is the primary metric because thresholds chosen on seen techniques often do not
transfer to unseen ones. Threshold-dependent metrics are reported using the threshold
selected on the validation split, so any calibration drop on unseen techniques is visible
rather than hidden. Where the number of test videos allows, results are reported as mean
± standard deviation over multiple random seeds.

## 14. Planned Experiments

1. **E0 – In-distribution reference:** train and test on the same techniques (identity-
   disjoint). Used only as an upper reference.
2. **E1 – Baseline, unseen techniques:** baseline trained on known techniques, tested on
   held-out techniques. Measures the generalization gap.
3. **E2 – Leave-one-technique-out:** repeat E1 holding out each technique in turn.
4. **E3 – Improved model, unseen techniques:** same protocol as E1/E2 with the proposed
   improvement(s).
5. **E4 – Cross-dataset:** train on one dataset, test on another (if the selected
   datasets allow).
6. **E5 – Robustness:** evaluate E1/E3 models under compression and resolution changes.

## 15. Ablation Study

Each factor is varied while others stay fixed, always evaluated on unseen techniques:

- each proposed improvement component on/off;
- number of sampled frames per video;
- frame-to-video aggregation method (mean, median, max, top-k mean);
- face crop margin and input resolution;
- backbone choice and pretrained vs. randomly initialised weights;
- augmentation strength.

## 16. Error Analysis

- Confusion matrices per technique (seen vs. unseen).
- Per-technique ROC curves and score distributions for real vs. fake.
- Inspection of the most confident false positives and false negatives.
- Grouping errors by video properties: compression level, resolution, face size, pose,
  lighting, occlusion, and face-detection failures.
- Visual explanations (e.g. Grad-CAM) to check whether the model attends to face regions
  or to background / compression artifacts.
- Calibration analysis (reliability diagrams) on seen vs. unseen techniques.

## 17. Limitations

- Available datasets cover a limited, somewhat dated set of generators; "unseen" means
  unseen within those datasets, not unseen in the real world.
- Face-centric detection does not address fully synthetic scenes, non-face manipulation,
  or audio deepfakes.
- Local development is CPU-only; large-scale training and extensive hyper-parameter search
  depend on access to cloud GPUs.
- Frame-based models largely ignore temporal artifacts.
- Dataset biases (identity, ethnicity, recording conditions, compression) may affect
  results and fairness.

## 18. Future Work

- Evaluate on newer diffusion-based and commercial video generators.
- Stronger temporal models and audio-visual consistency checks.
- Foundation-model features (e.g. large pretrained vision encoders) for generalization.
- Continual / few-shot adaptation to newly released generators.
- Calibration methods that remain reliable under generator shift.
- A demonstration interface (`demo/`) once the research results are established.

## 19. Project Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| 0 | Project setup, structure, config, environment validation | Done |
| 1 | Dataset review and selection; define technique-held-out protocol | Next |
| 2 | Preprocessing pipeline (frames, faces, crops) on a small subset | Planned |
| 3 | Split generation (technique- and identity-disjoint) in `data/splits/` | Planned |
| 4 | Dataset / DataLoader classes | Planned |
| 5 | Baseline training and evaluation (E0–E2), on cloud GPU if needed | Planned |
| 6 | Proposed improvements (E3) and cross-dataset / robustness (E4–E5) | Planned |
| 7 | Ablations and error analysis | Planned |
| 8 | Report writing; optional demo | Planned |

---

## Repository Structure

```
deepfake-generalization/
├── data/
│   ├── raw/          # original videos (not committed)
│   ├── frames/       # extracted frames (not committed)
│   ├── faces/        # face crops (not committed)
│   └── splits/       # split definition files (committed)
├── src/
│   ├── config.py     # experiment configuration
│   ├── preprocessing/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── docs/             # dataset selection and research protocol
├── experiments/      # experiment definitions / configs
├── results/          # metrics, figures, checkpoints, logs (not committed)
├── notebooks/        # exploratory analysis
├── demo/             # future demonstration (not started)
├── report/           # written report
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.config      # prints the active configuration and selected device
```

On a cloud GPU machine, install the CUDA build of `torch` / `torchvision` from
[pytorch.org](https://pytorch.org/get-started/locally/) first, then install the remaining
requirements.

### Configuration overrides

Any setting in `src/config.py` can be overridden with an environment variable prefixed
with `DFG_`, for example:

```powershell
$env:DFG_DATA_ROOT = "D:\datasets\deepfake"
$env:DFG_BATCH_SIZE = "64"
$env:DFG_DEVICE = "cpu"
```
#   d e e p f a k e - g e n e r a l i z a t i o n  
 