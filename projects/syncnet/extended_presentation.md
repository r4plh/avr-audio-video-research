---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 20px;
  }
  h1 {
    font-size: 32px;
  }
  h2 {
    font-size: 24px;
  }
  table {
    font-size: 18px;
  }
  th, td {
    padding: 8px 12px;
  }
  img {
    display: block;
    margin: 0 auto;
  }
---

<!-- _paginate: false -->

# SyncNet 
## Out of Time: Automated Lip Sync in the Wild

**Aman Agrawal**

---

# SyncNet Inference Pipeline

---

## Inference by SyncNet 

**Input Assumptions:** 25 FPS video, single speaker, no scene changes, face visible throughout

| **Stage** | **Operation** | **Details** |
|-----------|--------------|-------------|
| **1. Preprocessing** | Extract Frames & Audio | • Extract video frames → JPG images<br>• Convert audio → 16kHz mono WAV<br>|
| **2. Load & Convert** | Create Input Tensors | **Video :** Read frames → Stack → PyTorch tensor (1, T, C, H, W)<br>• T = total frames, C = 3 (RGB), H×W = frame dimensions<br><br>**Audio :** Load WAV → Extract MFCC → PyTorch tensor (1, 1, 13, T_audio)<br>• MFCC params: 25ms window, 10ms hop<br>• 13 = MFCC coefficients, T_audio = number of MFCC frames |
| **3. Validation** | Length Check | • Verify: `audio_samples/16000 == frames/25`<br>• Compute: `min_length = min(frames, ⌊audio_samples/640⌋)`<br>• 640 samples/frame = 16kHz / 25 FPS |

---

| **Stage** | **Operation** | **Details** |
|-----------|--------------|-------------|
| **4. Feature Extraction** | A-V Embedding (0.2s chunks) | **Video:** 5 frames → 3D CNN → 1024-D<br>• Index: `vframe : vframe+5`<br><br>**Audio:** 20 MFCC frames → 2D CNN → 1024-D<br>• Index: `vframe×4 : vframe×4+20`<br>• ×4 ratio: MFCC window length 25ms, hop 10ms vs video 40ms/frame<br>• 20 frames: 25ms+(19×10ms) ≈ 215ms ≈ 200ms (5 video frames) |
| **5. Sync Calculation** | Distance Across Shifts | • Pad audio embeddings ±vshift frames (e.g., vshift=10)<br>• Test 21 windows: [-10, ..., 0, ..., +10] (2×vshift+1)<br>• Compute L2 distance for each shift<br>• Find minimum: `offset = vshift - argmin(distance)` |
| **6. Output** | Sync Metrics | • **Offset:** Audio ahead (+) or behind (-) in frames<br>• **Confidence:** `median(dist) - min(dist)` <br>• **Frame-wise confidence:** Per-frame sync quality |

---

## Example Walkthrough - Part 1: Extraction & Loading

**Sample Video:** 36 seconds, 900 frames @ 25 FPS, 224×224 resolution

| **Step** | **Input** | **Operation** | **Output** | **Shape** |
|----------|-----------|---------------|------------|-----------|
| **1. Extract** | Video file | • Extract frames → JPG<br>• Extract audio → 16kHz WAV | • 900 frames<br>• 575,616 audio samples | Frames: 900<br>Audio: (575616,) |
| **2. Load** | Frames + Audio | • Stack frames → `imtv` tensor<br>• Extract MFCC → `cct` tensor | Video: `imtv`<br>Audio: `cct` | `(1, 3, 900, 224, 224)`<br>`(1, 1, 13, 3597)` |
| **3. Validate** | Duration check | • Video: 900/25 = 36.0s<br>• Audio: 575616/16000 = 35.98s<br>• `min_length = min(900, ⌊575616/640⌋)` | `min_length = 899`<br>`lastframe = 894` | ⚠️ Slight mismatch: 0.02s |

**Key Observations:**
- T (video) = 900 frames, T_audio (MFCC) = 3597 frames
- Ratio: 3597 / 900 ≈ 4 (MFCC hop 10ms vs video 40ms/frame @ 25 FPS)
- Will process 894 windows (0.2s chunks)

---

## Example Walkthrough - Part 2: Feature Extraction

**Batch Processing:** 894 windows in 45 batches (batch_size=20)

| **Component** | **Indexing Logic** | **Batch Dimensions** | **Example (vframe=0)** |
|---------------|-------------------|----------------------|------------------------|
| **Video Input** | `imtv[:,:,vframe:vframe+5,:,:]` | `(20, 3, 5, 224, 224)` | Frames 0→4<br>Slice: `[0:5]` |
| **Audio Input** | `cct[:,:,:,vframe×4:vframe×4+20]` | `(20, 1, 13, 20)` | MFCC 0→19<br>Slice: `[0:20]` |
| **3D CNN (Video)** | 5 frames → Conv3D layers | → `(20, 512)` flatten | → FC layers |
| **2D CNN (Audio)** | 20 MFCC frames → Conv2D | → `(20, 512)` flatten | → FC layers |
| **Embeddings** | After FC (512→1024) | `(20, 1024)` per batch | 1024-D vectors |

**After All Batches:**
- Video embeddings: `(894, 1024)`
- Audio embeddings: `(894, 1024)`

---

## Example Walkthrough - Part 3: Sync Calculation & Output

| **Step** | **Processing** | **Dimensions** | **Result** |
|----------|----------------|----------------|------------|
| **Padding** | Pad audio embeddings ±vshift | For vshift=10:<br>`(894+20, 1024)` | Audio padded to `(914, 1024)` |
| **Window Creation** | Create 21 shifted versions | 21 windows: [-10 to +10] | Each window: `(894, 1024)` |
| **Distance Calc** | For each video frame:<br>• Compare with 21 audio shifts<br>• L2 distance | Distance matrix:<br>`(894, 21)` | 894 frames × 21 shifts |
| **Mean Distance** | Average across all 894 frames | `(21,)` vector | 21 mean distance values |
| **Find Minimum** | `argmin` of mean distances | Scalar index | e.g., argmin = 7 |
| **Compute Offset** | `offset = vshift - argmin` | Scalar | 10 - 7 = **+3 frames**<br>(120ms @ 25 FPS) |
| **Confidence** | `median(distances) - min(distance)` | Scalar | e.g., **8.5** (high confidence) |

**Final Output:**
- **AV Offset:** +3 frames (audio 120ms ahead)
- **Confidence:** 8.5 (reliable sync detection)
- **Frame-wise Confidence:** Array shape `(894,)` with per-frame quality scores

---

## Training Performance Profiling

**Dataset:** 1,975 training steps, batch size 1024, Hardware: 224 CPUs, 2TB RAM
**Total Training Time:** 417 seconds (6.95 minutes / 0.12 hours)

### Timing Breakdown (These stats are for per batch that means per step of training in milliseconds)

| **Component** | **Mean** | **Min** | **Max** | **% of Total** |
|--------------|----------|---------|---------|----------------|
| **Data Loading** | | | | **3.0%** |
| └─ JPEG Load | 3.55 | 3.19 | 4.71 | |
| └─ Audio Load | 0.40 | 0.37 | 1.02 | |
| └─ Preprocessing | 1.41 | 0.71 | 2.81 | |
| └─ **Total** | **6.34** | 5.11 | 8.23 | |
| **Forward Pass** | | | | **40.3%** |
| └─ Video CNN | 81.00 | 78.48 | 1714.01 | |
| └─ Audio CNN | 3.36 | 3.29 | 9.09 | |
| └─ **Total** | **85.21** | 82.42 | 1718.33 | |
| **Backward Pass** | **79.26** | 78.76 | 84.34 | **37.5%** |
| **Loss Calculation** | **0.38** | 0.27 | 8.32 | **0.2%** |
| **Total Step Time** | **211.17** | 199.91 | 1844.93 | **100%** |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Throughput** | 4,898 samples/sec (avg) |
| **GPU Utilization** | Video CNN dominates (81ms vs 3.36ms audio) |
| **Bottleneck** | Forward + Backward passes (77.8% of time) |
| **Data Loading** | Well optimized (only 3% overhead) |

