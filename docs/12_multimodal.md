# Section 12: Multimodal Capability Under Extreme VRAM Constraints

## 12.1 Design Strategy: CPU-Primary, GPU-Projection

Traditional multimodal models load a vision encoder entirely on GPU (400+ MB).
We cannot afford this. Instead:

**Vision**: CPU-resident encoder → small GPU projection adapter (10 MB VRAM)
**Audio**: CPU-only Whisper-tiny → text tokens (0 VRAM)

Total additional VRAM: **10 MB** (2.4% of slack budget).

## 12.2 Vision Subsystem

### Architecture: Quantized MobileCLIP on CPU + Linear Projector on GPU

```
Image Input (224×224 or 336×336)
        │
        ▼
┌──────────────────────┐
│ MobileCLIP-S2 Encoder│  ← CPU-resident, INT8 quantized
│ (35M params, ~35 MB) │     Host RAM
│ Output: 512-dim vec  │
└──────────┬───────────┘
           │  512-dim embedding via PCIe (2 KB transfer)
           ▼
┌──────────────────────┐
│ Vision Projector     │  ← GPU-resident
│ (512 → 2048, 2-layer│     VRAM: 10 MB
│  MLP with GeLU)      │
│ Output: 2048-dim vec │
└──────────┬───────────┘
           │  Injected as "visual tokens" into GLA sequence
           ▼
┌──────────────────────┐
│ GLA Sequence Processor│  ← Existing architecture
│ Visual tokens treated │
│ as additional context │
└──────────────────────┘
```

### MobileCLIP-S2 on CPU

| Property | Value |
|----------|-------|
| Parameters | 35M |
| Precision | INT8 (post-training quantization) |
| RAM footprint | 35 MB |
| Input resolution | 224×224 (standard) or 336×336 (high-res mode) |
| Output | 512-dim embedding |
| Inference runtime | CPU (AVX2/NEON) |
| Latency (224×224) | ~150 ms on 8-core CPU |
| Latency (336×336) | ~300 ms on 8-core CPU |
| Framework | ONNX Runtime (CPU-only) |

### Why MobileCLIP-S2

- Designed for mobile/edge: smallest CLIP variant with competitive accuracy
- 35M params vs SigLIP-base 86M or CLIP ViT-L/14 428M
- Published ImageNet zero-shot: 67.8% (vs 75.3% for ViT-L/14)
- Gap closed by: (a) retrieval-augmented visual understanding, (b) multi-crop for high-res

### Vision Projector (GPU)

```
Linear(512, 1024) + GeLU + Linear(1024, 2048)

Parameters: 512*1024 + 1024 + 1024*2048 + 2048 = 2,627,072
At FP16: 2,627,072 * 2 = 5.25 MB
With buffer overhead: ~10 MB VRAM
```

### Visual Token Injection

Each image produces N_v visual tokens injected into the GLA sequence:

| Mode | Visual tokens | Latency | Quality |
|------|--------------|---------|---------|
| Single-crop (224) | 1 token (global) | 152 ms | Basic captioning |
| Multi-crop (4 crops) | 4 tokens | 600 ms | Object detection level |
| Patch-stream (16 patches) | 16 tokens | 2400 ms | Fine-grained VQA |

Default: **4-crop** mode (4 visual tokens). Each crop is 224×224 from different
regions of the image, providing spatial diversity.

```rust
fn process_image(image: &Image, encoder: &MobileCLIP, projector: &VisionProjector) -> Vec<VisualToken> {
    // Step 1: Generate crops (CPU)
    let crops = generate_crops(image, n_crops = 4);
    // [top-left, top-right, bottom-left, bottom-right] at 224×224 each

    // Step 2: Encode each crop on CPU (~150ms each, parallelizable across 4 CPU cores)
    let embeddings: Vec<[f32; 512]> = crops.par_iter()
        .map(|crop| encoder.encode(crop))  // CPU inference
        .collect();
    // Total: ~150ms with 4 parallel cores

    // Step 3: Project to model space on GPU
    let visual_tokens: Vec<[f16; 2048]> = embeddings.iter()
        .map(|emb| projector.forward(emb))  // GPU: 10MB model, ~0.01ms per token
        .collect();

    visual_tokens
}
```

### Quality Targets

| Task | Metric | Target | Baseline (ViT-L/14) |
|------|--------|--------|---------------------|
| Image captioning (COCO) | CIDEr | >= 80 | 120 |
| Visual QA (VQAv2) | Accuracy | >= 55% | 75% |
| Object recognition (ImageNet) | Top-5 acc | >= 85% | 95% |

[HEURISTIC] These targets are significantly below state-of-the-art because we're
using a 35M encoder vs 428M. The gap is partially closed by:
1. Topological memory retrieval of relevant visual descriptions (+10-15%)
2. Multi-crop spatial reasoning (+5%)
3. Chain-of-thought visual reasoning ("describe what you see, then answer") (+5%)

Falsifiable test: Run VQAv2 benchmark with full pipeline, measure accuracy.

## 12.3 High-Resolution Image Handling

For images larger than 336×336:

```
Strategy: Progressive resolution pyramid

Level 1: Downsample to 224×224, encode (1 global token)     — 150 ms
Level 2: 4 crops at 224×224 from quadrants (4 tokens)        — 150 ms (parallel)
Level 3: 16 patches at 224×224 from 4×4 grid (16 tokens)    — 600 ms (4 batches of 4)

Total for Level 1+2: 300 ms, 5 visual tokens (DEFAULT)
Total for Level 1+2+3: 900 ms, 21 visual tokens (HIGH-RES mode, user opt-in)
```

Maximum visual tokens per image: 21 (at high-res). Each is 2048-dim * 2 bytes =
4 KB. Total per image: 21 * 4 KB = 84 KB injected into GLA sequence.

## 12.4 Audio Subsystem

### Architecture: Whisper-Tiny on CPU → Text → Main Model

Audio is handled by converting speech to text on CPU. No audio-native processing
on GPU. This is a pragmatic choice given VRAM constraints.

```
Audio Input (16kHz WAV/PCM)
        │
        ▼
┌──────────────────────┐
│ Whisper-Tiny Encoder  │  ← CPU-resident, INT8
│ (39M params, ~39 MB)  │     Host RAM
│ Output: text tokens    │
└──────────┬────────────┘
           │  Text string (no PCIe transfer needed, just CPU→CPU)
           ▼
┌──────────────────────┐
│ Standard text tokenizer│
│ Text → token IDs       │
└──────────┬────────────┘
           │  Token IDs to GPU (negligible transfer)
           ▼
┌──────────────────────┐
│ GLA Sequence Processor│  ← Existing architecture
│ Transcribed text as   │
│ normal text input      │
└──────────────────────┘
```

### Whisper-Tiny on CPU

| Property | Value |
|----------|-------|
| Parameters | 39M |
| Precision | INT8 quantized |
| RAM footprint | 39 MB |
| Languages | 99 languages |
| Input | 16kHz mono audio, 30-second chunks |
| Output | Transcribed text with timestamps |
| Latency | ~500 ms per 30 seconds of audio (CPU) |
| WER (English) | ~8% (vs ~4% for Whisper-Large) |
| WER (multilingual avg) | ~15% |
| Framework | ONNX Runtime / whisper.cpp |

### VRAM Cost: Exactly 0 MB

Audio processing is entirely CPU-resident. The output is text, which enters the
standard text processing pipeline.

### Audio Quality Targets

| Task | Metric | Target | Whisper-Large baseline |
|------|--------|--------|-----------------------|
| English ASR | WER | <= 10% | 4% |
| Multilingual ASR | WER | <= 18% | 8% |
| Audio QA | Accuracy | >= 50% | 70% |

### Limitation
- No audio understanding beyond speech (music, environmental sounds, tone/emotion)
- No speaker diarization
- 30-second chunk limit per inference (longer audio processed in chunks)
- [HEURISTIC] For audio understanding beyond ASR, would need an audio encoder on
  GPU (~100 MB). This is a future extension if VRAM budget allows. Falsifiable:
  measure user satisfaction with text-only audio processing vs audio-native.

## 12.5 Multimodal Integration with GLA

Visual tokens are injected into the GLA recurrent state alongside text tokens:

```
Sequence: [SYS_TOKENS] [USER_TEXT] [VIS_1] [VIS_2] [VIS_3] [VIS_4] [USER_TEXT_CONT] [RESPONSE]

Where VIS_i are 2048-dim visual tokens from the projector.
```

### GLA State Update for Visual Tokens

Visual tokens use the same GLA update equations (§3.2) as text tokens:
```
For visual token v_i:
  g_t = σ(W_g · v_i + b_g)
  k_t = W_k · v_i
  v_t = W_v · v_i
  S_t = g_t ⊙ S_{t-1} + k_t ⊗ v_t
```

No architectural change needed. The projector ensures visual tokens are in the
same 2048-dim space as text embeddings.

### Cross-Modal Retrieval

The topological memory can store both text and visual entries:
- Visual entries: store the 512-dim MobileCLIP embedding (before projection)
- Retrieval: query can be text OR visual; both map to comparable embedding spaces
- CLIP's text-image alignment means text queries retrieve relevant images and vice versa

Additional RAM for visual entries: 512 * 2 bytes * (number of visual memories).
Budget for 100K visual memories: 100K * 1 KB = 100 MB RAM.

## 12.6 Vendor-Agnostic Considerations

The vision projector (10 MB, simple MLP) requires:
- FP16 matmul: 512×1024 + 1024×2048
- Trivially supported on all GPU backends (CUDA, HIP, SYCL)
- No custom kernel needed; uses standard GEMM from existing dequant infrastructure
- CPU fallback: 2 matrix multiplications = ~5M FLOPs = 0.01 ms (trivial)

MobileCLIP and Whisper-Tiny run on CPU via ONNX Runtime, which supports:
- x86 AVX2/AVX-512 (Intel/AMD)
- ARM NEON (if ever ported to ARM)
- No GPU dependency

## 12.7 Total Multimodal Costs

| Component | VRAM | RAM | Latency |
|-----------|------|-----|---------|
| MobileCLIP-S2 encoder | 0 | 35 MB | 150 ms/image |
| Vision projector | 10 MB | 0 | 0.04 ms/image |
| Whisper-Tiny encoder | 0 | 39 MB | 500 ms/30s audio |
| Visual memory entries (100K) | 0 | 100 MB | 0 |
| **TOTAL** | **10 MB** | **174 MB** | **150 ms/image, 500 ms/30s audio** |

**VRAM delta: 10 MB (within 420 MB slack, uses 2.4%)**
**RAM delta: 174 MB (within 7148 MB headroom, uses 2.4%)**

## 12.8 Failure Modes

| Failure | Detection | Fallback |
|---------|-----------|----------|
| MobileCLIP produces degenerate embedding (all zeros) | Norm check: ||emb|| < 0.01 | Return "unable to process image" |
| Image too large (>4K resolution) | Size check before processing | Downsample to 1024×1024, then crop |
| Audio too noisy for Whisper | WER heuristic: if output has >50% [UNKNOWN] tokens | Return "audio unclear, please rephrase" |
| Vision projector weights corrupted | NaN check on output | Bypass projector; use zero-padded 512-dim embedding (degraded quality) |
| ONNX Runtime crash | Process-level isolation (separate thread) | Restart ONNX Runtime; if repeated, disable modality |
