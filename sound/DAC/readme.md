# DAC and the Path to Autoregressive Audio Models

If you're trying to build something like a next-audio-token predictor - an autoregressive model for audio similar to how GPT predicts the next word - you'll quickly run into a fundamental problem: audio doesn't come in nice discrete tokens like text does. This is where neural audio codecs like DAC (Descript Audio Codec) become essential. They bridge the gap between continuous audio signals and discrete token sequences that transformers can work with.

This piece walks through how DAC's Residual Vector Quantization works, why the design choices make sense, and how it all connects to building audio language models. We'll focus on the 16kHz DAC model throughout since that's what we'll use later for building an autoregressive audio model.

---

## The Starting Point: Continuous Latent Representations

Let's trace what happens to 1 second of 16kHz audio through DAC. The encoder takes raw audio and outputs a continuous latent representation. For 16,000 samples, we get 50 time steps, each a 1024-dimensional vector.

Where does 50 come from? It's determined by the encoder's stride. The encoder uses a series of strided convolutions with factors [2, 4, 5, 8], giving a total downsampling of 2 × 4 × 5 × 8 = 320. So 16,000 samples ÷ 320 = 50 time steps. The frame rate follows directly from the architecture: sample rate divided by encoder stride gives you frames per second.

This 50 × 1024 representation is continuous - each value is a float. We can't store or transmit this efficiently, and more importantly, we can't use it directly as tokens for a language model. That's where RVQ comes in.

---

## How Residual Vector Quantization Works

RVQ converts continuous vectors into discrete codes. DAC uses 12 codebooks, each with 1024 entries. But here's something that often causes confusion: each codebook entry is an 8-dimensional vector, not 1024-dimensional. The 1024-dimensional space is where the encoder output and residuals live. The 8-dimensional space is where the codebook lookup happens.

### The Projection Strategy

Why use two different dimensionalities? The projection from 1024 to 8 dimensions before codebook lookup serves a specific purpose: it improves codebook utilization and prevents codebook collapse. Without this projection, many codebook entries would go unused - the model would learn to use only a small subset of the available codes. By projecting to a lower-dimensional space, we approximate principal components for lookup, maximizing entropy per codebook.

Think of it this way: lookup happens in a compressed similarity space, but reconstruction happens in a rich space.

### Step by Step Through One Codebook

Let's trace what happens for a single time step through the first codebook:

**Step 1 - Project down**: Each codebook has a learned in_proj layer (a Conv1d, 1024 → 8). We project our continuous vector:
```
z [1024](encoder's output) → in_proj → z_projected [8]
```

**Step 2 - L2 normalize**: Both the projected vector and codebook entries are L2-normalized. This converts Euclidean distance to cosine similarity, which trains more stably.

**Step 3 - Find nearest entry**: The codebook is a learned embedding table with 1024 entries, each 8-dimensional. We compute distance to all entries and take the argmin:
```
distances[i] = ||z_projected - codebook[i]||²
code_1 = argmin(distances)  # e.g., 742
```

This integer (0-1023) is our first code. It takes 10 bits to store (2^10 = 1024).

**Step 4 - Project back up**: Each codebook also has a learned out_proj layer (a Conv1d, 8 → 1024). We look up the codebook entry and project it back:
```
z_q_8dim = codebook[742]  # [8] - from lookup
z_q_1 = out_proj(z_q_8dim)  # [1024] - learned projection
```

This is important: out_proj is a learned linear transform, not a lookup table. The index selects the 8-dimensional vector; the projection turns it into 1024-dimensional (each of the 12 codebooks has its own in_proj/out_proj). No matter which index (0-1023) is selected, the same projection matrix is used. Only the 8-D lookup vector changes, not the projection.

**Step 5 - Compute residual**:
```
residual = z - z_q_1  # [1024]
```

The residual represents what remains after subtracting the first codebook's contribution. We calculate it in 1024 dimensions because that's the encoder latent space - the decoder expects that space, and the training losses operate there. This is an architectural consistency choice.

**Step 6 - Repeat**: The residual goes through codebook 2 with the same process: project to 8-dim, find nearest, project back to 1024-dim, compute new residual. Then codebook 3, and so on through all 12 codebooks.

### The Residual Intuition

The key insight behind residual quantization is that we're capturing what remains to be modeled at each stage. Codebook 1 explains part of the signal. The residual isn't a "failure" - it's the remaining signal for subsequent codebooks to model.

Each successive codebook captures finer details. The paper explicitly notes that earlier codebooks capture coarse bits while later codebooks capture fine bits. All codebooks share the same dimensionality, but each learns to model a different aspect of the residual. The residual gets smaller with each step, though it never truly reaches zero - there's always some quantization error.

### Final Outputs

After all 12 codebooks:
```
codes = [code_1, code_2, ..., code_12]  # 12 integers per time step
z_quantized = z_q_1 + z_q_2 + ... + z_q_12  # [1024] - sum of all projections
```

For 1 second of audio, we get 50 time steps × 12 codes = 600 integers.

---

## Why "Quantized" if the Output is Continuous?

The z_quantized values are continuous floats, but the term "quantized" is still accurate. Quantization fundamentally means a many-to-one mapping - continuous values get mapped to a finite set of representatives. Even though the final representation is continuous, the information has been forced through a discrete bottleneck (12 integers per time step). Given the codes, the reconstruction is deterministic - look up each codebook entry, apply each out_proj, sum them. The discreteness comes from the bottleneck, not from whether the output values are integers.

---

## Trainable Parameters

Each of the 12 codebooks has three sets of learnable parameters:
- **in_proj**: Conv1d(1024 → 8) - projects down for comparison
- **codebook**: Embedding(1024, 8) - the 1024 learned 8-dim vectors
- **out_proj**: Conv1d(8 → 1024) - projects back up for residual calculation

All of these are trained end-to-end with backpropagation. The straight-through estimator handles the non-differentiable argmin operation. During training, there are stochastic effects from the straight-through estimator and dropout over quantizers. Unlike rule-based tokenizers like BPE, DAC's tokenizer is learned. But once training is complete, it becomes deterministic at inference - the same audio always produces the same codes.

---

## Parallelization at Inference

Once training is done, all weights are fixed. Here's what's parallel and what's sequential:

**Parallel across time**: All 50 time steps can go through the RVQ simultaneously since they share the same fixed weights.

**Sequential within each time step**: For a given time step, we must go through codebooks 1, 2, ..., 12 in order because each depends on the residual from the previous one. This sequential dependency within each time step matters for latency discussions.

---

## The Connection to Audio Language Models

### The Tokenization Problem

Language models work on discrete tokens. In NLP, a tokenizer (BPE, WordPiece, Unigram) breaks text into subwords and maps them to integer IDs from a fixed vocabulary. The model predicts probabilities over that vocabulary. These tokenizers are rule-based and deterministic - algorithmic preprocessing, not learned neural networks.

Audio doesn't have natural discrete units. Two people saying the same word produce completely different waveforms. There's no space character, no punctuation, no obvious way to segment. Text tokenization works because the segregation is easy - same text typed by two people is identical, which isn't true for audio.

### DAC as a Learned Audio Tokenizer

This is exactly what DAC provides. The encoder + RVQ is effectively a learned tokenizer:
- **Input**: Raw audio waveform
- **Output**: Sequence of discrete codes (integers 0-1023)
- **Deterministic at inference**: Once trained, the same audio always produces the same codes

For 1 second of 16kHz audio, we get 50 time steps. Each time step has 12 codes. That's the token representation of audio.

### Vocabulary Size Considerations

If we treat all 12 codes as a single combined token, the theoretical vocabulary size is 1024^12 ≈ 10^36. To put that in perspective, grains of sand on all Earth's beaches number around 10^20 - 10^36 is a trillion trillion times larger. Completely impractical as a flat vocabulary.

But this isn't how audio language models actually work. We don't have one giant token with a flat softmax over 10^36 options. Instead, we have 12 discrete variables. The modeling uses factorized distributions - it's closer to 12 parallel vocabularies of 1024 each, not one massive vocabulary.

In practice, audio language models handle this through:
- **Sequential prediction**: Predict codes from each codebook one at a time. Vocabulary size = 1024 per prediction.
- **Hierarchical patterns**: First codebooks capture coarse structure, later ones add detail. Can predict coarse first, then refine.
- **Independent heads**: Usually 12 independent output heads, each predicting one codebook with a softmax over 1024.

### Building a Next-Audio-Token Predictor

Here's the analogy to GPT:

| Text (GPT) | Audio (with DAC) |
|------------|------------------|
| Raw text | Raw audio waveform |
| BPE tokenizer | DAC encoder + RVQ |
| Token IDs | Codes [50, 12] for 1 sec |
| Vocabulary size | 1024 per codebook |
| Sequence length | Number of time steps |
| Predict next token | Predict next audio codes |
| Softmax over vocab | 12 softmax heads, 1024 each |
| Detokenize | DAC decoder |

The training loop would look like:
1. Encode training audio with DAC → get codes
2. Train transformer to predict next codes autoregressively (with causal masking)
3. At inference, generate codes step by step
4. Decode generated codes with DAC decoder → audio

For the output layer, you'd have a softmax over 1024 for each codebook position. The conditioning structure matters - often a coarse-to-fine approach is used, where earlier codebook predictions condition later ones.

### Local vs Global Context

One important architectural point: DAC's encoder is purely convolutional with no self-attention. Each time step contains information from its local neighborhood - the hop length plus receptive field from dilated convolutions - not the entire audio.

Compare this to Whisper, where self-attention means encoder outputs have a global receptive field. But even Whisper's outputs are positionally biased - it's not that the entire audio is collapsed equally into each time step.

For an audio language model built on DAC codes, the transformer's attention provides the global context that DAC's encoder doesn't. The transformer sees all previous time steps when predicting the next one.

---

## Positioning DAC

It's worth noting that DAC's primary goal is high-fidelity audio compression. The paper positions it as achieving excellent reconstruction quality at low bitrates. Autoregressive audio modeling is a downstream beneficiary - DAC provides the discrete representation that makes it possible, but DAC itself wasn't designed specifically for building audio GPTs. It's a drop-in tokenizer that happens to work beautifully for that purpose.

---

## Practical Specifications: 16kHz Model

| Property | Value |
|----------|-------|
| Sample rate | 16,000 Hz |
| Encoder strides | [2, 4, 5, 8] |
| Hop length | 320 (product of strides) |
| Time steps per second | 50 |
| Codebooks | 12 |
| Codebook size | 1024 entries |
| Codebook dimension | 8 |
| Latent dimension | 1024 |
| Bitrate | 6 kbps |

### Bitrate Calculation

```
Bitrate = (Sample Rate ÷ Hop) × Codebooks × log2(Codebook Size)
        = (16000 ÷ 320) × 12 × 10
        = 50 × 12 × 10
        = 6000 bps = 6 kbps
```

---

## Summary

DAC provides the bridge between continuous audio and discrete tokens. The RVQ process:

1. **Projects** continuous 1024-dim vectors to 8-dim for codebook lookup
2. **Finds** the nearest codebook entry via argmin
3. **Projects back** to 1024-dim using a learned linear transform
4. **Computes residual** in 1024-dim space
5. **Repeats** for all 12 codebooks, each capturing finer details

The key design choices serve specific purposes: the low-dimensional projection maximizes codebook utilization and prevents collapse; residual computation in high dimensions preserves information for subsequent codebooks; factorized prediction across 12 vocabularies of 1024 makes modeling tractable.

For building autoregressive audio models, DAC gives us exactly what we need: a learned tokenizer that converts continuous audio into discrete sequences. Each second becomes 50 time steps of 12-integer codes, analogous to tokens in text. Train a transformer to predict these codes autoregressively, decode with DAC's decoder, and you have audio generation.
