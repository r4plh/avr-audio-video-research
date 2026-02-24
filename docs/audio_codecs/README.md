# Understanding Audio Codecs: From MP3 to Neural Codecs like DAC

Before diving into the DAC paper or any neural audio codec, it's worth spending some time understanding what codecs actually do, why they exist, and how to reason about them. This piece covers the fundamentals I found useful while exploring DAC.

---

## What Problem Do Codecs Solve?

Audio files are massive. A simple 3-minute song stored as raw audio can easily be 30-60 MB. Streaming services like Spotify have 500 million users playing billions of songs per month. Without compression, the bandwidth costs alone would be astronomical - we're talking hundreds of millions of dollars monthly just for transmission.

Codecs solve this by compressing audio. The idea is simple: take raw audio, encode it into a smaller representation, transmit or store that, and decode it back to audio when needed. The encoder runs once (on the server), and the decoder runs in real-time (on your phone). Every phone has MP3, AAC, and other decoders built in - that's why you can play music without installing anything special.

---

## Bit Depth and Bitrate: Getting the Units Right

These two terms sound similar but measure completely different things.

**Bit depth** is about precision - how accurately each audio sample's amplitude is stored. With 16-bit audio (the CD standard), each sample can take one of 65,536 possible values (ranging from -32768 to +32767). This gives roughly 96 dB of dynamic range, which is enough for human hearing. Higher bit depths like 24-bit are used in studios but 16-bit is standard for distribution.

**Bitrate** is about data flow - how many bits of data per second. This tells you how much storage or bandwidth you need. The unit is bits per second (bps), typically expressed as kilobits per second (kbps).

A quick note on units since this trips people up: lowercase 'b' means bits, uppercase 'B' means bytes. So `kb` is kilobits, `KB` is kilobytes, and `kbps` is kilobits per second. Since 8 bits = 1 byte, you divide by 8 to convert. In the decimal (SI) standard, 1000 kilobytes = 1 megabyte. So to go from kilobits to megabytes: divide by 8000.

---

## Calculating Raw Audio Bitrate

For uncompressed audio, bitrate follows a simple formula:

```
Bitrate = Sample Rate × Bit Depth × Channels
```

Take 16kHz, 16-bit, mono audio (typical for speech models):
```
16000 × 16 × 1 = 256,000 bps = 256 kbps
```

For CD quality (44.1kHz, 16-bit, stereo):
```
44100 × 16 × 2 = 1,411,200 bps ≈ 1411 kbps
```

Most speech and audio ML work uses mono (channels = 1), which is what both Whisper and DAC expect.

---

## File Size from Bitrate

Once you know bitrate, file size is straightforward:

```
File Size = Bitrate × Duration
```

For a 1-minute file at 256 kbps:
```
256 kbps × 60 sec = 15,360 kb = 1920 KB ≈ 1.9 MB
```

This relationship holds for both raw audio and compressed formats - you just use the appropriate bitrate.

---

## The Spotify Example

Here's where codecs earn their keep. Consider what happens when you press play on a 3-minute song:

Without compression (raw WAV at 44.1kHz stereo):
- File size: ~30 MB
- At 1.4 Mbps, streaming this in real time requires a consistently strong connection with no margin for fluctuation
- Spotify's monthly bandwidth bill would be catastrophic

With compression (MP3 at 128 kbps):
- File size: ~2.8 MB
- Only ~16 KB needs to arrive per second to keep up with playback - streams smoothly even on weak connections
- Bandwidth costs drop by 10x

The flow looks like this: the artist uploads a WAV file, Spotify's servers encode it to MP3 (this happens once and gets stored), when you stream it only 2.8 MB travels over the internet, and your phone's built-in MP3 decoder reconstructs the audio in real-time.

At Spotify's scale of hundreds of billions of streams per month, the difference between transmitting raw audio versus compressed audio is genuinely hundreds of millions of dollars in bandwidth costs.

---

## Codec Bitrate vs Raw Audio Bitrate

This distinction matters. Raw audio bitrate is calculated from audio properties (sample rate, bit depth, channels). Codec bitrate is a property of the codec itself - either fixed by architecture or configurable by the user.

For raw audio, you calculate it. For codecs, you look it up from the documentation or paper, then multiply by duration to get compressed file size.

---

## Two Types of Codecs

Codecs fall into two categories based on how their bitrate is determined.

**Configurable bitrate codecs** (traditional): MP3, AAC, OGG, FLAC. You choose the bitrate (128, 256, 320 kbps, etc.) and the encoder adjusts quality to fit that budget. Same algorithm, different target sizes. These have been around since the 90s.

To make this concrete: MP3 processes audio in fixed frames of 1152 samples. At 44.1kHz that's ~26ms per frame, giving ~38 frames per second. When you target 128 kbps, each frame gets a bit budget:

```
128,000 bps ÷ 38 frames/sec ≈ 3,400 bits per frame
```

The psychoacoustic model then decides how to spend those 3,400 bits — assigning more to frequencies that matter perceptually and fewer (or zero) to frequencies you won't notice missing. At 320 kbps the budget is ~8,400 bits per frame and almost nothing gets discarded. At 64 kbps the budget is ~1,700 bits and the model has to be much more aggressive. The algorithm is the same; only the per-frame budget changes. The standard bitrate options (32, 64, 128, 192, 320 kbps etc.) are a fixed list defined in the MP3 spec, chosen empirically through listening tests — not derived from a formula.

**Fixed bitrate codecs** (neural): DAC, EnCodec, SoundStream. The bitrate is determined by the architecture - number of codebooks, codebook size, temporal resolution. You can't change it without retraining the model. These are learned end-to-end on massive audio datasets.

---

## DAC Models: Fixed Bitrates from Architecture

DAC comes in three variants. Each has a fixed bitrate determined by its architecture:

| Model | Sample Rate | Hop | Codebooks | Bits | Steps/sec | Bitrate |
|-------|-------------|-----|-----------|------|-----------|---------|
| 16kHz | 16,000 | 320 | 12 | 10 | 50 | 6 kbps |
| 24kHz | 24,000 | 320 | 32 | 10 | 75 | 24 kbps |
| 44kHz | 44,100 | 512 | 9 | 10 | 86 | 8 kbps |

The calculation is:
```
Bitrate = (Sample Rate ÷ Hop) × Codebooks × Bits per code
```

For the 16kHz model:
```
(16000 ÷ 320) × 12 × 10 = 50 × 12 × 10 = 6000 bps = 6 kbps
```

**Understanding the columns:**

- **Hop**: The temporal downsampling factor. It's the product of encoder stride rates. For 16kHz model with strides [2,4,5,8], hop = 2×4×5×8 = 320.

- **Codebooks**: Number of residual vector quantization stages. More codebooks = more bits = higher quality but larger files.

- **Bits**: log2(codebook_size). With 1024 entries per codebook, you need 10 bits to index them (2^10 = 1024). Each code is an integer from 0-1023.

- **Steps/sec**: How many discrete time steps represent one second of audio. This is sample_rate ÷ hop. It's a property of the architecture - change the encoder strides and this changes.

The 8-dimensional vectors in each codebook are continuous floats, learned during training. But what gets stored/transmitted is just the integer index (10 bits). The decoder looks up these indices to reconstruct the continuous representation.

---

## MP3 vs DAC: The Efficiency Gap

MP3 is configurable, so comparing requires picking a target. For 16kHz audio (256 kbps raw), typical MP3 settings would be:
- 32 kbps: low quality, very compressed
- 64 kbps: acceptable for speech
- 128 kbps: good quality

Compare the compression ratios:
- MP3 at 64 kbps: 256 ÷ 64 = 4× compression
- DAC at 6 kbps: 256 ÷ 6 = 43× compression

For similar perceptual quality, MP3 needs around 64 kbps while DAC achieves it at 6 kbps. That's roughly 10× more efficient. This is why neural codecs matter - same quality, dramatically smaller files.

The difference comes from how they compress. MP3 uses hand-designed psychoacoustic models from the 1990s, removing frequencies humans supposedly can't hear. DAC learns optimal discrete representations end-to-end from millions of audio samples. The learned codebook captures patterns that no hand-designed algorithm could.

---

## Why This Matters for Audio ML

Beyond compression, neural codecs like DAC unlock something important for deep learning: discrete audio tokens.

Language models work on discrete tokens - words, subwords, characters. Raw audio is continuous waveform data. You can't directly train a GPT-style model on audio samples.

DAC's encoder produces discrete codes - integers from 0-1023, arranged in sequences. These are exactly like vocabulary tokens. You can train autoregressive models (AudioLM, MusicGen, VALL-E) to predict the next audio token, just like predicting the next word. Then decode back to audio.

For 1 second of 16kHz audio:
- Raw: 16,000 continuous samples
- After DAC: 50 time steps × 12 codebooks = 600 discrete tokens

This tokenization is what enables the current wave of audio language models.

---

## Summary

To reason about audio codecs:

1. **Raw audio bitrate** = sample rate × bit depth × channels (calculate it)
2. **Codec bitrate** = depends on codec architecture/settings (look it up)
3. **File size** = bitrate × duration
4. **Traditional codecs** (MP3, AAC): configurable bitrate, user chooses quality
5. **Neural codecs** (DAC, EnCodec): fixed bitrate from architecture, much more efficient

DAC achieves ~6-8 kbps while maintaining high fidelity - roughly 10× better than MP3 for equivalent quality. And the discrete codes it produces are exactly what audio language models need for training.