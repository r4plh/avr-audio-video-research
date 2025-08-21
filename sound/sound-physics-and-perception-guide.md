# Understanding Sound: From Frequency to Pitch

This document provides a comprehensive summary of the core concepts of sound, covering the physics of waves and the psychoacoustics of human perception. It is structured to build from fundamental principles to the more complex relationships revealed in a pitch-frequency chart.

## 1. The Fundamentals of a Wave

Every repeating wave, including sound, can be described by three basic properties:

- **Frequency (f)**: This is the number of complete cycles a wave completes in one second. It's measured in Hertz (Hz). We perceive frequency as pitch.

- **Wavelength (λ)**: This is the physical distance covered by one complete cycle of a wave.

- **Period (T)**: This is the time it takes for one complete cycle to pass a specific point. It's the reciprocal of frequency.

The fundamental relationship is:

```
T = 1/f
```

This means that for a wave with a high frequency (many cycles per second), the time period for each cycle must be very short.

## 2. Pitch vs. Loudness: What We Actually Hear

Our ears and brain interpret the physical properties of sound waves into two distinct perceptions: pitch and loudness.

### Pitch
- How "high" or "low" a sound is (e.g., a shrill whistle vs. a deep rumble)
- Determined by **Frequency**
- High Frequency = Fast vibrations of the e-drum = High Pitch

### Loudness
- How strong or weak a sound is
- Determined by **Amplitude** (the wave's maximum displacement)
- High Amplitude = A stronger pressure wave that moves the e-drum a larger distance = Loud Sound

> **Note**: A sound can be loud and low-pitched (like a foghorn) or quiet and high-pitched (like a whisper). The two properties are independent.

## 3. Simple vs. Complex Sounds: The Real World is Not a Sine Wave

### Simple Periodic Sound
- A sound with a pure, clean sine wave, like a tuning fork
- Contains only one frequency
- In this simple case, the general term "frequency" and "fundamental frequency" are the same

### Complex Sound (Quasi-Periodic)
- Sounds from real-world sources like musical instruments or the human voice
- Their waveforms are complex and jagged, but they have a repeating pattern
- These complex sounds are not made of one frequency but are a combination of many simple sine waves added together
- The mathematical "recipe" for this combination is called a **Fourier Series**

## 4. The Recipe of Sound: Fundamental Frequency and Harmonics

Every complex pitched sound is made of two types of ingredients:

### The Fundamental Frequency (f₀)
- The lowest and loudest frequency component in the sound mix
- Determines the pitch that we perceive
- When we say a note is "440 Hz," we are referring to its fundamental frequency
- Because it's the most dominant part of the sound, its repetition rate defines the overall Period (T) of the entire complex wave (T = 1/f₀)

### The Harmonics (Overtones)
- A series of higher, quieter frequencies that are perfect integer multiples of the fundamental (2f₀, 3f₀, 4f₀, etc.)
- They do not change the perceived pitch
- Their unique mix and relative loudness determine the sound's character or **timbre**
- Timbre is why a violin and a guitar playing the same note (same f₀) sound different

## 5. The Pitch-Frequency Graph: Visualizing a Law of Hearing

The pitch-frequency graph is a visual representation of a fundamental law of psychoacoustics.

### Y-Axis (Pitch)
- Represents our subjective, linear perception of how "high" a sound is
- Organized into musical notes (A, B, C, etc.) and octaves (A₀, A₁, A₂)
- Each step from one octave to the next (e.g., A₂ to A₃) is perceived as an equally sized musical interval

### X-Axis (Frequency)
- Represents the objective, physical measurement of the fundamental frequency (f₀) of each note
- Measured in Hertz

### The Logarithmic Curve
- Shows that to achieve the linear, equally spaced steps in pitch that we perceive, we need an exponential increase in frequency

## 6. The Rule of the Octave: A Universal Constant

The foundation of the pitch-frequency graph is the octave.

### The Rule
An octave is a perceptual phenomenon where two sounds are heard as being the "same" note but in a different register. This occurs when one sound has a fundamental frequency that is exactly double (a 2:1 ratio) the other.

### Universal Perception
This is not a cultural invention but a biological one, rooted in how our auditory system processes sound and the perfect alignment of harmonics.

### Mathematical Representation (MIDI)
In systems like MIDI, this perceptual relationship is standardized:
- An octave is always an equal interval of 12 units (semitones) on the MIDI scale
- Example: A₃ is MIDI note 57, and A₄ is MIDI note 69 (57 + 12)

## 7. Musical Instruments: Controllable Frequency Sources

The pitch-frequency graph can only be plotted for sound sources that can produce a range of stable, controllable fundamental frequencies.

### Musical Instruments
Any object or system that allows you to controllably and repeatedly change its physical properties (like string length, air column length, or electronic settings) to produce specific frequency ratios (like doubling for an octave) can be considered a musical instrument.

### Fixed or Aperiodic Sounds
This is why the graph does not apply to:
- A clap (aperiodic noise with no clear pitch)
- A tap on a glass (a pitched sound with a fixed, unchangeable fundamental frequency)