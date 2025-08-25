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
- High Frequency = Fast vibrations of the eardrum = High Pitch

### Loudness
- How strong or weak a sound is
- Determined by **Amplitude** (the wave's maximum displacement)
- High Amplitude = A stronger pressure wave that moves the eardrum a larger distance = Loud Sound

> **Note**: A sound can be loud and low-pitched (like a foghorn) or quiet and high-pitched (like a whisper). The two properties are independent.

## 3. Simple vs. Complex Sounds: The Real World is Not a Sine Wave

### Simple Periodic Sound
- A sound with a pure, clean sine wave, like a tuning fork
- Contains only one frequency
- In this simple case, the general term "frequency" and "fundamental frequency" are the same
- A pure tone can be described by a simple mathematical equation like `y(t) = A sin(2πft)`

### Complex Sound (Quasi-Periodic)
- Sounds from real-world sources like musical instruments or the human voice
- Their waveforms are complex and jagged, but they have a repeating pattern
- It is impractical to write one single, simple equation for such a complex shape
- Instead of a single equation, these sounds are understood as a combination of many simple sine waves added together
- The mathematical "recipe" for this combination is called a **Fourier Series**
- This means we don't look for one equation for the whole sound; we look for the many simple sine waves that build it

## 4. The Recipe of Sound: Fundamental Frequency and Harmonics

Every complex pitched sound is made of two types of ingredients:

### The Fundamental Frequency (f₀)
- The lowest and often the loudest frequency (not compulsory) component in the sound mix
- Determines the pitch that we perceive
- When we say a note is "440 Hz," we are referring to its fundamental frequency
- Because it's the most dominant part of the sound, its repetition rate defines the overall Period (T) of the entire complex wave (T = 1/f₀)
- This is the "prime" frequency that both our brain and measurement algorithms lock onto

### The Harmonics (Overtones)
- A series of higher, quieter frequencies that are perfect integer multiples of the fundamental (2f₀, 3f₀, 4f₀, etc.)
- They do not change the perceived pitch
- Their unique mix and relative loudness determine the sound's character or **timbre**
- Timbre is why a violin and a guitar playing the same note (same f₀) sound different

## 5. The Pitch-Frequency Graph: Visualizing a Law of Hearing

The pitch-frequency graph is not a plot of a single sound, but a chart that visualizes a fundamental law of psychoacoustics by plotting multiple, separate notes. It is a "law-based graph" derived from observation.

### How the Graph is Constructed
Each point on the graph is a separate measurement. A musician plays a note (e.g., A₂), and we measure its fundamental frequency (110 Hz). Then they play a different note (A₃), and we measure its fundamental frequency (220 Hz). The curve is the line that connects all these individual measurements.

### Y-Axis (Pitch)
- Represents our subjective, linear perception of how "high" a sound is
- Organized into musical notes (A, B, C, etc.) and octaves (A₀, A₁, A₂)
- Each step from one octave to the next (e.g., A₂ to A₃) is perceived as an equally sized musical interval
- This linear spacing is based on the universal biological perception of the octave

### X-Axis (Frequency)
- Represents the objective, physical measurement of the fundamental frequency (f₀) of each note, measured in Hertz
- We plot the fundamental frequency because, it is the "prime" frequency that our brain uses to identify the pitch of a complex sound

### The Logarithmic Curve
The graph's curve is the natural result of plotting our linear perception against physical reality. It shows that to achieve the linear, equally spaced steps in pitch that we perceive on the y-axis, we need an exponential increase in frequency on the x-axis.

![Pitch-Frequency Graph](assets/pitchFreqGraphOctave.png)

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

### What Makes a Source "Controllable"?
A musical instrument is any system that allows you to controllably and repeatedly change its physical properties to produce specific frequency ratios. This is achieved by:

- **Changing Length**: Shortening a guitar string with a fret or an air column in a flute with a key
- **Changing Tension**: Tightening a violin string with a tuning peg or tensing vocal cords
- **Changing Mass**: Using thicker or thinner strings on a bass vs. a guitar
- **Electronic Control**: Directly setting an oscillator's frequency in a synthesizer

### Fixed or Aperiodic Sounds
This is why the graph does not apply to:
- A clap (aperiodic noise with no clear pitch)
- A tap on a glass (a pitched sound with a fixed, unchangeable fundamental frequency)

These sources do not provide the variable range of fundamental frequencies needed to plot the curve.

## 8. Timbre: The "Color" of Sound

Timbre (pronounced TAM-ber) is the specific sound quality, character, or "color" of a note. It's precisely what you hear from the combination of the fundamental frequency and all of its unique harmonics.

Think of a sound as a recipe:

- **Fundamental Frequency** : This is the main ingredient. It tells you what the dish is (e.g., "pasta"). This is the pitch of the note.

- **Harmonics** : These are the spices and other ingredients. Their specific mix and strength are what make the dish unique (e.g., "pasta carbonara" vs. "pasta pesto"). This is the timbre.

The final sound you hear is the complete dish—the combination of the main ingredient and all the spices. Timbre is the reason a violin and a piano playing the exact same note (same fundamental frequency) sound so different from each other. They have the same main ingredient but a different blend of spices, determined by the instrument's physical construction and how it's played.

## 9. The Critical Distinction: Harmonics vs. New Fundamentals

This is a subtle but crucial point for true clarity.

- Harmonics are the higher frequencies that exist simultaneously with a single fundamental, giving that one note its unique timbre. A new note (like the next octave) is a completely new sound event with its own, new fundamental frequency.

When a guitarist plays an A₂ (110 Hz), 110 Hz is the fundamental. The harmonics (220 Hz, 330 Hz, etc.) exist relative to this fundamental. When they change their fingering to play an A₃ (220 Hz), they have created a new sound event where 220 Hz is the new fundamental.

It is less precise to say "the instrument is being played at its 2nd harmonic to get the octave." The more accurate and correct statement is:

"Every time it's played at a new fundamental frequency, which mathematically happens to be double the previous fundamental frequency."



# 9.1 Understanding The Critical Distinction: Harmonics vs. New Fundamentals in depth

## 9.1.1 The Two Pillars of Sound: Pitch and Timbre

Every musical sound has two primary characteristics:

* **Pitch:** Our perception of how "high" or "low" a note is. Pitch is determined by the sound's **Fundamental Frequency**.
* **Timbre (pronounced tam-ber):** The unique "color," character, or quality of a sound. It's what allows us to distinguish between a piano and a violin playing the exact same note. Timbre is determined by the presence and intensity of **Harmonics**.


## 9.1.2 The Fundamental Frequency: The Note's Identity

The fundamental frequency is the single most important component of a musical note.

### What is it?
The **fundamental frequency** is the lowest frequency present in a complex sound wave. It acts as the "foundation" or "floor" upon which the rest of the sound is built. When we name a note, like **A4 = 440 Hz**, we are referring to its fundamental frequency.

### The Brain's Perception: It's Not About Loudness 🧠
A crucial point is that the fundamental frequency does **not** have to be the loudest component for us to perceive its pitch. Our brain is a pattern-recognition machine. It identifies the pitch by recognizing the mathematical spacing of the entire harmonic series.

* **The Missing Fundamental:** Even if the fundamental is very quiet or completely absent, our brain can infer it from the pattern of the harmonics and we still hear the correct pitch,cannot go in more depth regarding this as it is more related to biologial perception of out brain.

## 9.1.3. Harmonics: The Sound's Character

If the fundamental gives a note its identity, the harmonics give it its personality.

### What are they?
**Harmonics** are additional frequencies that exist simultaneously within a single sound. Their frequencies are **integer multiples** of the fundamental frequency.

* **Example (for A0 = 27.5 Hz):**
    * **Fundamental:** 27.5 Hz (This defines the note as A0)
    * **2nd Harmonic:** 55.0 Hz (2 x 27.5)
    * **3rd Harmonic:** 82.5 Hz (3 x 27.5)
    * **4th Harmonic:** 110.0 Hz (4 x 27.5)
    * ...and so on.

### The Role of Context: Fundamental vs. Harmonic
A single frequency can play two different roles depending on the note being played. This is the key to understanding the nested structure of music.

* **Example with 55 Hz:**
    * When you play the **A0** note, 55 Hz exists as its **2nd harmonic**. It is part of the timbre of A0.
    * When you play the **A1** note, 55 Hz is the **fundamental frequency**. It is the pitch identity of A1.

Instruments are designed to play notes by producing specific **fundamentals**. The harmonics are the natural, simultaneous byproducts that give the instrument its unique sound.

## 9.1.4. Naming the Notes: Pitch Class and Octaves

How we refer to the "same note" has two layers of meaning.

### Pitch Class (Same Note Letter)
In a general sense, all notes with the same letter name (e.g., all C's, all F♯'s) belong to the same **pitch class**. Our brains perceive them as related because their fundamental frequencies are mathematically linked by octaves and octaves is again a biological universal thing which exists for normal human beings in their brain perception.

### Exact Pitch (Note + Octave)
For a specific, unique sound, we must define both the note letter and its octave, like **C4** (Middle C) or **A4**. Two notes have the exact same pitch only if they have the identical fundamental frequency.


## 9.1.5. Practical Application: Synthesizer Frequencies 🎹

The principles above are standardized in modern music using a system called **12-Tone Equal Temperament (12-TET)** with **A4 = 440 Hz** as the tuning reference. This provides a precise fundamental frequency for every note a synthesizer can produce.

## 9.1.5.1 The Universal Law of the Octave

The octave rule is a fundamental principle of music and physics. It states that to go up one octave, you double the frequency, and to go down one octave, you halve the frequency.

This rule applies equally to every note in the musical alphabet. It's the consistent mathematical relationship that makes an octave sound like the "same note, but higher or lower" to our ears, regardless of whether it's a C, a G, or an F♯.

For example:

We know A4 is 440 Hz. Therefore, A5 is 880 Hz (440 * 2), and A3 is 220 Hz (440 / 2).

Similarly, we know C4 is ~261.63 Hz. Therefore, C5 is ~523.25 Hz (261.63 * 2), and C3 is ~130.81 Hz (261.63 / 2).

This principle is the foundation of how musical scales and tuning systems are structured across the entire range of an instrument.

The chart below shows the fundamental frequency for each note in the standard piano range.

![Octave Notes with fundamental frequency](assets/synthOctaveNotes.png)


* **How to read the chart:**
    * Moving **vertically** down a column shows the **octave** relationship (doubling/halving the frequency).
    * Moving **horizontally** across a row shows the **semitone** relationship (multiplying by ~1.05946).

## 9.1.6. Summary of Key Takeaways

* **Pitch** comes from the **Fundamental Frequency**.
* **Timbre** comes from the **Harmonics** and other things
* The brain perceives pitch by recognizing the **harmonic pattern**, not just the loudest frequency.
* A frequency's role as a **fundamental** or a **harmonic** depends entirely on the **context** of which note is being played.
