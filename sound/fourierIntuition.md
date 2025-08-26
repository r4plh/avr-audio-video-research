# Understanding the Fourier Transform: An Intuitive Approach to Sound Frequency Analysis

I am writing my understanding of the Fourier transform, more like an intuition and what I understood from and in its application in sound frequency.The purpose of the blog post is to get the intuition of how fourier transform helped us to get to frequncy domain features from time domain features, not getting into heavy maths and derivations, instead we'll try to simplify the meaning conveyed from the complex maths equations.

So basically, we have the original audio signal which, for complex sounds (including harmonic sounds too) like human voice or musical instruments and other kinds of sounds, are often made up of a combination of frequencies - constituent frequencies, or let's say a superposition of frequencies. The continuous sound I'm talking about is in the time domain; it would basically be an amplitude vs. time graph. That is how the sampled points from the original sound are stored in the computer in the time domain in digital format, some basic understaning of how digital sound is stored and how sampling and quantization is done would be very beneficial to understand what lies ahead in this written piece of content. Basics could be covered from first few videos from here - [Basics of sound sampling](https://www.youtube.com/watch?v=KxRmbtJWUzI&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&index=12)

Now, the Fourier transform is the thing through which we get that graph to be converted from time domain (X-axis -> Time, Y-Axis -> Amplitude) into a frequency domain one (X-axis -> Frequency, Y-Axis -> Magnitude of contribution).

[Time to Frequency Coversion](./assets/timeTOFrequency) 

[Graph picture taken from from Valerio Velardo - Demystifying the Fourier Transform: The Intuition](https://www.youtube.com/watch?v=XQ45IgG6rJ4&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&index=10)

Now we'll talk more on an intuition level about what's our aim and how we get what we want from the Fourier transform.

Let's try to understand in an organized way:

## Our Goal
To get the values of those frequencies, the combination of which the original sound is made of. By original sound, what is meant is the digital sound that we have stored through sampling points and quantization through ADC into our digital system, in easy language to get the values of constituent frequencies from which the complex sound is made of.

## To Achieve the Goal
We want a graph which easily tells us which frequencies have what magnitude of contribution in making the original sound, so the x-axis of that graph should have all the frequency values and the y-axis should have the magnitude of the contribution corresponding to x, that is, frequency.

Our input would be the **Amplitude-Time graph** and output would be the **Magnitude-Frequency graph** for the Fourier Transform.
It's obvious, as the graphs sound so different, there would be mathematics involved, and to be honest, advanced tools of mathematics like the Fourier transform and complex numbers are used to convert from our input (Time domain graph) → Output (Frequency domain graph).

But to get the intuition of WHY the Fourier transform does the job correctly, that is, conversion of time domain to frequency domain, it's essential to understand WHAT the Fourier transform does such that our goal is achieved. Then we'll get to know HOW the Fourier transform helps us to achieve it at an intuition level.

So the **WHAT**, **WHY**, and **HOW**?

## The WHAT?

In answering the what, we don't need to see what math is going on inside, as we just want to know what the Fourier transform did, as in what output the Fourier transform gave such that we are able to determine the magnitude of contribution for all frequencies and able to find what consituent frequencies were involved in making the original sound.

For understanding the WHAT part of the Fourier transform, we will treat it like a black box and we'll talk in terms of the input it receives and the output it gives level by level. (FT → Fourier Transform)

First of all, the input to FT is a real number and the output is a complex number.

The input is one frequency value f for which we will get to know the magnitude corresponding to that frequency. So what FT does is take one frequency value as input and output a complex number through which we get to know what is the magnitude of contribution for that frequency. Ideally (not digitally) , this is done for all frequencies in real numbers, so we get a continuous plot of frequency vs. magnitude in the x-y axis. So basically, FT is applied across all frequencies in $\mathbb{R}$ and in real that is we'll be dealing with digital data so we would apply on discrete points not on continious, anyways doesn't matter continue reading further. The magnitude of the complex number as output of FT is actually the magnitude of contribution to that frequency. That complex number is also termed as the Fourier transform coefficient of the input frequency $f$.

Now the question is: WHAT is that complex number that FT outputs, right? Like, what is the interpretation of that?

In answering the WHAT only, let's move one level deeper. We know the input to FT is one frequency $f$ from real numbers. Now, although as said above, the output is a complex number whose magnitude fulfills our purpose of knowing the contribution of the frequency $f$ in the original audio digital signal, let's see how that complex Fourier transform for that $f$ came. So we are seeing how that complex number came. That complex number is actually the:

$$\text{Complex Fourier Coefficient} = \text{Centre Of Mass(COM)} \times \text{Number of Time Steps}$$

Once we understand whose COM we're talking about and what we mean by the number of time steps, we'll be able to understand the significance of that complex number (complex Fourier coefficient of input frequency $f$).

The output of FT is a complex-shaped graph. That complex graph depends on:
- The input frequency f given to FT
- The nature of the original audio digital signal
- Time (by this it means for how long a duration we are considering with respect to the original audio for conversion of time domain to frequency domain, and in terms of digital signal, it would mean how many time steps we are taking into account - more clarity latter)

**COM of what?** The FT of an input frequency $f$ will give a graph in the complex plane with real and imaginary axes. The graph depends upon the input frequency $f$, the original audio signal, and the time duration we are considering for making the graph. Consider that the graph is made up of uniform mass density, and what we want are the coordinates (real, imaginary) of the COM of the graph.

Now calculate magnitude of the COM co ordinates given for this assumed massed graph:
$$\text{Magnitude} = \sqrt{(\text{Real})^2 + (\text{Imaginary})^2}$$

This is basically how we calculate COM distance from origin in complex plane. We want that value, so we get the distance from the origin of this COM of this graph.

Now, to know the contribution of the input frequency $f$, what we have to do is multiply the distance from origin of COM, we calculated from the above step with the number of time steps we have used.

This will give us one set of coordinates for the frequency domain graph. The x-axis value would be the input frequency $f$ and the y-axis value would be the magnitude we calculated for the input frequency $f$ through FT, the magnitude of contribution.

Now we know what FT does to input frequency $f$, so that our goal is achieved. This FT is applied for all frequencies in real numbers and gets the magnitude of contribution for every value of $f$ and gets the continuous distribution of the frequency domain curve, and this is how we the graph which is on the right side of the picture above which we are giving term as frequency domain graph.

## Now, The HOW?

How does FT do this thing? Basically, what kind of math is applied and how is that complex graph made for the input frequency $f$? So we need to understand how that complex graph for the input frequency $f$ is made (anyways, we know its COM × number of time steps gives what we want, which we already discussed above in the WHY part). 

So how that complex graph is made is through multiplication of the original audio signal × $e$ to the power $(-2\pi itf)$, so if $g(t)$ is the original audio signal, the complex graph would be made by:

$$g(t) \cdot e^{-2\pi itf}$$

where $i$ is iota and we plot this graph in the complex plane. So while constructing the graph for a particular frequency $f$ through FT, the $f$ would be fixed when the graph would be made on the complex plane and what would be varying is time. The $f$ involved would eventually define how much time it takes to complete one full circle/rotation/cycle, basically after how much time the product of $2\pi tf = 2\pi$, which signifies one full circle is completed (when $t \cdot f = 1$). So the time period of this graph in the complex plane depends on the input frequency f we have taken for this FT. This is how that graph is made whose COM gives us what we require.

### A rough idea about the complex graph in the complex plane:
- The graph depends on the duration of the original sound audio signal g(t)
- The nature of the original audio
- It depends on frequency f (input of FT)

### How the graph would look like $g(t) \cdot e^{-2\pi itf}$ in the complex plane:
It's essential to understand how to visualize and interpret complex numbers
- $g(t)$ is the distance from the origin at an angle of $(-2\pi tf)$ in a clockwise direction from the real axis at a particular time $t$; the value of $g(t)$ is the distance from origin which magnitude of a complex number represent.
- The angle varies with time; at a particular time or snapshot, the angle from the real axis clockwise is given by $(-2\pi tf)$ (only time varies here.)
- So how we can interpret this thing is that we are wrapping or winding the original sound signal $g(t)$ around a unit circle, and the speed of winding and wrapping depends upon $f$
- To visualize how this winding happens, see the video; it will show the complex graph shape in the complex plane at different frequencies → [3Blue1Brown Video](https://www.youtube.com/watch?v=spUNpyF58BY)

## How to Calculate the COM Coordinates?

Having a previous understanding of how sampling is done for audio processing for Machine Learning/Deep Learning will help you understand better.

So our original sound $g(t)$ which we are taking, as it is a digitally stored signal in a computer, it won't be continuous and we would have sampled points of the original sound. The corresponding same sampled points would be there on the complex plane too $(g(t) \cdot e^{-2\pi itf})$, so that's why we will get points on the complex plane. Joining those points would give us the complex graph which we were talking about above. The more sampled points there are in the original audio, the more corresponding points there would be on the complex plane. It's an obvious thing.

So whatever the number of sampled points we have on the complex plane which are the result of whatever sampled points we have in our digital audio signal g(t), we will give this to our audio signal function → $g(t) \cdot e^{-2\pi itf}$ and plot the output on the complex plane. Now we have to take the COM of all these points. To find the coordinates of COM, the formula would be:

$$X_{\text{coordinate}} = \frac{\sum \text{Real(points)}}{\text{Number of points}}$$

$$Y_{\text{coordinate}} = \frac{\sum \text{Imaginary(points)}}{\text{Number of points}}$$

Now just imagine if this is not done digitally. In that case, we don't need sampled points and we can work on a continuous function. That means we will have infinite continuous points of original audio and corresponding infinite points on the complex plane. Instead of summation, we can integrate, so Fourier is nothing but:

$$\text{FT}(f) = \int_{t_1}^{t_2} g(t) \cdot e^{-2\pi ift} \, dt$$

Integration over Limits → $t_1$ and $t_2$ (time duration of original sound), integration over → $g(t) \cdot e^{-2\pi ft}$, and the output is the complex Fourier transform coefficient for that frequency $f$, which is the COM of the continuous graph.

Now when we get the coordinates of COM, calculate its distance from the origin and multiply the distance with the number of sampled points we have taken from the original audio. This will give the magnitude of the contribution of this frequency $f$ in the original audio.

The purpose of FT was to decompose the original audio signal into its constituent frequencies so that we can know which combination of frequencies the original audio signal is made of. It's analogous to having a bucket in which all colors are mixed and we want to segregate the constituent colors. The bucket mixed with colors is the original audio signal and the constituent colors are the constituent frequencies.

## Phase - The Hidden Variable

Now let's talk about one of the hidden variables, which is the phase. This thing would be better understood if one understands phase and phase difference in terms of waves and sinusoidal waves.

So the COM we get as output from FT is a complex number having a real part, an imaginary part, and also an angle associated with it which would be given by:

$$\text{Phase} = \tan^{-1}\left(\frac{\text{Imaginary}}{\text{Real}}\right)$$

That angle is the phase difference or the phase we are talking about. The magnitude that we got from (COM × Number of time steps) is at this phase difference between the original audio and the wave of frequency $f$.

It's essential to remember that we do this FT for a range of frequencies (many frequencies in terms of digital signal) and continuous in range all real numbers if we're talking in theory.

So for frequencies which are the constituent part in making the original audio signal, it's obvious their magnitude of contribution would be high. But this thing is also obvious: this will happen when they have 0 phase difference between them (the original audio signal and $f$ are in phase or in complete sync with each other). So when a constituent frequency $f$ (with respect to original audio) is given as input to FT, we'll get significantly high magnitude of contribution. The distance of COM of the graph would be far from the origin and the direction of the COM would be such that the phase difference is 0, which means $\tan^{-1}(\text{Img}/\text{Real}) = 0$, so the COM will lie on the real axis.

For other frequencies which are not constituent frequencies of the original audio signal, when they are given as input to FT, there are fair chances that COM doesn't lie on the real axis and $\text{Img(COM)} \neq 0$. It's quite intuitive: when we take a wave of non-constituent frequency, it will adjust itself in a way so that it can sync or contribute at maximum with respect to the original audio. So for them, it's not necessary for them to have a phase difference equal to 0, and often times non-constituent frequencies have non-zero phase difference to give out their complex fourier transform co-efficient, basically the the COM co ordinate have fair chance of having non zero phase differnce angle.

This was the insight related to phase difference on an intuition basis.

## How to Get/Plot the Frequency Domain Graph

This FT would be made for all frequencies in real numbers ideally, and so the graph would include both constituent frequencies and non-constituent frequencies (obviously, the high majority would be non-constituent frequencies, and for constituent frequencies we will get peaks on the graph which would be fewer in number). So at the end, after plotting the frequency domain graph for all frequencies, we will have the magnitude of contribution towards the original audio. The ones which will have the highest contribution are represented by peaks in that graph, and they are the constituent frequencies of the original sound.

This is how we get the frequency domain graph from the time domain graph.

## Now, The WHY?

Apart from visualization which can be done from the 3Blue1Brown video, everything should be clear, although some basic understanding of sound, sound waves, sampling, and complex numbers is required. So why Fourier works to help us achieve our goal is a combination of the magic of Math and Physics. At its core, why it works would require heavy math derivation, which isn't the purpose of this writing work. But **IT WORKS!**