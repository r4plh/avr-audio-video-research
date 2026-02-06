# SyncNet: AI-Powered Audio-Visual Synchronization Without Manual Labels
## A Deep Dive into "Out of Time: Automated Lip Sync in the Wild"

## Introduction

Ever watched a badly dubbed movie where the lips don't match the words? Or been on a video call where someone's mouth moves out of sync with their voice? These sync issues are more than just annoying - they're a real problem in video production, broadcasting, and real-time communication. The Syncnet paper tackles this head-on with a clever self-supervised approach that can automatically detect and fix audio-video sync problems without needing any manual annotations. What's particularly cool is that the same model that fixes sync issues can also figure out who's speaking in a crowded room - all by learning the natural correlation between lip movements and speech sounds.

## Core Applications

The downstream tasks which can be performed with the output of the trained ConvNet have vital applications which include determining the lip-sync error in videos, detecting the speaker in a scene with multiple faces, and lip reading. Developing the lip-sync error application, if the sync offset is present in the -1 to +1 second range (this range could be varied, but generally it suffices for TV broadcast audio-video) - that is, video lags audio or vice-versa in -1 to +1 second - we can determine how much the offset is. For example, say it comes out to be 200 ms audio lags video, that means video is 200 ms ahead of audio, and in that case we can shift the audio 200 ms forward and can make the offset sync issue near 0 by this, so it also has applications to make audio-video in sync (if offset lies in the range we have taken here -1 to +1 seconds).

## Self-Supervised Training Approach

The training method is self-supervised, which means no human annotations or manual labelling is required; the positive and negative pairs for training the model happen without manual labelling. This method assumes the data we get is already in sync (audio-video in sync), so we get the positive pair already in which the audio and video are in sync, and we make the false pair in which audio and video are not in sync by shifting the audio by ± some seconds to make it async (false pair for training network). The advantage is that we can have almost an infinite amount of data to train, provided it is synced and already has no sync issue in source, so that positive and negative pairs can be made easily for training.

## Network Architecture: Dual-Stream CNN

Coming to the architecture, it has 2 streams: audio stream and video stream, or in layman's terms, the architecture is divided into 2 branches - one for audio and one for video. Both streams expect 0.2 second input. The audio stream expects 0.2 seconds of audio and the video stream expects 0.2 second video.
Both network architectures for audio and video streams are CNN-based, which expect 2D data. For video (frames/images), CNN seems natural, but for audio, a CNN-based network is also trained. For both video and the corresponding audio, first their respective data preprocessing is done and then they are fed into their respective CNN architectures.

### Audio Data Preprocessing

Audio data preprocessing - The raw 0.2 second audio goes through a series of steps to give an MFCC 13 x 20 matrix. 13 are the DCT coefficients associated for that audio chunk which represent the features for that audio, and 20 is in the time direction, because MFCC frequency was 100 Hz, so for 0.2 seconds, 20 samples will be there and each sample's DCT coefficients are represented by one column of the 13 x 20 matrix. The matrix 13 x 20 is input to the CNN audio stream. Output of the network is a 256-dimensional embedding, representation of the 0.2 second audio.

### Video Data Preprocessing

Video preprocessing - The CNN here expects input of 111×111×5 (W×H×T), 5 frames of (h,w) = (111,111) gray-scale image of mouth. Now for 25 fps, 0.2 seconds translates to 5 frames. The raw video of 0.2 seconds goes through video preprocessing at 25 fps and gets converted into 111x111x5 and fed into the CNN network. Output of the network is a 256-dimensional embedding, representation of the 0.2 second video.

The audio preprocessing is simpler and less complex than video. Let's understand how the 0.2 second video and its corresponding audio are chosen from the original source. Our goal is to get a video clip where there is only 1 person and no scene change should be there in the 0.2 second, one person who's speaking in the 0.2 second duration. Anything other than this is bad data for our model at this stage. So we run video data preprocessing on the video, in which we do scene detection, then face detection, and then face tracking, crop the mouth part and convert all frames/images for the video into grayscale images of 111x111 and give it to the CNN model, and the corresponding audio part is converted into a 13x20 matrix and given to the audio CNN. The clips where faces > 1 are rejected; there's no scene change in 0.2 seconds clip as we have applied scene detection in the pipeline. So what we have at last is a video in which audio is there and one person is there in video, which suffices the primary need of the data pipeline.

## Joint Embedding Space Learning

The network learns a joint embedding space, which means the audio embedding and video embedding will be plotted in a common embedding space. The joint embedding space will have those audio and video embeddings close to each other which are in sync, and those audio and video embeddings will be far apart in the embedding space which are not in sync, that's it.
The Euclidean distance between synced audio and video embeddings will be less and vice-versa.

### Loss Function and Training Refinement

The loss function used is contrastive loss. For a positive pair (sync audio-video 0.2 second example), the square of Euclidean distance between audio and video embedding should be minimum; if that comes high, a penalty would be imposed, so for positive pairs, the square of Euclidean distance is minimised, and for negative pairs, max(margin - Euclidean distance, 0)² is minimised.

We refine the training data by removing the false positives from our data. Our data still contains false positives (noisy data), and we remove the false positives by initially training the syncnet on noisy data and removing those positive pairs (which are marked as synced audio-video positive pairs) which fail to pass a certain threshold. The noisy data (false positives) are there maybe because of dubbed video, someone else speaking over the speaker from behind, off-sync maybe present, or these types of things get filtered out in this refining step.

## Inference and Applications

Now the network gets trained, so let's talk about inference and experimentation results derived from the trained model.

There is test data in which positive and negative pairs for audio-video are present, so our model's inference should give low value (min Euclidean distance) for positive pairs in the test data and high value (max Euclidean distance) for negative pairs in the test data. This is one kind of experiment or inference result of our model.

Determining the offset is also one kind of experiment, or can say one kind of application by our trained model inference. The output would be the offset, like for example audio leads by 200 ms or video leads by 170 ms - determining the sync offset value for which video or audio lags. That means by adjusting the offset determined by the model should fix the sync issue and should make the clip in-sync from off-sync.
If by adjusting the audio-video by the offset value fixes the sync issue, that means success; otherwise failure of the model (provided the synced audio to be present for that one fixed video in the range we are calculating the Euclidean distance between fixed video (0.2 s) and various audio chunks (0.2 s each, sliding over some -x to +x seconds range, x = 1s)). This sync offset for the source clip could be either determined by calculating the sync offset value for 1 single 0.2 second video from source clip, or it could be determined by doing an average over several 0.2 second samples from the source clip and then give the averaged offset sync value. The latter would be more stable than the former, also being proved by test data benchmarks that taking average is the more stable and better way to tell the sync offset value. There's a confidence score associated with this offset number which the model gives, which is termed as AV sync confidence score. For example, it would be said like the source clip has an offset, audio leads video by 300 ms with a confidence score of 11. So knowing how this confidence score is calculated is important, and let's understand it with an example.

## Practical Example: Offset and Confidence Score Calculation

Let's say we have a source clip of 10 seconds and we know this source clip has sync offset in which audio leads video by 300 ms.
Now we'll see how our syncnet is used to determine this offset.

We take ten 0.2 s videos as v1, v2, v3…….. v10.

Let's understand how sync score and confidence score will be calculated for v5, and similar will happen with all 10 video bins/samples/chunks.

Source Clip: 10 seconds total

```
v1: 0.3-0.5s    [--]
v2: 1.2-1.4s        [--]
v3: 2.0-2.2s            [--]
v4: 3.1-3.3s                [--]
v5: 4.5-4.7s                    [--]
v6: 5.3-5.5s                        [--]
v7: 6.6-6.8s                            [--]
v8: 7.4-7.6s                                [--]
v9: 8.2-8.4s                                    [--]
v10: 9.0-9.2s                                       [--]
```

Let's take v5 as one fixed video of 0.2s duration. Now using our trained syncnet model, we'll calculate the Euclidean distance between several audio chunks (will use a sliding window approach) and this fixed video chunk. Here's how:

The audio sampling for v5 will happen from 3.5s to 5.7s (±1s of v5),
which gives us a 2200ms (2.2 second) range to search.

With overlapping windows:
- Window size: 200ms (0.2s)
- Hop length: 100ms
- Number of windows: 21

```
Window 1:  3500-3700ms → Distance = 14.2
Window 2:  3600-3800ms → Distance = 13.8
Window 3:  3700-3900ms → Distance = 13.1
………………..
Window 8:  4200-4400ms → Distance = 2.8  ← MINIMUM (audio 300ms early)
Window 9:  4300-4500ms → Distance = 5.1
………………..
Window 20: 5400-5600ms → Distance = 14.5
Window 21: 5500-5700ms → Distance = 14.9
```

Sync offset for v5 = -300ms (audio leads video by 300ms)
Confidence_v5 = median(~12.5) - min(2.8) = 9.7

So the confidence score for v5 for 300 ms offset is 9.7, and this is how confidence score given by syncnet is calculated, which is equal to median(over all windows or audio bins) - min(over all windows or audio bins) for the fixed v5.

Similarly, every other video bin has an offset value with an associated confidence score.

```
v1 (0.3-0.5s):   Offset = -290ms, Confidence = 8.5
v2 (1.2-1.4s):   Offset = -315ms, Confidence = 9.2  
v3 (2.0-2.2s):   Offset = 0ms,    Confidence = 0.8  (silence period)
v4 (3.1-3.3s):   Offset = -305ms, Confidence = 7.9
v5 (4.5-4.7s):   Offset = -300ms, Confidence = 9.7
v6 (5.3-5.5s):   Offset = -320ms, Confidence = 8.8
v7 (6.6-6.8s):   Offset = -335ms, Confidence = 10.1
v8 (7.4-7.6s):   Offset = -310ms, Confidence = 9.4
v9 (8.2-8.4s):   Offset = -325ms, Confidence = 8.6
v10 (9.0-9.2s):  Offset = -295ms, Confidence = 9.0
```

Averaging (ignoring low confidence v3):
(-290 - 315 - 305 - 300 - 320 - 335 - 310 - 325 - 295) / 9 = -305ms

Or if including all 10 with weighted averaging based on confidence:
Final offset ≈ -300ms (audio leads video by 300ms) → This is how offset is calculated for the source clip.

**Important note** - Either do weighted avg based on confidence score or remove the ones which have low confidence, because not doing so will give:

Simple Average (INCLUDING silence) - WRONG: (-290 - 315 + 0 - 305 - 300 - 320 - 335 - 310 - 325 - 295) / 10 = -249.5ms
This is way off from the true 300ms!

This shows why the paper achieves 99% accuracy with averaging but only 81% with single samples. Proper confidence-based filtering/weighting eliminates the misleading silence samples.

## Speaker Identification in Multi-Person Scenes

One more important application of sync score is speaker identification in multi-person scenes. When multiple faces are visible but only one person's audio is heard, syncnet computes the sync confidence for each face against the same audio stream. Instead of sliding audio temporally for one face, we evaluate all faces at the same time point - each face's mouth movements are compared with the current audio to generate confidence scores. The speaking face naturally produces high confidence (strong audio-visual correlation) while silent faces yield low confidence (no correlation). By averaging these measurements over 10-100 frames, transient errors from blinks or motion blur get filtered out, similar to how silence periods were handled in sync detection.

## Conclusion

Syncnet demonstrates that sometimes the best solutions come from rethinking the problem entirely. Instead of requiring tedious manual labeling of sync errors, it cleverly uses the assumption that most video content starts out correctly synced - turning regular videos into an unlimited training dataset. The beauty lies in its simplicity: train two CNNs to create embeddings where synced audio-video pairs naturally cluster together. With 99% accuracy when averaging over multiple samples and the ability to handle everything from broadcast TV to wild YouTube videos, this approach has proven remarkably robust. Whether you're fixing sync issues in post-production or building the next video conferencing app, the principles behind Syncnet offer a practical blueprint for solving real-world audio-visual alignment problems at scale.

## Sources and Implementation

- **GitHub Implementation**: [https://github.com/joonson/syncnet_python](https://github.com/joonson/syncnet_python) - Python implementation of the SyncNet model
- **Official Project Page**: [https://www.robots.ox.ac.uk/~vgg/software/lipsync/](https://www.robots.ox.ac.uk/~vgg/software/lipsync/) - Original research from Visual Geometry Group, University of Oxford