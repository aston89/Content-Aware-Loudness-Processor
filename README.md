# Content-Aware Loudness Processor 

**CALP** is an offline audio processing concept and reference implementation designed to make loudness adjustment more musical, more predictable and less destructive than traditional “stupid” normalization or aggressive automatic gain control.
Instead of blindly pushing every quiet moment toward a fixed target, it analyzes the structure of the audio first and then shapes loudness according to what the signal actually contains:
- how loud the material is overall
- how dynamic it is
- how dense the transient content is
- how much crest factor it has
- how much energy is concentrated in different frequency regions

The result is a processed file that keeps its identity but lands closer to a useful broadcast-ready loudness.

---

## Why CALP exists
Anyone who has ever run old recordings, lo-fi material, podcasts or highly dynamic music through a simple normalizer knows the problem:
- quiet passages get boosted too much
- whispered or sparse sections become unnaturally loud
- sudden drums, hits, or transients can break the result
- the processing is technically “correct” but musically wrong

**This project aims to solve them all:**
The goal is not to flatten everything into a generic target but to **preserve proportions** while still moving the material toward a consistent targeted output loudness making it especially interesting for broadcast prep, archival content, older albums and general-purpose audio cleanup where the source material has wide dynamic variation.

---

## How CALP differ from a classic normalizer, compressor or limiter ?

### A normalizer
A normalizer usually measures the file globally and applies a single gain value or it works with simple loudness rules. That is fine for uniform material but it can behave badly on music with big internal contrasts.

### A compressor
A compressor reacts to level over time and reduces dynamic range once a threshold is crossed. That can work well but it's still mostly level-driven, it does not automatically *understand* whether a quiet section is a delicate intro, a whispered vocal or part of a highly contrasty arrangement.

### A limiter
A limiter is primarily a protection device, it prevents peaks from exceeding a ceiling, it's excellent at safety but by itself it does not decide *how* the audio should be shaped before that ceiling is reached.

### CALP combines several ideas into one coherent system:
- **perceptual weighting** before analysis
- **content-aware gain shaping** instead of blunt level chasing
- **curve-based decisions** instead of hard on/off thresholds
- **multiband processing** so different frequency regions can behave differently
- **final peak protection** with a limiter stage

So instead of just looking at “how loud is”, looks at “what kind of loudness is”
That is the whole point.

---

## What CALP does ?
At a high level, the processor:
1. Loads an audio file offline.
2. Analyzes the signal in windows.
3. Estimates loudness, crest factor, and transient density.
4. Applies smooth, continuous gain curves.
5. Splits the signal into frequency bands and processes each one gently.
6. Recombines the bands.
7. Applies final peak safety.
8. Exports a file that sounds closer to the source, but more manageable in level.

The intent is not to “master” the material in a commercial sense.
The intent is to produce a **ready-to-go** file for a broadcast-oriented pipeline.

---

## How CALP works ?
The implementation is built around a few ideas:

### 1. Perceptual weighting
Before making decisions, the signal is weighted in a way that better reflects perceived loudness than raw sample amplitude alone.
This helps avoid overreacting to material that is technically loud in the waveform but not particularly loud to the ear.

### 2. Loudness and structure analysis
The processor looks at:
- integrated loudness estimate
- local loudness windows
- crest factor
- transient activity

This is what makes it content-aware, a sparse passage is not treated the same way as a dense drum-heavy section.

### 3. Curve-based gain shaping
Instead of using brittle `if/else` logic, the gain is controlled by smooth curves.
That means:
- no hard switching
- no sudden behavior jumps
- no aggressive pumping from simple threshold crossings

The gain moves like a continuous control surface, not a binary gate.

### 4. Multiband response
A low-frequency-heavy section does not behave exactly like a vocal-heavy midrange section or a bright transient-rich section.
The processor splits the audio into bands and gives each region its own slightly different behavior.
That helps keep the response more coherent across different genres and source types.

### 5. Final limiter stage
After the shaping stage, a limiter protects the output from overshooting.
This is the final safety net, not the main loudness strategy.

---

## What makes CALP useful for radio broadcasting ?
This type of processing is especially useful when preparing audio that will later be mixed, sequenced, or played in a broadcast context.
Ideal use cases include:
- radio music libraries
- pre-broadcast content prep
- archival music collections
- older albums with inconsistent loudness
- lo-fi or indie material that needs gentle leveling
- spoken-word or mixed-content libraries where a natural feel matters

In a radio workflow, the goal is often not maximum loudness but a **consistent, predictable loudness without wrecking the character of the material**.
This makes CALP a good pre-stage before the file enters a broader broadcast chain.

---

## Why CALP it's not just another loudness tool ?
Most loudness tools are built around one of these ideas:
- global normalization
- compression
- peak limiting
- broadcast metering
- automatic gain control

**CALP tries to combine the useful pieces while avoiding the usual failure mode of each one.**
It does not simply chase a target.
It does not simply squash peaks.
It does not blindly boost quiet sections.
Instead, it tries to preserve the musical proportions inside the file while still making the overall result more broadcast-friendly.

---

## Design philosophy:
The project is based on a simple belief:
> A loudness processor should understand the shape of the content, not just its level.
That means the system should behave more like a careful engineer than a brute-force normalizer.
The ideal result is not a “louder” file in the cheap sense.
The ideal result is a file that feels controlled, coherent, and ready to move through a broadcast chain without nasty surprises.

---

## Current status:
This project is a **reference implementation / prototype**.
It is intentionally understandable and hackable like a solid experimental base that people can:
- run as-is
- study
- modify
- improve
- port into other environments

**It is not meant to be a black box.**

---

## Future directions
Possible upgrades include:
- a more exact LUFS / EBU R128 measurement path
- better transient detection
- smarter multiband rules
- adaptive parameter tuning
- optional machine-learning-based parameter suggestion
- a GUI wrapper or plugin version

---

## Acknowledgements
This project draws inspiration from ideas found in loudness normalizations, broadcast metering, multiband processing and classic dynamic range control but combines them into a single, content-aware offline workflow.

---

## Installation

install the required Python dependencies:

```bash/cmd
pip install numpy scipy soundfile pyloudnorm
```
or
```bash/cmd
pip install -r requirements.txt
```

If you want full reproducibility, create a virtual environment first:

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

---

### Basic usage (cli)

```
python CALP_DSP.py input.wav output.wav --target-lufs -14
```

### Parameters

- `input` → input audio file (WAV recommended)
- `output` → processed output file
- `--target-lufs` → loudness target (default: -14 LUFS)
- `--verbose` → output more info

### Example workflows

#### 1. Radio prep (balanced loudness)
```bash
python CALP_DSP.py track.wav track_broadcast.wav --target-lufs -14
```

#### 2. More aggressive leveling
```bash
python CALP_DSP.py track.wav track_loud.wav --target-lufs -10
```

#### 3. Safer archival processing
```bash
python CALP_DSP.py old_song.wav cleaned.wav --target-lufs -16
```

---

## Expected behavior
After processing, you should expect:
- Slight reduction in extreme dynamic swings
- More consistent perceived loudness
- Preservation of musical structure
- Reduced need for manual normalization or compression
- A final limiter stage preventing clipping

---

## Notes
- This tool is **offline only** (not real-time)
- It is optimized for full-track processing, not live input
- Results depend heavily on source material (lo-fi, dynamic recordings, modern mastered tracks will behave differently)
- Heavy ram usage, single core, not optimized for performance, with a 5min audio file it eats up to 4gb of ram.
- Output file will be 44100hz / 16bit

---

## Suggested workflow integration

This tool works best as a **pre-broadcast or pre-mastering stage**.
It is not intended to replace a mastering engineer but to make material more consistent before it enters a broadcast pipeline.

---

## Emergent behavior under extreme gain targets (experimental notes)

During testing, CALP was intentionally driven beyond typical mastering ranges like -6 LUFS region and below in order to observe failure modes and limiter interaction under extreme conditions.

What emerged was not a traditional collapse or hard clipping behavior but rather a gradual transition of the system’s operating regime.
- **At moderate targets** (≈ -16 to -12 LUFS), the processor behaves as intended: a content-aware loudness shaper that preserves the original dynamic structure - while improving perceived loudness consistency.

- **As the target approaches more aggressive loudness values** (≈ -10 to -8 LUFS), the limiter begins to play a more active role, but remains secondary to the gain-shaping logic.

- **Beyond this region** (≈ -7 LUFS and higher loudness demands), the system transitions into a ***hybrid state** where peak protection and loudness shaping are no longer clearly separable: In this regime, the interaction between band-aware gain control, transient suppression and true-peak limiting produces a form of **controlled, distributed saturation**, this is not an explicitly designed “coloration stage" but rather an **emergent consequence** of overlapping constraints competing for headroom, this behavior resemble a **tape-like or analog saturation character like IVGI2 or SSL4000 vst's** but is fundamentally different from a static non-linear processors: the distortion is time-dependent, context-aware and dynamically allocated across frequency bands rather than applied as a single transfer curve.

**At extreme lufs settings the system no longer behaves as a transparent loudness corrector**, instead, it enters a regime best described as constraint-driven reshaping where the limiter is continuously engaged and the gain model adapts in real time to preserve intelligibility and prevent catastrophic clipping.

**This behavior is not the primary design goal** and should not be considered a feature in the conventional sense.
However, it is a predictable outcome of the architecture under extreme input conditions and **may be of interest for those who like lo-fi or music creators**  where controlled non-linearity is desirable.

