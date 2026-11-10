"""A short original music-box tune (7.5 s, loops), with a soft pad underneath."""
import os, sys, wave
import numpy as np

SR = 44100
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 7.5
BPM = 80.0
BEAT = 60 / BPM                      # 0.75 s -> 10 beats in 7.5 s
N = int(DUR * SR)
t = np.arange(N) / SR


def hz(note):
    names = {'C': -9, 'D': -7, 'E': -5, 'F': -4, 'G': -2, 'A': 0, 'B': 2}
    n = names[note[0]] + (1 if '#' in note else 0)
    octave = int(note[-1])
    return 440.0 * 2 ** ((n + 12 * (octave - 4)) / 12)


def music_box(f, dur, amp):
    n = int(dur * SR); tt = np.arange(n) / SR
    env = np.exp(-tt / 0.9) * (1 - np.exp(-tt / 0.002))
    x = (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt / 0.3)
         + 0.12 * np.sin(2 * np.pi * 3.01 * f * tt) * np.exp(-tt / 0.15)
         + 0.05 * np.sin(2 * np.pi * 5.4 * f * tt) * np.exp(-tt / 0.05))
    return amp * env * x


def pad(freqs, start, dur, amp):
    n = int(dur * SR); tt = np.arange(n) / SR
    env = np.minimum(1, tt / 0.6) * np.minimum(1, (dur - tt) / 0.8).clip(0, 1)
    x = sum(np.sin(2 * np.pi * f * tt + 0.3 * np.sin(2 * np.pi * 0.2 * tt)) for f in freqs) / len(freqs)
    return amp * env * x


out = np.zeros(N + SR * 2, np.float32)


def add(x, at):
    i = int(at * SR)
    out[i:i + len(x)] += x


# melody: (note, beats)
mel = [('E5', 1), ('G5', 0.5), ('A5', 0.5), ('G5', 1), ('E5', 1),
       ('D5', 1), ('E5', 0.5), ('G5', 0.5), ('C5', 2),
       ('E5', 0.5), ('D5', 0.5), ('C5', 1)]
pos = 0.25
for note, beats in mel:
    add(music_box(hz(note), 2.2, 0.32), pos * BEAT)
    pos += beats
# a few soft lower notes on the downbeats
for b, note in ((0.25, 'C4'), (4.25, 'A3'), (6.25, 'F3'), (8.25, 'G3')):
    add(music_box(hz(note), 2.5, 0.16), b * BEAT)
# pad chords
for b, ch, l in ((0, ('C4', 'E4', 'G4'), 4), (4, ('A3', 'C4', 'E4'), 2), (6, ('F3', 'A3', 'C4'), 2),
                 (8, ('G3', 'B3', 'D4'), 2)):
    add(pad([hz(n) for n in ch], 0, l * BEAT + 0.6, 0.07), b * BEAT)

# fold the tail back to the start so the clip loops without a gap
tail = out[N:N + SR * 2].copy()
out = out[:N]
out[:len(tail)] += tail[:N]
# gentle room tone
out += np.random.default_rng(3).normal(0, 0.0025, N).astype(np.float32)
out /= np.abs(out).max() / 0.7
st = np.stack([out, out], 1)
w = wave.open(sys.argv[1], 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((st * 32767).astype(np.int16).tobytes()); w.close()
print('wrote', sys.argv[1])
