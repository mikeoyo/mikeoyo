"""Audio for the montage: BGM + game sound that follows every speed ramp + impact booms.

The game sound is time-stretched with WSOLA (waveform-similarity overlap-add),
so it keeps its normal pitch at 0.2x slow-mo and at 2x fast-forward alike, and
plays under the whole video instead of only the real-time stretches.
"""
import os, sys, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import montage as M

SR = 44100
ROOT = M.ROOT
GAME_GAIN = 0.85     # game sound about as loud as the music
MUSIC_GAIN = 0.72


def readwav(p):
    w = wave.open(p)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return x.reshape(-1, w.getnchannels())


N = int(M.DUR * SR)
t = np.arange(N) / SR
music = readwav(os.path.join(ROOT, 'audio/song_full.wav'))[:N]
if len(music) < N:
    music = np.vstack([music, np.zeros((N - len(music), 2), np.float32)])

games = {v: readwav(os.path.join(ROOT, f'audio/game_{v}.wav'))[:, 0] for v in ('345', '346', '347', '348')}


def wsola():
    L, Ha, tol = 1024, 512, 384
    win = np.hanning(L + 1)[:L].astype(np.float32)        # periodic Hann: sums to 1 at 50% overlap
    out = np.zeros(N + L, np.float32)
    prev = None                                          # (vid, start) of previous grain
    for o in range(0, N, Ha):
        tc = (o + L / 2) / SR
        sg = M.find_seg(min(tc, M.DUR - 1e-3))
        src = games[sg['vid']]
        sp = float(sg['df'](tc))
        p = int(round(float(sg['f'](tc)) * SR - L / 2))
        if prev is not None and prev[0] is sg:
            q = prev[1] + Ha                             # natural continuation of last grain
            ref = src[max(q, 0):max(q, 0) + L]
            lo = max(p - tol, 0)
            cand = src[lo:p + tol + L]
            if len(ref) == L and len(cand) >= L:
                c = np.correlate(cand, ref, 'valid')
                p = lo + int(np.argmax(c))
        g = src[max(p, 0):max(p, 0) + L]
        if len(g) < L:
            g = np.pad(g, (0, L - len(g)))
        gain = 1.0 if sp > 0.5 else 0.85                 # a touch lower in deep slow-mo
        out[o:o + L] += win * g * gain
        prev = (sg, p)
    return out[:N]


game = wsola() * GAME_GAIN


def boom(dur, amp):
    tt = np.arange(int(dur * SR)) / SR
    f = 42 + 70 * np.exp(-tt / 0.05)
    ph = 2 * np.pi * np.cumsum(f) / SR
    b = np.sin(ph) * np.exp(-tt / 0.33)
    noise = np.random.default_rng(7).standard_normal(len(tt)) * np.exp(-tt / 0.03)
    noise = np.convolve(noise, np.ones(12) / 12, 'same')
    return amp * (0.9 * b + 0.35 * noise)


sfx = np.zeros(N, np.float32)
for t0 in M.IMPACTS:
    amp = 0.32 if abs(t0 - M.FINAL_KILL) < 0.01 else 0.2
    b = boom(1.2, amp); i = int(t0 * SR)
    sfx[i:i + len(b)] += b[:max(0, N - i)]

mix = music * MUSIC_GAIN + (game + sfx)[:, None]
fade = np.clip((M.DUR - 0.05 - t) / 0.85, 0, 1) ** 1.5
fade *= np.clip(t / 0.05, 0, 1)
mix *= fade[:, None]
pk = np.abs(mix).max()
mix *= 0.7 / max(pk, 1.0)   # headroom; ffmpeg brings the level back up through a limiter
print('pre-limiter peak %.2f' % pk)
open(os.path.join(ROOT, 'out/mix_gain.txt'), 'w').write('%.4f' % (max(pk, 1.0) / 0.7))
assert np.abs(mix).max() < 1.0
out = os.path.join(ROOT, 'out/mix.wav')
w = wave.open(out, 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()
gr = 20 * np.log10(np.sqrt(np.mean(game ** 2)) + 1e-9)
mr = 20 * np.log10(np.sqrt(np.mean((music * MUSIC_GAIN) ** 2)) + 1e-9)
print('wrote', out, 'peak %.2f' % np.abs(mix).max(), 'music rms %.1f dB, game rms %.1f dB' % (mr, gr))
