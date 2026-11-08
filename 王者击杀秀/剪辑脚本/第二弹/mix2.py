"""Audio for montage2: the user's BGM + game sound that follows every speed ramp (WSOLA,
pitch kept), mixed so the game sound sits about level with the music."""
import os, sys, wave
import numpy as np
sys.argv = [sys.argv[0], 'none']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import montage2 as M

SR = 44100
ROOT = M.ROOT


def readwav(p):
    w = wave.open(p)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return x.reshape(-1, w.getnchannels())


N = int(M.DUR * SR)
t = np.arange(N) / SR
music = readwav(os.path.join(ROOT, 'audio/bgm.wav'))[:N]
if len(music) < N:
    music = np.vstack([music, np.zeros((N - len(music), 2), np.float32)])
games = {sg['vid']: readwav(os.path.join(ROOT, 'audio/game_%s.wav' % sg['vid']))[:, 0] for sg in M.SEGS}


def wsola():
    L, Ha, tol = 1024, 512, 384
    win = np.hanning(L + 1)[:L].astype(np.float32)
    out = np.zeros(N + L, np.float32)
    prev = None
    for o in range(0, N, Ha):
        tc = (o + L / 2) / SR
        sg = M.find_seg(min(tc, M.DUR - 1e-3))
        src = games[sg['vid']]
        sp = float(sg['df'](tc))
        p = int(round(float(sg['f'](tc)) * SR - L / 2))
        if prev is not None and prev[0] is sg:
            q = prev[1] + Ha
            ref = src[max(q, 0):max(q, 0) + L]
            lo = max(p - tol, 0)
            cand = src[lo:p + tol + L]
            if len(ref) == L and len(cand) >= L:
                p = lo + int(np.argmax(np.correlate(cand, ref, 'valid')))
        g = src[max(p, 0):max(p, 0) + L]
        if len(g) < L:
            g = np.pad(g, (0, L - len(g)))
        out[o:o + L] += win * g * (1.0 if sp > 0.5 else 0.85)
        prev = (sg, p)
    return out[:N]


game = wsola()


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


# game sound about level with the music (music ~0.5 dB on top, as in the last montage)
g_db, m_db = rms_db(game), rms_db(music.mean(1))
music_gain = 10 ** ((g_db - m_db + 0.5) / 20)


def boom(dur, amp):
    tt = np.arange(int(dur * SR)) / SR
    f = 42 + 70 * np.exp(-tt / 0.05)
    b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.33)
    noise = np.convolve(np.random.default_rng(7).standard_normal(len(tt)) * np.exp(-tt / 0.03), np.ones(12) / 12, 'same')
    return amp * (0.9 * b + 0.35 * noise)


sfx = np.zeros(N, np.float32)
for T, kind, slow, vid in M.EVENTS:
    if slow:                       # a soft low hit only under the four slow-mo kills
        b = boom(1.2, 0.18); i = int(T * SR)
        sfx[i:i + len(b)] += b[:max(0, N - i)]

mix = music * music_gain + (game + sfx)[:, None]
fade = np.clip((M.DUR - t) / 0.9, 0, 1) ** 1.5 * np.clip(t / 0.05, 0, 1)
mix *= fade[:, None]
pk = np.abs(mix).max()
mix *= 0.7 / max(pk, 1.0)
w = wave.open(os.path.join(ROOT, 'out/mix.wav'), 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()
open(os.path.join(ROOT, 'out/mix_gain.txt'), 'w').write('%.4f' % (max(pk, 1.0) / 0.7))
print('music gain %.3f (music %.1f dB -> %.1f dB), game %.1f dB, pre-limiter peak %.2f' % (
    music_gain, m_db, m_db + 20 * np.log10(music_gain), g_db, pk))
