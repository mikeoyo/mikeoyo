"""Audio for montage4: the DJ track + game sound that follows every speed ramp (WSOLA, pitch
kept), level with the music; softer under the slow-mo memory and the ending. Whooshes on whip
cuts, a swell into the white-out, a soft low hit under the two long kill close-ups."""
import os, sys, wave
import numpy as np
sys.argv = [sys.argv[0], 'none']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import montage4 as M

SR = 44100
N = int(M.DUR * SR)
t = np.arange(N) / SR
rng = np.random.default_rng(11)


def readwav(p):
    w = wave.open(p)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return x.reshape(-1, w.getnchannels())


music = readwav(os.path.join(M.HERE, 'audio/song.wav'))[:N]
if len(music) < N:
    music = np.vstack([music, np.zeros((N - len(music), 2), np.float32)])
games = {sg['vid']: readwav(os.path.join(M.HERE, 'audio/game_%s.wav' % sg['vid']))[:, 0] for sg in M.SEGS}


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
        gain = 1.0 if sp > 0.5 else 0.8
        if sg['look'] == 'memory':
            gain *= 0.45
        if sg['look'] == 'outro' and tc > M.FINAL + 0.5:
            gain *= max(0.35, 1 - (tc - M.FINAL - 0.5) / 2.5)
        out[o:o + L] += win * g * gain
        prev = (sg, p)
    return out[:N]


game = wsola()
# short fades at every cut so the game sound never clicks
for sg in M.SEGS[1:]:
    i = int(sg['t0'] * SR); k = int(0.015 * SR)
    game[max(0, i - k):i] *= np.linspace(1, 0, min(k, i))
    game[i:i + k] *= np.linspace(0, 1, k)[:len(game[i:i + k])]


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


nm = (t > 6.2) & (t < 78.0)
g_db, m_db = rms_db(game[nm]), rms_db(music.mean(1)[nm])
music_gain = 10 ** ((g_db - m_db + 0.5) / 20)


def bandnoise(dur, f0, f1, amp, rise):
    n = int(dur * SR)
    x = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, np.float32); blk = 2048
    for s0 in range(0, n, blk // 2):
        seg = x[s0:s0 + blk]
        if len(seg) < 32:
            break
        fc = f0 * (f1 / f0) ** (s0 / n)
        S = np.fft.rfft(seg * np.hanning(len(seg)))
        ff = np.fft.rfftfreq(len(seg), 1 / SR)
        S *= np.exp(-0.5 * (np.log(np.maximum(ff, 1) / fc) / 0.5) ** 2)
        out[s0:s0 + len(seg)] += np.fft.irfft(S, len(seg)).astype(np.float32)
    tt = np.linspace(0, 1, n)
    env = np.where(tt < rise, (tt / rise) ** 2, np.exp(-(tt - rise) / 0.12))
    out *= env
    return amp * out / (np.abs(out).max() + 1e-9)


def boom(dur, amp):
    tt = np.arange(int(dur * SR)) / SR
    f = 42 + 70 * np.exp(-tt / 0.05)
    return amp * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.33)


sfx = np.zeros(N, np.float32)


def put(x, at):
    i = int(at * SR)
    if i < 0:
        x = x[-i:]; i = 0
    sfx[i:i + len(x)] += x[:max(0, N - i)]


for tc, kind in M.CUTS:
    if kind == 'whip':
        put(bandnoise(0.34, 600, 4000, 0.14, 0.5), tc - 0.17)
    elif kind == 'white':
        put(bandnoise(0.75, 300, 6000, 0.15, 0.92), tc - 0.69)
for t0, hold in M.GOD_T:
    put(bandnoise(0.22, 3000, 900, 0.07, 0.15), t0 - 0.03)          # small swish into each close-up
    if hold > 0.4:
        put(boom(1.2, 0.2), t0)

mix = music * music_gain + (game + sfx)[:, None]
fade = np.clip((M.DUR - t) / 1.2, 0, 1) ** 1.3 * np.clip(t / 0.05, 0, 1)
mix *= fade[:, None]
pk = np.abs(mix).max()
mix *= 0.7 / max(pk, 1.0)
os.makedirs(os.path.join(M.HERE, 'out'), exist_ok=True)
w = wave.open(os.path.join(M.HERE, 'out/mix.wav'), 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()
open(os.path.join(M.HERE, 'out/mix_gain.txt'), 'w').write('%.4f' % (max(pk, 1.0) / 0.7))
print('music gain %.3f, game %.1f dB, music %.1f dB -> %.1f dB, pre-limiter peak %.2f' % (
    music_gain, g_db, m_db, m_db + 20 * np.log10(music_gain), pk))
