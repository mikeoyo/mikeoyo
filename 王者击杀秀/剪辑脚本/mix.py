"""Audio for the montage: BGM + time-remapped game sound + impact booms."""
import os, sys, wave
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import montage as M

SR = 44100
ROOT = M.ROOT


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
game = np.zeros(N, np.float32)
gain = np.zeros(N, np.float32)
for sg in M.SEGS:
    idx = np.where((t >= sg['t0']) & (t < sg['t1']))[0]
    if not len(idx):
        continue
    s = sg['f'](t[idx]); sp = sg['df'](t[idx])
    src = games[sg['vid']]
    game[idx] = np.interp(s * SR, np.arange(len(src)), src)
    gain[idx] = ((sp > 0.8) & (sp < 1.35)).astype(np.float32)
    # short fade around every cut so the remapped audio never clicks
    edge = int(0.012 * SR)
    gain[idx[:edge]] *= np.linspace(0, 1, min(edge, len(idx)))
    gain[idx[-edge:]] *= np.linspace(1, 0, min(edge, len(idx)))
k = int(0.04 * SR)
gain = np.convolve(gain, np.ones(k) / k, 'same')
game *= gain * 0.22


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
    amp = 0.42 if abs(t0 - M.B(160)) < 0.01 else (0.34 if abs(t0 - M.B(176)) < 0.01 else 0.26)
    b = boom(1.2, amp); i = int(t0 * SR)
    sfx[i:i + len(b)] += b[:max(0, N - i)]

mix = music * 0.88 + (game + sfx)[:, None]
fade = np.clip((M.DUR - 0.05 - t) / 0.85, 0, 1) ** 1.5
fade *= np.clip(t / 0.05, 0, 1)
mix *= fade[:, None]
mix = np.clip(mix, -1, 1)
out = os.path.join(ROOT, 'out/mix.wav')
w = wave.open(out, 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()
print('wrote', out, 'peak', np.abs(mix).max())
