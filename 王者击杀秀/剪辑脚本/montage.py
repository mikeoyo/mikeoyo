"""王者荣耀击杀秀 — beat-synced kill montage renderer (game footage only, no text).

Every cut, kill and effect is placed on the beat grid of the BGM
(如果可以 DJ版, 145.2 BPM after the 1.1x speed-up in the Douyin clip).
Source time is mapped from output time with monotone PCHIP curves through
(beat, source-second) anchors: smooth speed ramps that land each kill on a beat.
The 30 fps recordings are optical-flow interpolated (补帧) to 60 fps, which also
keeps the deep slow-motion smooth.
"""
import os, sys, math, subprocess
import numpy as np, cv2
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'cache')
FPS = 60                           # output
SFPS = 30                          # source recordings
W, H = 1920, 1080
SW, SH = 1280, 720
P, OFF = 0.41320, 0.0108          # beat period and first-beat offset (s)
DUR = 79.2                         # song is cut hard at 79.3 in the source clip
NFR = {'345': 1109, '346': 1330, '347': 1995, '348': 3102}


def B(b):
    return OFF + P * b


def TB(t):
    return (t - OFF) / P


# ------------------------------------------------------------------ timeline
SEGS = []  # dict(vid, anchors[(beat, src)], style, cut)


def seg(vid, anchors, style='normal', cut='blur'):
    SEGS.append(dict(vid=vid, anchors=anchors, style=style, cut=cut))


# intro: engage fading up from black, then four 1-beat flash teasers, all at real speed
seg('348', [(TB(0.0), 75.4), (4, 77.05)], style='intro', cut='hard')
seg('348', [(4, 88.3), (5, 88.71)], style='teaser', cut='flash')
seg('345', [(5, 14.2), (6, 14.61)], style='teaser', cut='flash')
seg('346', [(6, 10.3), (7, 10.71)], style='teaser', cut='flash')
seg('348', [(7, 84.2), (8, 84.61)], style='teaser', cut='flash')


def single(vid, k, s, slow):
    if slow:   # the flashiest kills: rush in, short slow-mo on the hit, snap back
        seg(vid, [(s, k - 2.5), (s + 4, k), (s + 5.25, k + 0.15), (s + 8, k + 1.45)])
    else:      # rush in at ~1.6x and straight through the kill on the bar downbeat
        seg(vid, [(s, k - 2.7), (s + 4, k), (s + 8, k + 2.0)])


# section A (beats 8-64): seven solo kills on bar downbeats; slow-mo only on the two
# biggest-looking ones (347 52s red explosion, 348 4s 横扫千军 burst)
for i, (vid, k, slow) in enumerate([('347', 5.27, False), ('347', 29.27, False), ('347', 52.27, True),
                                    ('347', 59.27, False), ('348', 4.17, True), ('348', 10.07, False),
                                    ('348', 47.77, False)]):
    single(vid, k, 8 + 8 * i, slow)

# section B (beats 64-124): the two triple kills in real time, short slow-mo on each third kill
seg('346', [(64, 1.6), (72, 4.97), (80, 8.87), (83, 9.99), (84, 10.37), (85.25, 10.52), (92, 13.4)], cut='flash')
seg('345', [(92, 4.4), (100, 7.77), (103, 9.07), (115, 13.86), (116, 14.27), (117.25, 14.42), (124, 17.0)])
# section C (beats 124-end): the five-kill team fight in real time; only the last kill
# (the team wipe) drops into slow-mo, which carries through the ending
seg('348', [(124, 70.5), (128, 73.5), (150, 82.57), (154, 84.27), (160, 86.87), (164, 88.37),
            (179, 94.26), (180, 94.67), (183, 95.0), (TB(DUR) + 0.05, 96.3)], style='finale')

for sg in SEGS:
    bs = np.array([a[0] for a in sg['anchors']], float)
    ss = np.array([a[1] for a in sg['anchors']], float)
    sg['t0'], sg['t1'] = B(bs[0]), B(bs[-1])
    sg['f'] = PchipInterpolator(B(bs), ss)
    sg['df'] = sg['f'].derivative()
SEGS[0]['t0'] = 0.0
SEGS[-1]['t1'] = DUR + 1

# events --------------------------------------------------------------------
PUNCH = []    # (T, strength)
FLASH = []    # (T, amount)
for i in range(7):
    PUNCH.append((B(12 + 8 * i), 1.15))
for b in (4, 5, 6, 7):
    PUNCH.append((B(b), 1.2))
# multi-kill beats (2nd/3rd/4th/5th kill of a streak hit harder)
MULTI = [(B(80), 1.3), (B(84), 1.6), (B(103), 1.3), (B(116), 1.6),
         (B(154), 1.3), (B(160), 1.5), (B(164), 1.8), (B(180), 2.0)]
PUNCH += [(B(72), 1.1), (B(100), 1.1), (B(150), 1.15)] + MULTI
IMPACTS = [t for t, _ in MULTI]          # sub-boom SFX in the audio mix
FINAL_KILL = B(180)
# white flash only on five moments: the two drops, the two triple finishers, the final kill
FLASH += [(B(64), 0.5), (B(128), 0.5), (B(84), 0.3), (B(116), 0.3), (FINAL_KILL, 0.4)]
DROPS = [B(64), B(128)]
CUTS = [(sg['t0'], sg['cut']) for sg in SEGS[1:]]
RISERS = [(B(60), B(64)), (B(124), B(128))]


# ------------------------------------------------------------------ helpers
def find_seg(T):
    for sg in SEGS:
        if sg['t0'] <= T < sg['t1']:
            return sg
    return SEGS[-1]


def env_punch(t, decay=0.22, attack=0.033):
    if t < 0:
        return 0.0
    if t < attack:
        return t / attack
    return math.exp(-(t - attack) / decay)


class Ctx:
    pass


C = Ctx()


def init_worker():
    C.frames = {}
    C.flows = {}
    C.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    C.vign = np.clip(1.0 - 0.38 * np.clip(r - 0.5, 0, None) ** 1.5, 0.55, 1)[..., None].astype(np.float32)
    C.sgx, C.sgy = np.meshgrid(np.arange(SW, dtype=np.float32), np.arange(SH, dtype=np.float32))


def load(vid, i):
    i = int(min(max(i, 0), NFR[vid] - 1))
    key = (vid, i)
    im = C.frames.get(key)
    if im is None:
        im = cv2.imread(os.path.join(CACHE, vid, '%05d.jpg' % i))
        if len(C.frames) > 24:
            C.frames.pop(next(iter(C.frames)))
        C.frames[key] = im
    return im


def flow(vid, i, j):
    key = (vid, i, j)
    f = C.flows.get(key)
    if f is None:
        a = cv2.cvtColor(cv2.resize(load(vid, i), (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(load(vid, j), (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        f = C.dis.calc(a, b, None)
        f = cv2.resize(f, (SW, SH), interpolation=cv2.INTER_LINEAR) * 2.0
        if len(C.flows) > 8:
            C.flows.pop(next(iter(C.flows)))
        C.flows[key] = f
    return f


def sample(vid, s, speed):
    """Source frame at fractional time s. Optical-flow in-between frames (补帧)."""
    x = s * SFPS
    i = int(math.floor(x)); a = x - i
    if a < 0.08 or a > 0.92:
        return load(vid, int(round(x)))
    I0, I1 = load(vid, i), load(vid, i + 1)
    if speed > 1.7:   # fast ramps: a plain blend reads as motion blur
        return cv2.addWeighted(I0, 1 - a, I1, a, 0)
    f01 = flow(vid, i, i + 1); f10 = flow(vid, i + 1, i)
    w0 = cv2.remap(I0, C.sgx - a * f01[..., 0], C.sgy - a * f01[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(I1, C.sgx - (1 - a) * f10[..., 0], C.sgy - (1 - a) * f10[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(w0, 1 - a, w1, a, 0)


def grade(img, sat=1.25, con=1.10, bright=0.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if sat != 1.0:
        hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=sat)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.convertScaleAbs(img, alpha=con, beta=-128 * (con - 1) + bright)


def radial_blur(img, amt, cx, cy):
    acc = img.astype(np.float32)
    n = 6
    for k in range(1, n):
        s = 1 + amt * k / n
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (acc / n).astype(np.uint8)


# ------------------------------------------------------------------ frame
def render(n):
    T = n / FPS
    sg = find_seg(T)
    s = float(sg['f'](T)); sp = float(sg['df'](T))
    img = sample(sg['vid'], s, sp)

    style = sg['style']
    if style == 'intro':
        u = min(1, T / B(4))
        img = cv2.GaussianBlur(img, (0, 0), 0.1 + 5 * (1 - u))
        img = grade(img, sat=0.9 + 0.35 * u, con=1.1, bright=-45 * (1 - u))
    elif style == 'finale' and T > B(184):
        u = min(1, (T - B(184)) / 1.2)
        img = grade(img, sat=1.25 - 0.9 * u, con=1.1, bright=-20 * u)
    else:
        img = grade(img)

    # ---- camera: zoom punches, beat pulses, risers, shake
    Z = 1.0; shx = shy = rot = 0.0; rgb = 0.0
    # no screen shake anywhere: a soft zoom-in on each kill is the only camera hit
    for t0, st in PUNCH:
        dt = T - t0
        if -0.05 < dt < 1.2:
            Z += 0.04 * st * env_punch(dt, decay=0.25, attack=0.05)
            if dt >= 0:
                rgb += 4 * st * math.exp(-dt / 0.08)
    for t0 in DROPS:
        dt = T - t0
        if 0 <= dt < 1.5:
            Z += 0.1 * math.exp(-dt / 0.3)
            rgb += 7 * math.exp(-dt / 0.12)
    for a, b in RISERS:
        if a <= T < b:
            u = (T - a) / (b - a)
            Z += 0.06 * u ** 2
            rgb += 3 * u ** 2
    if style == 'finale' and T > FINAL_KILL:   # slow push-in through the final slow-mo
        Z += 0.08 * min(1, (T - FINAL_KILL) / 4.5)
    if style == 'intro':
        Z += 0.06 * (1 - T / B(4))

    k = (W / SW) * Z
    c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    csx, csy = SW * 0.5, SH * 0.52
    cox, coy = W * 0.5 + shx, H * 0.52 + shy
    M = np.float32([[k * c, -k * s_, cox - k * (c * csx - s_ * csy)],
                    [k * s_, k * c, coy - k * (s_ * csx + c * csy)]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ---- transitions: zoom blur around cuts
    for tc, kind in CUTS:
        dt = T - tc
        if kind == 'blur' and -0.1 < dt < 0.1:
            amt = 0.1 * (1 - abs(dt) / 0.1)
            out = radial_blur(out, amt, W * 0.5, H * 0.52)
            rgb += 5 * (1 - abs(dt) / 0.1)
    for a, b in RISERS:          # rushing zoom-blur through the risers
        if a + 0.4 <= T < b:
            out = radial_blur(out, 0.05 * (T - a - 0.4) / (b - a - 0.4), W * 0.5, H * 0.52)
    if style == 'teaser':
        rgb += 3

    if rgb > 0.8:
        d = int(round(rgb))
        b_, g_, r_ = cv2.split(out)
        r_ = np.roll(r_, d, axis=1); b_ = np.roll(b_, -d, axis=1)
        out = cv2.merge([b_, g_, r_])

    out = out.astype(np.float32) * C.vign

    # ---- flashes
    fl = 0.0
    for t0, a in FLASH:
        dt = T - t0
        if 0 <= dt < 0.6:
            fl = max(fl, a * math.exp(-dt / 0.085))
    if fl > 0.01:
        out = out * (1 - fl) + 255 * fl

    # ---- fade in / out
    if T < 0.5:
        out *= (T / 0.5) ** 1.5
    if T > 78.3:
        out *= 1 - min(1, (T - 78.3) / 0.85)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_bytes(n):
    return render(n).tobytes()


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'preview':
        init_worker()
        outdir = os.path.join(ROOT, 'preview'); os.makedirs(outdir, exist_ok=True)
        for t in sys.argv[2:]:
            n = int(round(float(t) * FPS))
            cv2.imwrite(os.path.join(outdir, 'p_%06.2f.jpg' % float(t)), render(n), [cv2.IMWRITE_JPEG_QUALITY, 85])
    elif mode == 'video':
        out = sys.argv[2]
        t_a = float(sys.argv[3]) if len(sys.argv) > 3 else 0
        t_b = float(sys.argv[4]) if len(sys.argv) > 4 else DUR
        from multiprocessing import Pool
        frames = range(int(round(t_a * FPS)), int(round(t_b * FPS)))
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
               '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '15',
               '-pix_fmt', 'yuv420p', out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with Pool(4, initializer=init_worker) as pool:
            for i, b in enumerate(pool.imap(render_bytes, frames, chunksize=8)):
                p.stdin.write(b)
                if i % 600 == 0:
                    print('frame', i, flush=True)
        p.stdin.close(); p.wait()
        print('done', flush=True)
