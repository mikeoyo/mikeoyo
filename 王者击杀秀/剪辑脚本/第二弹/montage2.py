"""王者荣耀击杀集锦 第二弹 — game footage only, cut to the user's BGM.

Longer clips (from before the fight to a while after it), cuts and kills on the
beats of the BGM, near-real-time playback, slow-mo only on the four flashiest
kills, no shake, four soft flashes, 60 fps optical-flow interpolation (补帧).
Source time follows a monotone PCHIP curve through (output time, source time)
anchors, so kills land exactly on beats.
"""
import os, sys, math, subprocess
import numpy as np, cv2
from scipy.interpolate import PchipInterpolator

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, 'cache')
FPS, SFPS = 60, 30
W, H = 1920, 1080
SW, SH = 1280, 720
DUR = 144.4
BT = np.load(os.path.join(ROOT, 'audio/bt.npy'))      # beat times of the BGM (they drift a little)
NFR = {v: len(os.listdir(os.path.join(CACHE, v))) for v in os.listdir(CACHE)}


def B(b):
    """Time of (fractional) beat index b."""
    return float(np.interp(b, np.arange(len(BT)), BT))


# ------------------------------------------------------------------ timeline
# (clip, out start, out end, src start, src end, events[(src time, kind, slow)])
# out start/end are beat indices; None means the very start / end of the video.
PLAN = [
    ('355',  None, 20,  0.0, 10.0, [(8.17, 'K', 0)]),
    ('354',  20,   44,  0.0, 13.1, [(11.17, 'K', 0)]),
    ('358b', 44,   68,  0.0, 11.5, [(4.17, 'A', 0), (5.27, 'K', 0), (8.87, 'K', 0)]),
    ('358c', 68,   100, 0.0, 15.5, [(5.07, 'A', 0), (13.47, 'K', 0)]),
    ('356a', 100,  128, 0.5, 13.5, [(6.67, 'K', 0), (10.27, 'A', 0)]),
    ('359',  128,  160, 0.0, 15.4, [(1.57, 'K', 0), (13.17, 'K', 1)]),
    ('356b', 160,  192, 1.5, 16.2, [(4.37, 'A', 0), (9.87, 'K', 0), (11.97, 'K', 0), (14.87, 'A', 0)]),
    ('357a', 192,  224, 20.0, 34.4, [(27.57, 'A', 0), (33.17, 'K', 1)]),
    ('360',  224,  260, 0.5, 18.0, [(4.17, 'K', 0), (8.27, 'K', 0), (10.17, 'K', 1, 244), (15.57, 'K', 0, 256)]),
    ('362',  260,  None, 3.5, 21.0, [(6.97, 'K', 0, 267), (12.17, 'K', 0, 278), (15.67, 'K', 0, 285),
                                     (16.17, 'A', 0, 286), (19.07, 'K', 2, 292)]),
]
FINAL_KILL_BEAT = 292

SEGS, EVENTS = [], []      # EVENTS: (output time, kind, slow, clip)
for vid, b0, b1, s0, s1, evs in PLAN:
    T0 = 0.0 if b0 is None else B(b0)
    T1 = DUR if b1 is None else B(b1)
    r = (s1 - s0) / (T1 - T0)
    anchors = [(T0, s0)]
    last_b = b0 if b0 is not None else -1
    for ev in evs:
        e, kind, slow = ev[:3]
        if len(ev) > 3:          # beat pinned by hand
            be = ev[3]
        else:
            Tn = T0 + (e - s0) / r
            be = int(np.argmin(np.abs(BT - Tn)))
            be = max(be, last_b + 2)
            if b1 is not None:
                be = min(be, b1 - (3 if slow else 1))
            assert be > last_b, (vid, e)
        anchors.append((B(be), e))
        if slow:
            anchors.append((B(be + 1.25), e + 0.15))
        EVENTS.append((B(be), kind, slow, vid))
        last_b = be + (2 if slow else 0)
    anchors.append((T1, s1))
    ts = np.array([a[0] for a in anchors]); ss = np.array([a[1] for a in anchors])
    f = PchipInterpolator(ts, ss)
    SEGS.append(dict(vid=vid, t0=T0, t1=T1, f=f, df=f.derivative(), anchors=anchors))
SEGS[-1]['t1'] = DUR + 1

# events -> effects ------------------------------------------------------------
PUNCH = []
for T, kind, slow, vid in EVENTS:
    PUNCH.append((T, 0.8 if kind == 'A' else (1.8 if slow else 1.15)))
FLASH = [(B(44), 0.3), (B(176), 0.3), (B(224), 0.4), (B(FINAL_KILL_BEAT), 0.35)]   # only four, soft
CUTS = [sg['t0'] for sg in SEGS[1:]]
FINAL_KILL = B(FINAL_KILL_BEAT)


def find_seg(T):
    for sg in SEGS:
        if sg['t0'] <= T < sg['t1']:
            return sg
    return SEGS[-1]


def env_punch(t, decay=0.25, attack=0.05):
    if t < 0:
        return 0.0
    if t < attack:
        return t / attack
    return math.exp(-(t - attack) / decay)


class Ctx:
    pass


C = Ctx()


def init_worker():
    C.frames, C.flows = {}, {}
    C.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    C.vign = np.clip(1.0 - 0.38 * np.clip(r - 0.5, 0, None) ** 1.5, 0.55, 1)[..., None].astype(np.float32)
    C.sgx, C.sgy = np.meshgrid(np.arange(SW, dtype=np.float32), np.arange(SH, dtype=np.float32))


def load(vid, i):
    i = int(min(max(i, 0), NFR[vid] - 1))
    im = C.frames.get((vid, i))
    if im is None:
        im = cv2.imread(os.path.join(CACHE, vid, '%05d.jpg' % i))
        if len(C.frames) > 24:
            C.frames.pop(next(iter(C.frames)))
        C.frames[(vid, i)] = im
    return im


def flow(vid, i, j):
    f = C.flows.get((vid, i, j))
    if f is None:
        a = cv2.cvtColor(cv2.resize(load(vid, i), (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(load(vid, j), (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        f = cv2.resize(C.dis.calc(a, b, None), (SW, SH), interpolation=cv2.INTER_LINEAR) * 2.0
        if len(C.flows) > 8:
            C.flows.pop(next(iter(C.flows)))
        C.flows[(vid, i, j)] = f
    return f


def sample(vid, s, speed):
    """Frame at fractional source time s; optical-flow in-betweens (补帧)."""
    x = s * SFPS
    i = int(math.floor(x)); a = x - i
    if a < 0.08 or a > 0.92:
        return load(vid, int(round(x)))
    I0, I1 = load(vid, i), load(vid, i + 1)
    if speed > 1.7:
        return cv2.addWeighted(I0, 1 - a, I1, a, 0)
    f01 = flow(vid, i, i + 1); f10 = flow(vid, i + 1, i)
    w0 = cv2.remap(I0, C.sgx - a * f01[..., 0], C.sgy - a * f01[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(I1, C.sgx - (1 - a) * f10[..., 0], C.sgy - (1 - a) * f10[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(w0, 1 - a, w1, a, 0)


def grade(img, sat=1.25, con=1.10, bright=0.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=sat)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.convertScaleAbs(img, alpha=con, beta=-128 * (con - 1) + bright)


def radial_blur(img, amt, cx, cy):
    acc = img.astype(np.float32)
    for k in range(1, 6):
        s = 1 + amt * k / 6
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (acc / 6).astype(np.uint8)


def render(n):
    T = n / FPS
    sg = find_seg(T)
    s = float(sg['f'](T)); sp = float(sg['df'](T))
    img = sample(sg['vid'], s, sp)
    if T > FINAL_KILL + 1.5:
        u = min(1, (T - FINAL_KILL - 1.5) / 2.0)
        img = grade(img, sat=1.25 - 0.7 * u, bright=-15 * u)
    else:
        img = grade(img)

    Z = 1.0; rgb = 0.0
    for t0, st in PUNCH:          # soft push-in on kills and assists, no shake
        dt = T - t0
        if -0.05 < dt < 1.2:
            Z += 0.04 * st * env_punch(dt)
            if dt >= 0:
                rgb += 4 * st * math.exp(-dt / 0.08)
    if T > FINAL_KILL:
        Z += 0.06 * min(1, (T - FINAL_KILL) / 4.5)
    k = (W / SW) * Z
    cx, cy = SW * 0.5, SH * 0.52
    M = np.float32([[k, 0, W * 0.5 - k * cx], [0, k, H * 0.52 - k * cy]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    for tc in CUTS:               # short zoom-blur across each cut
        dt = T - tc
        if -0.08 < dt < 0.08:
            out = radial_blur(out, 0.08 * (1 - abs(dt) / 0.08), W * 0.5, H * 0.52)
            rgb += 4 * (1 - abs(dt) / 0.08)
    if rgb > 0.8:
        d = int(round(rgb))
        b_, g_, r_ = cv2.split(out)
        out = cv2.merge([np.roll(b_, -d, axis=1), g_, np.roll(r_, d, axis=1)])

    out = out.astype(np.float32) * C.vign
    fl = 0.0
    for t0, a in FLASH:
        dt = T - t0
        if 0 <= dt < 0.6:
            fl = max(fl, a * math.exp(-dt / 0.085))
    if fl > 0.01:
        out = out * (1 - fl) + 255 * fl
    if T < 0.5:
        out *= (T / 0.5) ** 1.5
    if T > DUR - 0.9:
        out *= max(0.0, (DUR - T) / 0.9)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_bytes(n):
    return render(n).tobytes()


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'plan':
        for sg in SEGS:
            ts = np.linspace(sg['t0'], min(sg['t1'], DUR) - 1e-3, 600); sp = sg['df'](ts)
            print('%-5s T %6.2f-%6.2f  src %5.2f-%5.2f  speed %.2f-%.2f' % (
                sg['vid'], sg['t0'], min(sg['t1'], DUR), sg['f'](ts[0]), sg['f'](ts[-1]), sp.min(), sp.max()))
        for T, kind, slow, vid in EVENTS:
            print('  %s %s at %.2f%s' % (vid, kind, T, ' (slow-mo)' if slow else ''))
    elif mode == 'preview':
        init_worker()
        outdir = os.path.join(ROOT, 'preview'); os.makedirs(outdir, exist_ok=True)
        for t in sys.argv[2:]:
            cv2.imwrite(os.path.join(outdir, 'p_%06.2f.jpg' % float(t)), render(int(round(float(t) * FPS))),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    elif mode == 'video':
        out = sys.argv[2]
        from multiprocessing import Pool
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
               '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '15',
               '-pix_fmt', 'yuv420p', out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with Pool(4, initializer=init_worker) as pool:
            for i, b in enumerate(pool.imap(render_bytes, range(int(round(DUR * FPS))), chunksize=8)):
                p.stdin.write(b)
                if i % 1200 == 0:
                    print('frame', i, flush=True)
        p.stdin.close(); p.wait()
        print('done', flush=True)
