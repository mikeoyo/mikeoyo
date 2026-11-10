"""王者荣耀击杀秀 第三弹 — the 第二弹 footage and BGM, re-cut in the style of the reference edits.

Techniques taken from the references: white-out transitions, whip-pan motion-blur cuts,
big push-ins on the fights, glow on skill effects, a darker grade, motion blur on fast
ramps. (The grey smeared frames and long freezes in the references were playback glitches
from the phone editor, not effects, so they are not copied.)
Own additions: a rewind-and-replay of one kill, a dark "time stop" while the music cuts
out, a zoom-out reveal for the opening, afterimage trails in the last slow-mo kills.
Still: no text, no screen shake, only two white-outs, 60 fps optical-flow 补帧.
"""
import os, sys, math, subprocess
import numpy as np, cv2
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'm2')          # frame caches + audio from 第二弹
CACHE = os.path.join(DATA, 'cache')
FPS, SFPS = 60, 30
W, H = 1920, 1080
SW, SH = 1280, 720
DUR = 144.4
BT = np.load(os.path.join(DATA, 'audio/bt.npy'))
NFR = {v: len(os.listdir(os.path.join(CACHE, v))) for v in os.listdir(CACHE)}


def B(b):
    return float(np.interp(b, np.arange(len(BT)), BT))


# ------------------------------------------------------------------ timeline (beat, source second)
def piece(vid, anchors, cut_in='blur'):
    ts = [a[1] if a[0] == 't' else B(a[0]) for a in anchors]
    ss = [a[2] if a[0] == 't' else a[1] for a in anchors]
    f = PchipInterpolator(ts, ss)
    return dict(vid=vid, t0=ts[0], t1=ts[-1], f=f, df=f.derivative(), cut_in=cut_in)


SEGS = [
    piece('355', [('t', 0.0, 0.0), (16, 8.17), (20, 10.0)], cut_in='none'),
    piece('354', [(20, 0.0), (40, 11.17), (44, 13.1)], cut_in='whip'),
    piece('358b', [(44, 0.0), (53, 4.17), (55, 5.27), (63, 8.87), (68, 11.5)], cut_in='white'),
    piece('358c', [(68, 0.0), (78, 5.07), (96, 13.47), (100, 15.5)], cut_in='whip'),
    piece('356a', [(100, 0.5), (113, 6.67), (121, 10.27), (128, 13.5)], cut_in='blur'),
    # kill at 154, rewind 155->156, replay into slow-mo kill at 158
    piece('359', [(128, 0.0), (131, 1.57), (154, 13.17), (155, 13.62), (156, 12.4), (158, 13.17),
                  (160, 13.6)], cut_in='whip'),
    # music cuts out at beats 171-173: the picture nearly stops too
    piece('356b', [(160, 1.5), (166, 4.37), (171, 6.75), (173, 6.95), (178, 9.87), (183, 11.97),
                   (189, 14.87), (192, 16.2)], cut_in='blur'),
    # kill at 221 then slow-mo through the music break, white-out into the climax
    piece('357a', [(192, 20.0), (209, 27.57), (221, 33.17), (222, 33.32), (224, 33.78)], cut_in='whip'),
    piece('360', [(224, 0.5), (231, 4.17), (240, 8.27), (244, 10.17), (245.25, 10.32), (256, 15.57),
                  (260, 18.0)], cut_in='white'),
    piece('362', [(260, 3.5), (267, 6.97), (278, 12.17), (285, 15.67), (286, 16.17), (292, 19.07),
                  (293.25, 19.22), ('t', DUR + 1, 21.2)], cut_in='whip'),
]
SEGS[-1]['t1'] = DUR + 1

KILLS = [16, 40, 55, 63, 96, 113, 131, 154, 158, 178, 183, 221, 231, 240, 244, 256, 267, 278, 285, 292]
ASSISTS = [53, 78, 121, 166, 189, 209, 286]
EVENTS = [(B(b), 'K') for b in KILLS] + [(B(b), 'A') for b in ASSISTS]
BIG = {154: 1.4, 158: 1.6, 221: 1.6, 244: 1.7, 292: 1.8, 183: 1.3, 256: 1.3}
BUILD = [(B(42), B(44))]                           # music break before the first white-out
PUNCH = [(B(b), BIG.get(b, 1.0)) for b in KILLS] + [(B(b), 0.7) for b in ASSISTS] + [(B(173), 1.2)]
DROPS = [B(205)]                                   # end of the riser in 357a
RISER = (B(201), B(205))
CUTS = [(sg['t0'], sg['cut_in']) for sg in SEGS[1:]]
WHITE = [t for t, k in CUTS if k == 'white']
TIMESTOP = (B(171), B(173))
SLOWDARK = (B(221.3), B(224))
ECHO = [(B(244), B(247.5)), (B(292), DUR)]
FINAL = B(292)
INTRO = B(0)                                       # zoom-out reveal before the first beat


def find_seg(T):
    for sg in SEGS:
        if sg['t0'] <= T < sg['t1']:
            return sg
    return SEGS[-1]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def env_punch(t, decay=0.25, attack=0.05):
    if t < 0:
        return 0.0
    return t / attack if t < attack else math.exp(-(t - attack) / decay)


# ------------------------------------------------------------------ frame access
class Ctx:
    pass


C = Ctx()


def init_worker():
    C.frames, C.flows = {}, {}
    C.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    C.vign = np.clip(1.0 - 0.42 * np.clip(r - 0.45, 0, None) ** 1.4, 0.5, 1)[..., None].astype(np.float32)
    C.vign_dark = np.clip(1.0 - 0.9 * np.clip(r - 0.25, 0, None) ** 1.2, 0.15, 1)[..., None].astype(np.float32)
    C.sgx, C.sgy = np.meshgrid(np.arange(SW, dtype=np.float32), np.arange(SH, dtype=np.float32))
    C.scan = np.ones((H, 1, 1), np.float32); C.scan[::4] = 0.82


def load(vid, i):
    i = int(min(max(i, 0), NFR[vid] - 1))
    im = C.frames.get((vid, i))
    if im is None:
        im = cv2.imread(os.path.join(CACHE, vid, '%05d.jpg' % i))
        if len(C.frames) > 32:
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


def blend_nf(vid, s):
    x = max(s, 0) * SFPS; i = int(math.floor(x)); a = x - i
    return cv2.addWeighted(load(vid, i), 1 - a, load(vid, i + 1), a, 0)


def sample(vid, s, speed):
    """Frame at fractional source time s; optical-flow in-betweens (补帧), motion blur when fast."""
    s = max(s, 0.0)
    if abs(speed) > 1.35:      # motion blur: blend three sub-frame samples
        d = speed / FPS / 3
        acc = sum(blend_nf(vid, s + k * d).astype(np.float32) for k in (-1, 0, 1))
        return (acc / 3).astype(np.uint8)
    x = s * SFPS
    i = int(math.floor(x)); a = x - i
    if a < 0.08 or a > 0.92:
        return load(vid, int(round(x)))
    I0, I1 = load(vid, i), load(vid, i + 1)
    f01 = flow(vid, i, i + 1); f10 = flow(vid, i + 1, i)
    w0 = cv2.remap(I0, C.sgx - a * f01[..., 0], C.sgy - a * f01[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    w1 = cv2.remap(I1, C.sgx - (1 - a) * f10[..., 0], C.sgy - (1 - a) * f10[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(w0, 1 - a, w1, a, 0)


def base_frame(T):
    sg = find_seg(T)
    return sample(sg['vid'], float(sg['f'](T)), float(sg['df'](T)))


# ------------------------------------------------------------------ looks
def grade(img, sat=1.28, con=1.12, bright=-4.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=sat)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.convertScaleAbs(img, alpha=con, beta=-128 * (con - 1) + bright)


def bloom(img, amt=0.45):
    small = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum = small.max(2, keepdims=True)
    hi = small * np.clip((lum - 200) / 55, 0, 1)
    g = cv2.GaussianBlur(hi, (0, 0), 4) + 0.4 * cv2.GaussianBlur(hi, (0, 0), 12)
    g = cv2.resize(g, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(img.astype(np.float32) + amt * g, 0, 255).astype(np.uint8)


def echo(vid, s, img):
    out = img.astype(np.float32)
    for k, w in ((1, 0.6), (2, 0.42), (3, 0.28), (4, 0.18)):
        p = blend_nf(vid, s - 0.035 * k).astype(np.float32)
        m = np.clip((p.max(2, keepdims=True) - 165) / 70, 0, 1) * w
        out = np.maximum(out, p * m + out * (1 - m))
    return out.astype(np.uint8)


def radial_blur(img, amt, cx, cy):
    acc = img.astype(np.float32)
    for k in range(1, 6):
        s = 1 + amt * k / 6
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        acc += cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    return (acc / 6).astype(np.uint8)


def hblur(img, length):
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    k = max(1, int(length) // 2)
    small = cv2.blur(small, (k, 1))
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def fight_zoom(T):
    z = 0.0
    for te, kind in EVENTS:
        dt = T - te
        if -1.9 < dt < 1.3:
            w = smooth((dt + 1.9) / 1.3) if dt < -0.6 else (1.0 if dt < 0.5 else 1 - smooth((dt - 0.5) / 0.8))
            z = max(z, w * (0.17 if kind == 'K' else 0.12))
    return z


# ------------------------------------------------------------------ frame
def render(n):
    T = n / FPS
    sg = find_seg(T)
    vid, s, sp = sg['vid'], float(sg['f'](T)), float(sg['df'](T))
    img = sample(vid, s, sp)
    if any(a <= T < b for a, b in ECHO):
        img = echo(vid, s, img)

    # ---- grade / looks
    dark = 0.0
    if TIMESTOP[0] <= T < TIMESTOP[1]:
        dark = min(1, (T - TIMESTOP[0]) / 0.12)
    if SLOWDARK[0] <= T < SLOWDARK[1]:
        dark = max(dark, 0.6 * min(1, (T - SLOWDARK[0]) / 0.25))
    if T > FINAL + 1.2:
        u = min(1, (T - FINAL - 1.2) / 2.0)
        img = grade(img, sat=1.28 - 0.75 * u, bright=-4 - 14 * u)
    else:
        img = grade(img, sat=1.28 - 0.5 * dark)
    if sp < -0.2:                                  # rewind look: cooler, washed, scanlines
        r = min(1, (-sp - 0.2) / 0.8)
        img = cv2.addWeighted(img, 1 - 0.45 * r, cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                              cv2.COLOR_GRAY2BGR), 0.45 * r, 0)
        img = cv2.add(img, np.array([18 * r, 6 * r, -10 * r, 0]))
    img = bloom(img, 0.45 + 0.35 * dark)

    # ---- camera (no shake): fight push-ins, kill punches, riser, freeze push
    Z = 1.0 + fight_zoom(T)
    rgb = 0.0
    for t0, st in PUNCH:
        dt = T - t0
        if -0.05 < dt < 1.2:
            Z += 0.045 * st * env_punch(dt)
            if dt >= 0:
                rgb += 4 * st * math.exp(-dt / 0.08)
    for t0 in DROPS:
        dt = T - t0
        if 0 <= dt < 1.5:
            Z += 0.1 * math.exp(-dt / 0.3); rgb += 6 * math.exp(-dt / 0.12)
    if RISER[0] <= T < RISER[1]:
        u = (T - RISER[0]) / (RISER[1] - RISER[0]); Z += 0.12 * u * u; rgb += 3 * u
    if SLOWDARK[0] <= T < SLOWDARK[1]:
        Z += 0.1 * smooth((T - SLOWDARK[0]) / (SLOWDARK[1] - SLOWDARK[0]))
    for a, b in BUILD:
        if a <= T < b:
            Z += 0.12 * smooth((T - a) / (b - a))
    if T < INTRO:
        Z += 0.22 * (1 - smooth(T / INTRO))
    if TIMESTOP[0] <= T < TIMESTOP[1]:
        Z += 0.06 * smooth((T - TIMESTOP[0]) / (TIMESTOP[1] - TIMESTOP[0]))
    if T > FINAL:
        Z += 0.07 * min(1, (T - FINAL) / 4.5)
    if sp < -0.2:
        Z += 0.05

    dx = 0.0; whip = 0.0
    for tc, kind in CUTS:
        dt = T - tc
        if kind == 'whip' and -0.14 < dt < 0.14:
            u = 1 - abs(dt) / 0.14
            whip = max(whip, u)
            dx = (-1 if dt < 0 else 1) * W * 0.22 * u * u
    k = (W / SW) * Z
    cx, cy = SW * 0.5, SH * 0.52
    M = np.float32([[k, 0, W * 0.5 + dx - k * cx], [0, k, H * 0.52 - k * cy]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ---- transitions
    if whip > 0.02:
        out = hblur(out, 6 + 260 * whip * whip)
        rgb += 16 * whip
    for tc, kind in CUTS:
        dt = T - tc
        if kind == 'blur' and -0.09 < dt < 0.09:
            out = radial_blur(out, 0.09 * (1 - abs(dt) / 0.09), W * 0.5, H * 0.52)
            rgb += 5 * (1 - abs(dt) / 0.09)
    if RISER[0] + 1.0 <= T < RISER[1]:
        out = radial_blur(out, 0.035 * (T - RISER[0] - 1.0) / (RISER[1] - RISER[0] - 1.0), W * 0.5, H * 0.52)
    for a, b in BUILD:
        if a <= T < b - 0.3:
            out = radial_blur(out, 0.04 * (T - a) / (b - 0.3 - a), W * 0.5, H * 0.52)
    if T < INTRO:
        out = radial_blur(out, 0.08 * (1 - smooth(T / INTRO)), W * 0.5, H * 0.52)
    for tc in WHITE:
        dt = T - tc
        if -0.3 < dt < 0:
            u = (dt + 0.3) / 0.3
            out = radial_blur(out, 0.12 * u * u, W * 0.5, H * 0.52)
    if rgb > 0.8:
        d = int(round(rgb))
        b_, g_, r_ = cv2.split(out)
        out = cv2.merge([np.roll(b_, -d, axis=1), g_, np.roll(r_, d, axis=1)])

    out = out.astype(np.float32) * C.vign
    if dark > 0:
        out = out * (1 - dark + dark * C.vign_dark)
    if sp < -0.2:
        out = out * C.scan
    for tc in WHITE:                               # the only two white-outs
        dt = T - tc
        if -0.3 < dt < 0:
            u = (dt + 0.3) / 0.3
            out = out * (1 + 1.6 * u * u) + 255 * 0.85 * u ** 3
        elif 0 <= dt < 0.5:
            w = 0.85 * math.exp(-dt / 0.1)
            out = out * (1 - w) + 255 * w
    if T < INTRO:
        out *= smooth(T / INTRO) ** 0.8
    if T > DUR - 0.9:
        out *= max(0.0, (DUR - T) / 0.9)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_bytes(n):
    return render(n).tobytes()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'none'
    if mode == 'plan':
        for sg in SEGS:
            ts = np.linspace(sg['t0'], min(sg['t1'], DUR) - 1e-3, 1200); sp = sg['df'](ts)
            print('%-5s %-5s T %6.2f-%6.2f  speed %.2f..%.2f' % (sg['vid'], sg['cut_in'], sg['t0'],
                  min(sg['t1'], DUR), sp.min(), sp.max()))
    elif mode == 'preview':
        init_worker()
        outdir = os.path.join(HERE, 'preview'); os.makedirs(outdir, exist_ok=True)
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
