"""王者荣耀击杀秀 第四弹 — six clips (363-368) cut to the provided DJ track, 燃 + 伤感.

Arc: sad cold open (slow-mo black-and-white flash-forward of the pentakill's last kill)
-> cold, letterboxed approach that opens up on the drop -> beat-synced kills -> riser and
white-out into the 五连绝世 in the middle -> two more fights -> a slow-mo kill that fades to
black and white and closes the letterbox.

Kill effect learned from the reference (神特写击杀): on the hit the frame turns into a
high-contrast black-and-white close-up with white speed streaks, held for a moment, then
eases back to colour. No text, no shake, 60 fps optical-flow interpolation (补帧).
"""
import os, sys, math, subprocess
import numpy as np, cv2
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
FPS, SFPS = 60, 30
W, H = 1920, 1080
SW, SH = 1280, 720
DUR = 84.4
BT = np.load(os.path.join(HERE, 'audio/bt.npy'))
NFR = {v: len(os.listdir(os.path.join(CACHE, v))) for v in os.listdir(CACHE)}


def B(b):
    if b < 0:
        return float(BT[0] + b * (BT[1] - BT[0]))
    return float(np.interp(b, np.arange(len(BT)), BT))


def piece(vid, anchors, cut_in, look='normal'):
    ts = [a[1] if a[0] == 't' else B(a[0]) for a in anchors]
    ss = [a[2] if a[0] == 't' else a[1] for a in anchors]
    f = PchipInterpolator(ts, ss)
    return dict(vid=vid, t0=ts[0], t1=ts[-1], f=f, df=f.derivative(), cut_in=cut_in, look=look)


SEGS = [
    # cold open: the pentakill's last kill, slow-mo, black and white, like a memory
    piece('365', [('t', 0.0, 15.35), (11, 16.95), ('t', B(13), 17.35)], 'none', look='memory'),
    # cold approach, letterboxed; the first kill lands on the drop at beat 44
    piece('367', [(13, 5.0), (44, 17.83), (48, 19.6)], 'fade', look='intro'),
    piece('366', [(48, 0.9), (53, 3.45), (63, 7.85), (64, 8.4)], 'whip'),
    piece('363', [(64, 0.9), (70, 3.95), (76, 6.6)], 'blur'),
    piece('364', [(76, 4.3), (81, 6.65), (91, 12.35), (92, 12.75)], 'whip'),
    piece('363', [(92, 39.6), (95, 40.97), (105, 45.36), (106, 45.9)], 'blur'),
    # 五连绝世 in the middle, entered through a white-out after the riser
    piece('365', [(106, 1.0), (112, 3.55), (115, 4.85), (124, 9.15), (133, 13.25), (141, 16.95),
                  (143, 17.1), (145.5, 17.5), (147, 18.3)], 'white', look='penta'),
    piece('364', [(147, 62.6), (151, 64.56), (159, 68.16), (160, 68.6)], 'whip'),
    piece('368', [(160, 13.4), (170, 17.97), (171, 18.46), (172, 19.0)], 'whip'),
    # sad ending: last kill, then slow-mo into black and white
    piece('366', [(172, 40.0), (176, 41.86), (184, 42.9), ('t', DUR + 1, 43.1)], 'fade', look='outro'),
]
SEGS[-1]['t1'] = DUR + 1

KILL_BEATS = [44, 53, 63, 70, 81, 91, 95, 105, 112, 115, 124, 133, 141, 151, 159, 170, 171, 176]
ASSIST_BEATS = [((13.09 - 5.0) / (17.83 - 5.0)) * 31 + 13, ((16.08 - 5.0) / (17.83 - 5.0)) * 31 + 13]
ASSIST_T = [B(b) for b in ASSIST_BEATS]
# the reference effect: (beat, hold seconds); long ones also get slow-mo from the timeline
GOD = [(44, 0.22), (63, 0.16), (112, 0.16), (133, 0.16), (141, 0.55), (159, 0.18), (171, 0.2), (176, 0.45)]
GOD_T = [(B(b), h) for b, h in GOD]
BIG = {141: 2.0, 176: 1.6, 44: 1.5, 112: 1.3, 133: 1.3}
PUNCH = [(B(b), BIG.get(b, 1.0)) for b in KILL_BEATS] + [(t, 0.6) for t in ASSIST_T]
CUTS = [(sg['t0'], sg['cut_in']) for sg in SEGS[1:]]
WHITE = [t for t, k in CUTS if k == 'white']
RISER = (B(103), B(106))
FINAL = B(176)
EVENT_T = sorted([B(b) for b in KILL_BEATS] + ASSIST_T)


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


class Ctx:
    pass


C = Ctx()


def init_worker():
    C.frames, C.flows = {}, {}
    C.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    C.vign = np.clip(1.0 - 0.42 * np.clip(r - 0.45, 0, None) ** 1.4, 0.5, 1)[..., None].astype(np.float32)
    C.vign_dark = np.clip(1.0 - 0.75 * np.clip(r - 0.3, 0, None) ** 1.2, 0.2, 1)[..., None].astype(np.float32)
    C.sgx, C.sgy = np.meshgrid(np.arange(SW, dtype=np.float32), np.arange(SH, dtype=np.float32))


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
    if abs(speed) > 1.35:
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


# ------------------------------------------------------------------ looks
def grade(img, sat=1.22, con=1.1, bright=-3.0, cool=0.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=sat)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = cv2.convertScaleAbs(img, alpha=con, beta=-128 * (con - 1) + bright)
    if cool > 0:                                     # teal shadows for the sad parts
        f = img.astype(np.float32)
        f = f * np.array([1 + 0.10 * cool, 1 + 0.02 * cool, 1 - 0.08 * cool], np.float32)
        img = np.clip(f, 0, 255).astype(np.uint8)
    return img


def bloom(img, amt=0.35):
    small = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum = small.max(2, keepdims=True)
    hi = small * np.clip((lum - 200) / 55, 0, 1)
    g = cv2.GaussianBlur(hi, (0, 0), 4) + 0.4 * cv2.GaussianBlur(hi, (0, 0), 12)
    g = cv2.resize(g, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(img.astype(np.float32) + amt * g, 0, 255).astype(np.uint8)


def mono(img, con=1.15):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = np.clip((g - 118) * con + 122, 0, 255)
    return cv2.merge([g, g, g])


def god_look(img, k, seed):
    """The reference's kill close-up: hard black-and-white, blown highlights, white speed streaks."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = np.clip((g - 112) * 1.5 + 118, 0, 255)
    hi = np.clip(g - 175, 0, 255)
    small = cv2.resize(hi, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    ang = -8 + 16 * ((seed * 0.6180339) % 1)       # a different slant for each kill
    L = 61
    ker = np.zeros((L, L), np.float32); ker[L // 2, :] = 1
    M = cv2.getRotationMatrix2D((L / 2, L / 2), ang, 1.0)
    ker = cv2.warpAffine(ker, M, (L, L)); ker /= ker.sum()
    st = cv2.filter2D(small, -1, ker)
    st = cv2.resize(st, (W, H), interpolation=cv2.INTER_LINEAR)
    out = np.clip(g * 0.97 + 1.1 * st + 0.2 * cv2.GaussianBlur(hi, (0, 0), 6), 0, 255)
    out = cv2.merge([out, out, out]).astype(np.float32)
    col = img.astype(np.float32)
    return col * (1 - k) + out * k


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
    for te in EVENT_T:
        dt = T - te
        if -1.9 < dt < 1.3:
            w = smooth((dt + 1.9) / 1.3) if dt < -0.6 else (1.0 if dt < 0.5 else 1 - smooth((dt - 0.5) / 0.8))
            z = max(z, w * 0.08)
    return z


def god_amount(T):
    """0..1 strength of the black-and-white kill close-up at time T, and its zoom."""
    k = 0.0; z = 0.0; idx = 0
    for i, (t0, hold) in enumerate(GOD_T):
        dt = T - t0
        if -0.04 <= dt < hold + 0.4:
            if dt < 0:
                v = (dt + 0.04) / 0.04
            elif dt < hold:
                v = 1.0
            else:
                v = 1 - smooth((dt - hold) / 0.4)
            if v > k:
                k = v; idx = i
            z = max(z, (0.07 + 0.06 * min(hold, 0.5)) * (1.0 if dt < hold else 1 - smooth((dt - hold) / 0.45)) if dt >= 0 else 0.0)
    return k, z, idx


# ------------------------------------------------------------------ frame
def render(n):
    T = n / FPS
    sg = find_seg(T)
    vid, s, sp = sg['vid'], float(sg['f'](T)), float(sg['df'](T))
    img = sample(vid, s, sp)
    look = sg['look']

    # ---- colour: sad parts cold and drained, the drop in full colour
    lb = 0.0                                       # letterbox amount
    bw = 0.0                                       # plain black-and-white amount
    if look == 'memory':
        img = grade(img, sat=0.5, con=1.05, bright=-12, cool=1.0)
        bw = 0.85; lb = 1.0
    elif look == 'intro':
        u = smooth((T - sg['t0']) / (B(44) - sg['t0']))
        img = grade(img, sat=0.55 + 0.67 * u, con=1.05 + 0.05 * u, bright=-10 + 7 * u, cool=1 - u)
        lb = 1 - smooth((T - (B(44) - 0.25)) / 0.35)
    elif look == 'outro':
        u = smooth((T - (FINAL + 0.35)) / 2.2)
        img = grade(img, sat=1.22 - 0.6 * u, bright=-3 - 10 * u, cool=u)
        bw = 0.9 * u
        lb = smooth((T - (FINAL + 0.5)) / 1.2)
    else:
        img = grade(img)
    img = bloom(img, 0.35)
    if bw > 0:
        img = (img.astype(np.float32) * (1 - bw) + mono(img) * bw).astype(np.uint8)

    # ---- camera (no shake)
    gk, gz, gi = god_amount(T)
    Z = 1.0 + min(0.16, fight_zoom(T) + gz)
    rgb = 0.0
    for t0, st in PUNCH:
        dt = T - t0
        if -0.05 < dt < 1.2:
            Z += 0.04 * st * env_punch(dt)
            if dt >= 0:
                rgb += 3.5 * st * math.exp(-dt / 0.08)
    if RISER[0] <= T < RISER[1]:
        u = (T - RISER[0]) / (RISER[1] - RISER[0]); Z += 0.14 * u * u; rgb += 3 * u
    if look == 'memory':
        Z += 0.08 * T / B(13)
    if look == 'outro' and T > FINAL:
        Z += 0.08 * smooth((T - FINAL) / 4.0)

    dx = 0.0; whip = 0.0
    for tc, kind in CUTS:
        dt = T - tc
        if kind == 'whip' and -0.14 < dt < 0.14:
            u = 1 - abs(dt) / 0.14
            whip = max(whip, u)
            dx = (-1 if dt < 0 else 1) * W * 0.2 * u * u
    k = (W / SW) * Z
    cx, cy = SW * 0.5, SH * 0.5
    M = np.float32([[k, 0, W * 0.5 + dx - k * cx], [0, k, H * 0.5 - k * cy]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ---- kill close-up (reference effect), with a zoom-blur rush into it
    if gk > 0.01:
        t0 = GOD_T[gi][0]
        if T < t0 + 0.06:
            out = radial_blur(out, 0.035 * gk, W * 0.5, H * 0.5)
        out = god_look(out, gk, gi + 1).astype(np.uint8)

    # ---- transitions
    if whip > 0.02:
        out = hblur(out, 6 + 240 * whip * whip)
        rgb += 14 * whip
    for tc, kind in CUTS:
        dt = T - tc
        if kind == 'blur' and -0.09 < dt < 0.09:
            out = radial_blur(out, 0.09 * (1 - abs(dt) / 0.09), W * 0.5, H * 0.5)
            rgb += 5 * (1 - abs(dt) / 0.09)
    if RISER[0] + 0.5 <= T < RISER[1]:
        out = radial_blur(out, 0.04 * (T - RISER[0] - 0.5) / (RISER[1] - RISER[0] - 0.5), W * 0.5, H * 0.5)
    for tc in WHITE:
        dt = T - tc
        if -0.3 < dt < 0:
            u = (dt + 0.3) / 0.3
            out = radial_blur(out, 0.12 * u * u, W * 0.5, H * 0.5)
    if rgb > 0.8:
        d = int(round(rgb))
        b_, g_, r_ = cv2.split(out)
        out = cv2.merge([np.roll(b_, -d, axis=1), g_, np.roll(r_, d, axis=1)])

    out = out.astype(np.float32) * C.vign
    if look in ('memory', 'outro'):
        dk = 1.0 if look == 'memory' else smooth((T - FINAL) / 2.0)
        out = out * (1 - 0.6 * dk + 0.6 * dk * C.vign_dark)
    for tc in WHITE:                               # the one white-out, into the pentakill
        dt = T - tc
        if -0.3 < dt < 0:
            u = (dt + 0.3) / 0.3
            out = out * (1 + 1.5 * u * u) + 255 * 0.8 * u ** 3
        elif 0 <= dt < 0.5:
            w = 0.8 * math.exp(-dt / 0.1)
            out = out * (1 - w) + 255 * w
    # dip through black between the cold open and the approach, and at the very ends
    tc = B(13)
    if abs(T - tc) < 0.35:
        out *= smooth(abs(T - tc) / 0.35)
    if T < 0.6:
        out *= smooth(T / 0.6)
    if T > DUR - 1.2:
        out *= max(0.0, (DUR - T) / 1.2)
    if lb > 0.001:
        hb = int(round(132 * smooth(lb)))
        if hb > 0:
            out[:hb] = 0; out[H - hb:] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def render_bytes(n):
    return render(n).tobytes()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'none'
    if mode == 'plan':
        for sg in SEGS:
            ts = np.linspace(sg['t0'], min(sg['t1'], DUR) - 1e-3, 1200); sp = sg['df'](ts)
            print('%-5s %-6s %-6s T %6.2f-%6.2f src %6.2f-%6.2f speed %.2f..%.2f' % (
                sg['vid'], sg['cut_in'], sg['look'], sg['t0'], min(sg['t1'], DUR), sg['f'](ts[0]), sg['f'](ts[-1]),
                sp.min(), sp.max()))
        print('god shots at', [round(t, 2) for t, h in GOD_T])
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
