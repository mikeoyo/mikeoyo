"""Build a hand-drawn notebook scene: a lined page with the 520 proof and figure,
lying on a textbook and a worksheet. Everything is drawn from scratch."""
import os, math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(os.path.dirname(HERE), 'fonts')
rng = np.random.default_rng(520)

F_CJK = os.path.join(FONTS, 'LongCang.ttf')
F_LAT = os.path.join(FONTS, 'Caveat500.ttf')
F_PRINT = os.path.join(FONTS, 'NotoSerifSC-Regular.ttf')
_fc = {}


def font(path, size):
    k = (path, int(size))
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, int(size))
    return _fc[k]


# ------------------------------------------------------------------ ink primitives (alpha canvases, float32 0..1)
def stroke(canvas, pts, width=4.2, wobble=1.2, seed=None):
    """Hand-drawn polyline through pts, with low-frequency wobble and slight width change."""
    r = np.random.default_rng(seed)
    pts = np.asarray(pts, np.float64)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        L = np.linalg.norm(b - a)
        n = max(2, int(L / 6))
        t = np.linspace(0, 1, n + 1)[1:]
        nrm = np.array([-(b - a)[1], (b - a)[0]]) / (L + 1e-9)
        ph = r.uniform(0, 6.28, 2)
        w = wobble * (np.sin(t * math.pi * r.uniform(1, 2.2) + ph[0]) * 0.7 +
                      np.sin(t * math.pi * r.uniform(3, 5) + ph[1]) * 0.3) * np.sin(t * math.pi)
        seg = a[None] + t[:, None] * (b - a)[None] + w[:, None] * nrm[None]
        out.extend(seg)
    out = np.array(out)
    S = 4                                                    # supersample for clean AA
    x0, y0 = np.floor(out.min(0) - width * 2 - 4).astype(int)
    x1, y1 = np.ceil(out.max(0) + width * 2 + 4).astype(int)
    patch = np.zeros(((y1 - y0) * S, (x1 - x0) * S), np.uint8)
    p = ((out - [x0, y0]) * S).astype(np.int32)
    for i in range(len(p) - 1):
        wv = width * (0.9 + 0.2 * r.random())
        cv2.line(patch, tuple(p[i]), tuple(p[i + 1]), 255, max(1, int(round(wv * S))), cv2.LINE_AA)
    patch = cv2.resize(patch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    sub = canvas[y0:y1, x0:x1]
    np.maximum(sub, patch * (0.88 + 0.12 * r.random()), out=sub)


def dashed(canvas, a, b, dash=30, gap=17, width=3.6, seed=None):
    r = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    L = np.linalg.norm(b - a); d = (b - a) / L
    s = r.uniform(0, 6)
    while s < L:
        e = min(L, s + dash * r.uniform(0.8, 1.2))
        stroke(canvas, [a + d * s, a + d * e], width, wobble=0.4, seed=int(r.integers(1 << 30)))
        s = e + gap * r.uniform(0.8, 1.25)


def glyph(canvas, ch, x, base, size, seed, ink=1.0):
    """Draw one handwritten glyph with its left edge at x and baseline at base; return advance."""
    r = np.random.default_rng(seed)
    sz = size * r.uniform(0.95, 1.05)
    rot = r.uniform(-3.5, 3.5)
    dy = r.uniform(-3, 3)
    if ch in '∵∴':                                         # three dots
        h = sz * 0.42; w = sz * 0.42
        pts = [(x + w * 0.5, base - h), (x, base), (x + w, base)] if ch == '∴' else \
              [(x, base - h), (x + w, base - h), (x + w * 0.5, base)]
        for (px, py) in pts:
            px += r.uniform(-2, 2); py += r.uniform(-2, 2) + dy
            cv2.circle(canvas, (int(px), int(py)), max(3, int(sz * 0.045)), 1.0, -1, cv2.LINE_AA)
        return w + sz * 0.32
    if ch == '∠':                                          # quick "<" with a base stroke
        h = sz * 0.5; w = sz * 0.52
        stroke(canvas, [(x + w, base - h + dy), (x + r.uniform(0, 4), base + dy - 2),
                        (x + w * 1.02, base + dy - 1)], width=sz * 0.05, wobble=0.8, seed=seed)
        return w + sz * 0.12
    if ch == 'L':                                          # plain two-stroke L (Caveat's L reads as "(")
        h = sz * 0.6; w = sz * 0.36
        stroke(canvas, [(x + w * 0.18, base - h + dy), (x + w * 0.1, base + dy),
                        (x + w * 1.0, base + dy - 2)], width=sz * 0.05, wobble=0.7, seed=seed)
        return w + sz * 0.14
    cjk = ord(ch) > 0x2e80
    f = font(F_CJK if cjk else F_LAT, sz * (0.92 if cjk else 1.0))
    bb = f.getbbox(ch)
    pad = int(sz * 0.3)
    wpx, hpx = bb[2] + 2 * pad, int(sz * 1.6) + 2 * pad
    im = Image.new('L', (max(4, wpx), hpx), 0)
    asc = f.getmetrics()[0]
    ImageDraw.Draw(im).text((pad, pad), ch, font=f, fill=255)
    im = im.rotate(rot, resample=Image.BICUBIC, center=(pad + bb[2] / 2, pad + asc))
    a = np.asarray(im, np.float32) / 255 * ink
    px, py = int(round(x - pad)), int(round(base - asc - pad + dy))
    sub = canvas[py:py + a.shape[0], px:px + a.shape[1]]
    np.maximum(sub, a[:sub.shape[0], :sub.shape[1]], out=sub)
    adv = f.getlength(ch)
    return adv * (1.0 if cjk else 0.98) + r.uniform(-1, 2)


def write(canvas, text, x, base, size, seed, spacing=1.0, ink=1.0):
    r = np.random.default_rng(seed)
    for ch in text:
        if ch == ' ':
            x += size * 0.28 * spacing
            continue
        x += glyph(canvas, ch, x, base + r.uniform(-1.5, 1.5), size, int(r.integers(1 << 30)), ink) * spacing
    return x


# ------------------------------------------------------------------ paper
def paper(h, w, tone=238, seed=0):
    r = np.random.default_rng(seed)
    p = np.full((h, w), tone, np.float32)
    lo = cv2.resize(r.normal(0, 1, (h // 64 + 2, w // 64 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    p += lo * 2.5
    fine = cv2.GaussianBlur(r.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8)
    p += fine * 2.2
    return p


# ------------------------------------------------------------------ notebook page (local coords, ruled lines horizontal)
PW, PH = 2600, 3400
LS = 95                                   # ruled line spacing
LINE0 = 160


def ruled_y(k):
    return LINE0 + k * LS


def notebook_page():
    pg = paper(PH, PW, 239, seed=1)
    lines = np.zeros((PH, PW), np.float32)
    for k in range(0, (PH - LINE0) // LS + 1):
        y = ruled_y(k)
        cv2.line(lines, (0, y), (PW, y + 2), 1.0, 2, cv2.LINE_AA)
    pg = pg * (1 - 0.2 * lines)
    ink = np.zeros((PH, PW), np.float32)

    # earlier homework at the top of the page (my own exercise, written smaller and faster)
    write(ink, '7. 已知 f(x) = x² − 4x + 3, 0 ≤ x ≤ 5', 70, ruled_y(1) - 10, 62, 11)
    write(ink, '(1) 求 f(x) 的单调区间;', 120, ruled_y(2) - 10, 60, 12)
    write(ink, '解: 对称轴为 x = 2', 120, ruled_y(3) - 10, 60, 13)
    write(ink, '∴ f(x) 在 [0, 2] 上单调递减, 在 [2, 5] 上单调递增', 160, ruled_y(4) - 10, 58, 14)
    write(ink, '(2) f(x)min = f(2) = −1, f(x)max = f(5) = 8', 120, ruled_y(5) - 10, 58, 15)

    # ---- the proof
    X0, K0 = 330, 9
    write(ink, '证明:', X0, ruled_y(K0) - 12, 82, 101)
    write(ink, '∵ 我喜欢你', X0 + 25, ruled_y(K0 + 1) - 12, 82, 102)
    end3 = write(ink, '∴ ∠L + ∠O + ∠V + ∠e', X0 - 15, ruled_y(K0 + 2) - 12, 92, 103, spacing=1.02)
    write(ink, '= 520°', X0 + 75, ruled_y(K0 + 3) - 12, 96, 104)
    write(ink, '∴ Love = 520.', X0 + 45, ruled_y(K0 + 4) - 12, 96, 105)

    # ---- the figure (same construction as the reference: ABCD with diagonals meeting at O,
    #      dashed outline through L, e, V)
    # shape relative to D, measured against the ruled lines; L sits just right of "∠e"
    shape = {'D': (0, 0), 'C': (281, -41), 'B': (362, 132), 'A': (74, 176), 'L': (-82, 210),
             'e': (376, 502), 'V': (504, 101), 'X': (180, 65)}
    sc = 1.2
    Lpos = np.array([end3 + 120, ruled_y(K0 + 2) - 36.0])
    Dpos = Lpos - sc * np.array(shape['L'], float)

    def Q(k):
        return Dpos + sc * np.array(shape[k], float)
    D, C, B, A, L_, E_, V_, X_ = (Q(k) for k in 'DCBALeVX')
    s = 10
    for a, b in ((D, B), (A, C), (D, A), (A, B), (C, B)):
        stroke(ink, [a, b], width=4.4, wobble=1.3, seed=s); s += 1
    for a, b in ((D, L_), (L_, E_), (E_, V_), (V_, C)):
        dashed(ink, a, b, seed=s); s += 1
    labels = [('D', D, (-62, -14)), ('C', C, (-14, -28)), ('B', B, (18, 30)), ('A', A, (-18, 92)),
              ('O', X_, (-40, -26)), ('V', V_, (24, -8)), ('e', E_, (18, 62)), ('L', L_, (-70, 40))]
    for ch, p, (ox, oy) in labels:
        glyph(ink, ch, p[0] + ox, p[1] + oy, 84, s); s += 1
    # ink look: gel pen, slight bleed
    ink = np.clip(cv2.GaussianBlur(ink, (0, 0), 0.6) * 1.08, 0, 1)
    # writing on the back of the sheet shows through faintly (mirrored)
    back = np.zeros((PH, PW), np.float32)
    for k in range(2, 30, 1):
        if k % 7 in (3, 6):
            continue
        txt = ['解: 由题意得 a + b = 3, ab = −1', '∴ a² + b² = (a + b)² − 2ab = 11', 'f(1) = 2, f(−1) = 0',
               '设 x₁ < x₂, 则 f(x₁) − f(x₂) < 0', '∴ 函数在区间上单调递增', 'y = 2x + 1, x = 3']
        write(back, txt[k % len(txt)], 140 + (k * 37) % 160, ruled_y(k) - 12, 70, 300 + k)
    back = cv2.GaussianBlur(back[:, ::-1], (0, 0), 3.0)
    pg = pg * (1 - 0.035 * back)
    pg = pg * (1 - ink) + 52 * ink
    np.save(os.path.join(HERE, 'page_gray.npy'), pg.astype(np.float32))
    alpha = np.ones((PH, PW), np.float32)
    # torn-free but slightly rounded corners / soft edge
    alpha[:, :2] = 0.6; alpha[:2, :] = 0.6
    return pg, alpha


# ------------------------------------------------------------------ underlying sheets
def printed_page(h, w, seed):
    pg = paper(h, w, 232, seed)
    im = Image.fromarray(np.zeros((h, w), np.uint8))
    d = ImageDraw.Draw(im)
    lines = ['§3.2 函数的基本性质', '3.2.1 单调性与最大(小)值',
             '例 1  求下列函数的单调区间, 并指出其在区间上是增函数还是减函数:',
             '(1) y = x² − 6x + 5;        (2) y = −3/x;        (3) y = |x − 2|.',
             '解: (1) 函数 y = x² − 6x + 5 的图象开口向上, 对称轴为 x = 3,',
             '所以函数在 (−∞, 3] 上单调递减, 在 [3, +∞) 上单调递增.',
             '练习', '1. 判断函数 f(x) = −2x + 1 在 R 上的单调性, 并用定义证明.',
             '2. 已知函数 f(x) = x² + 2ax + 2, x ∈ [−5, 5].',
             '(1) 当 a = −1 时, 求函数 f(x) 的最大值和最小值;',
             '(2) 求实数 a 的取值范围, 使 y = f(x) 在区间 [−5, 5] 上是单调函数.',
             '3. 求函数 y = 2^(x² − 2x) 的值域.', '4. 设 g(x) = lg(4 − x²), 求 g(x) 的定义域与单调区间.',
             '5. 已知 f(x) 是定义在 R 上的奇函数, 当 x > 0 时, f(x) = x² − 2x,',
             '求 f(x) 的解析式.', '6. 证明: 函数 f(x) = x + 1/x 在 (0, 1) 上单调递减.']
    y = 90
    i = 0
    while y < h - 60:
        t = lines[i % len(lines)]
        sz = 46 if i % len(lines) in (0, 1, 6) else 38
        d.text((90 if i % len(lines) not in (4, 5, 9, 10, 14) else 140, y), t, font=font(F_PRINT, sz), fill=255)
        y += int(sz * 1.75); i += 1
    a = np.asarray(im, np.float32) / 255
    return pg * (1 - a) + 30 * a


def worksheet(h, w, seed):
    pg = paper(h, w, 236, seed)
    ink = np.zeros((h, w), np.float32)
    im = Image.fromarray(np.zeros((h, w), np.uint8)); d = ImageDraw.Draw(im)
    d.text((80, 70), '课时作业(十二)  函数的单调性', font=font(F_PRINT, 44), fill=255)
    d.text((80, 160), '1. 函数 f(x) = 3 − x 的单调递减区间是 ________.', font=font(F_PRINT, 36), fill=255)
    d.text((80, 520), '2. 若函数 f(x) = x² + (a − 1)x + 3 在 [2, +∞) 上单调递增,', font=font(F_PRINT, 36), fill=255)
    d.text((80, 580), '   则实数 a 的取值范围是 ________.', font=font(F_PRINT, 36), fill=255)
    a = np.asarray(im, np.float32) / 255
    write(ink, '解: f(x) = 3 − x', 120, 300, 66, 201)
    write(ink, '∴ 在 R 上单调递减', 160, 400, 66, 202)
    write(ink, '− (a − 1)/2 ≤ 2', 140, 720, 66, 203)
    write(ink, '∴ a ≥ −3', 160, 820, 66, 204)
    write(ink, 'f(2) = 3', 600, 940, 62, 205)
    ink = np.clip(cv2.GaussianBlur(ink, (0, 0), 0.6), 0, 1)
    return pg * (1 - a) + 30 * a, ink


def place(scene, img, alpha, angle_deg, cx, cy, shadow=0.35):
    """Rotate img by angle about its centre and place its centre at (cx, cy) with a soft shadow."""
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    M[0, 2] += cx - w / 2; M[1, 2] += cy - h / 2
    Hs, Ws = scene.shape
    wi = cv2.warpAffine(img, M, (Ws, Hs), flags=cv2.INTER_LINEAR)
    wa = cv2.warpAffine(alpha, M, (Ws, Hs), flags=cv2.INTER_LINEAR)
    sh = cv2.GaussianBlur(np.roll(np.roll(wa, 10, 0), 6, 1), (0, 0), 14)
    scene *= (1 - shadow * sh * (1 - wa))
    scene[:] = scene * (1 - wa) + wi * wa
    return M


if __name__ == '__main__':
    SH_, SW_ = 7000, 7400
    OX, OY = 900, 2300                       # room for the far side of the desk
    scene = np.full((SH_, SW_), 120, np.float32)
    # textbook spread across the whole desk
    tb = printed_page(6400, 3500, 31)
    place(scene, tb, np.ones_like(tb), -17, 1700 + OX, 2000 + OY - 600, shadow=0.0)
    tb2 = printed_page(6400, 3500, 32)
    place(scene, tb2, np.ones_like(tb2), -17, 4300 + OX + 60, 1250 + OY - 600, shadow=0.0)
    # book gutter: soft dark fold along the seam of the spread
    g = np.zeros_like(scene)
    ang = math.radians(17)
    cx, cy = 3000 + OX, 1625 + OY - 600       # seam passes near here
    d = np.array([math.sin(ang), math.cos(ang)])     # seam direction (rotated vertical)
    yy, xx = np.mgrid[0:SH_, 0:SW_].astype(np.float32)
    dist = (xx - cx) * d[1] - (yy - cy) * d[0]
    scene *= 1 - 0.42 * np.exp(-(dist / 45) ** 2) - 0.12 * np.exp(-(dist / 260) ** 2)
    # worksheet at the lower left
    ws, wsi = worksheet(1700, 2200, 41)
    ws = ws * (1 - wsi) + 55 * wsi
    place(scene, ws, np.ones_like(ws), -22, 1500 + OX, 3500 + OY, shadow=0.35)
    # the notebook page on top
    pg, pa = notebook_page()
    M = place(scene, pg, pa, 20, 2750 + OX, 2100 + OY, shadow=0.4)
    np.save(os.path.join(HERE, 'page_to_scene.npy'), M)
    cv2.imwrite(os.path.join(HERE, 'scene.png'), np.clip(scene, 0, 255).astype(np.uint8))
    # handy for framing: where the proof/figure centre ends up
    c = M @ np.array([900, ruled_y(10), 1.0])
    np.save(os.path.join(HERE, 'focus.npy'), c)
    print('scene', scene.shape, 'focus', c)
