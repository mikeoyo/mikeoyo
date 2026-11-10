"""Phone-camera render of the notebook scene, as a Live Photo: still, then a short handheld
move (pull back, drift in), settling back on the still."""
import os, sys, math, subprocess
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1440, 1080
FPS = 60
DUR = 7.5
SCENE = cv2.imread(os.path.join(HERE, 'scene.png'), cv2.IMREAD_GRAYSCALE).astype(np.float32)
SCENE = cv2.GaussianBlur(SCENE, (0, 0), 0.5)
SCENE2 = cv2.resize(SCENE, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)   # mip for wide shots
M = np.load(os.path.join(HERE, 'page_to_scene.npy'))
SH_, SW_ = SCENE.shape


def page2scene(u, v):
    return M @ np.array([u, v, 1.0])


FOCUS = page2scene(1150, 1150)          # between the proof and the figure
FOC = 1150.0                            # focal length in output px (phone main camera, 4:3)


def homography(P, d, tilt, yaw, roll):
    """Scene plane (Z=0, X right, Y down, Z into the desk) -> image, camera looking at P."""
    t, y, r = map(math.radians, (tilt, yaw, roll))
    g = np.array([-math.sin(y), math.cos(y), 0.0])                 # toward the viewer, on the desk
    C = np.array([P[0], P[1], 0.0]) + d * (math.sin(t) * g + math.cos(t) * np.array([0, 0, -1.0]))
    f = (np.array([P[0], P[1], 0.0]) - C) / d
    x = np.cross(g, f); x /= np.linalg.norm(x)
    yv = np.cross(f, x)
    cr, sr = math.cos(r), math.sin(r)
    x, yv = cr * x + sr * yv, -sr * x + cr * yv
    R = np.stack([x, yv, f])
    tv = -R @ C
    K = np.array([[FOC, 0, W / 2], [0, FOC, H / 2], [0, 0, 1.0]])
    return K @ np.column_stack([R[:, 0], R[:, 1], tv])


KEY = dict(dx=0.0, dy=0.0, d=1750.0, tilt=38.0, yaw=0.0, roll=0.0)
# start of the live clip: the phone is higher, further back and turned a little
LIVE0 = dict(dx=-220.0, dy=-380.0, d=2900.0, tilt=45.0, yaw=-7.0, roll=4.0)
T_STILL1, T_FADE, T_MOVE_END = 2.9, 0.3, 6.35


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3 * (1 + 3 * u) if False else u * u * (3 - 2 * u)


def cam_at(T):
    """Camera parameters at time T (seconds)."""
    if T < T_STILL1 or T >= T_MOVE_END:
        return dict(KEY), 0.0
    a, b = T_STILL1, T_MOVE_END
    u = (T - a) / (b - a)
    # fast settle early, slow creep at the end (like moving a phone in, then holding)
    e = 1 - (1 - u) ** 2.6
    c = {k: LIVE0[k] + (KEY[k] - LIVE0[k]) * e for k in KEY}
    # handheld shake, fading out as the shot settles
    amp = (1 - e) ** 0.7
    s = lambda f, p: math.sin(2 * math.pi * f * T + p)
    c['dx'] += amp * (38 * s(0.9, 0.3) + 14 * s(2.3, 1.1))
    c['dy'] += amp * (30 * s(0.7, 2.0) + 12 * s(2.9, 0.4))
    c['roll'] += amp * (1.2 * s(0.8, 1.7) + 0.4 * s(3.1, 0.2))
    c['tilt'] += amp * (1.0 * s(0.6, 0.9))
    return c, (1 - e)


def H_of(c):
    P = FOCUS + np.array([c['dx'], c['dy']])
    return homography(P, c['d'], c['tilt'], c['yaw'], c['roll'])


def project(c):
    Hm = H_of(c)
    # choose the mip level from the scale at the image centre
    J = np.linalg.inv(Hm)
    p0 = J @ [W / 2, H / 2, 1]; p1 = J @ [W / 2 + 1, H / 2, 1]
    scale = np.linalg.norm(p1[:2] / p1[2] - p0[:2] / p0[2])     # scene px per output px
    if scale > 1.6:
        S = np.diag([0.5, 0.5, 1.0])
        return cv2.warpPerspective(SCENE2, Hm @ np.linalg.inv(S), (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
    return cv2.warpPerspective(SCENE, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
# window light from the upper right + the phone/hand's soft shadow at the lower left
LIGHT = (0.93 + 0.07 * (XX / W) - 0.05 * (YY / H))
HAND = np.exp(-(((XX - 0.08 * W) / (0.55 * W)) ** 2 + ((YY - 1.15 * H) / (0.45 * H)) ** 2))
LIGHT = LIGHT * (1 - 0.30 * HAND)
r = np.sqrt(((XX - W / 2) / (W / 2)) ** 2 + ((YY - H / 2) / (H / 2)) ** 2)
LIGHT *= np.clip(1 - 0.16 * np.clip(r - 0.45, 0, None) ** 1.6, 0.7, 1)


def lens(img, focus_y, live):
    """Shallow depth of field: sharp band around focus_y, softer toward the far top and near bottom."""
    b1 = cv2.GaussianBlur(img, (0, 0), 1.6)
    b2 = cv2.GaussianBlur(img, (0, 0), 3.6)
    dist = (focus_y - YY) / H                       # >0 = farther away (top of frame)
    far = np.clip((dist - 0.12) / 0.38, 0, 1)
    near = np.clip((-dist - 0.32) / 0.3, 0, 1) * 0.6
    m = np.maximum(far, near)
    w1 = np.clip(m * 2, 0, 1); w2 = np.clip(m * 2 - 1, 0, 1)
    out = img * (1 - w1) + b1 * w1
    return out * (1 - w2) + b2 * w2


NOISE_STILL = np.random.default_rng(7).normal(0, 1, (H, W)).astype(np.float32)


def render(n):
    T = n / FPS
    c, motion = cam_at(T)
    if motion > 0.05:                         # motion blur from the moving phone
        dt = 1 / FPS / 2
        acc = sum(project(cam_at(T + k * dt)[0]) for k in (-1, 0, 1)) / 3
        img = acc
    else:
        img = project(c)
    Hm = H_of(c)
    fy = (Hm @ np.array([*FOCUS, 1.0])); fy = fy[1] / fy[2]
    img = lens(img, fy, motion)
    img = img * LIGHT
    live = T_STILL1 <= T < T_MOVE_END
    if live:
        # the live clip is a touch darker and softer than the still, like a phone's Live Photo video
        u = (T - T_STILL1) / (T_MOVE_END - T_STILL1)
        expo = 1 - 0.09 * math.exp(-((u - 0.12) / 0.2) ** 2)
        img = img * expo
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        noise = np.random.default_rng(n).normal(0, 1, (H, W)).astype(np.float32) * 3.0
    else:
        noise = NOISE_STILL * 2.0
    img = img + cv2.GaussianBlur(noise, (0, 0), 0.7) * 1.6
    # cross-fade from the still into the start of the live clip (the ghosting a Live Photo shows)
    if T_STILL1 <= T < T_STILL1 + T_FADE:
        a = (T - T_STILL1) / T_FADE
        still = still_frame()
        img = still * (1 - a) + img * a
    # tone: high-key black and white, gentle contrast
    img = np.clip((img - 18) * 1.05, 0, 255)
    return img


_still = None


def still_frame():
    global _still
    if _still is None:
        _still = render(0)
    return _still


def to_bgr(img):
    g = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.merge([g, g, g])


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'frames':
        for t in sys.argv[2:]:
            cv2.imwrite(os.path.join(HERE, 'f_%05.2f.jpg' % float(t)), to_bgr(render(int(round(float(t) * FPS)))),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    elif mode == 'video':
        out = sys.argv[2]
        cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
               '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16',
               '-pix_fmt', 'yuv420p', out]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for n in range(int(round(DUR * FPS))):
            p.stdin.write(to_bgr(render(n)).tobytes())
        p.stdin.close(); p.wait()
        print('done')
