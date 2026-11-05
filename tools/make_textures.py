"""Generate the tobacco-leaf textures for Cigar Anatomy 3D.

Run from the project folder:  python3 tools/make_textures.py
Writes JPGs into assets/tex/. Leaf textures tile seamlessly: x runs around
the cigar, y runs along its length. The foot texture is a single disc.
"""
import os
import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "tex")
os.makedirs(OUT, exist_ok=True)
TAU = np.pi * 2


def spec(n, fmin, fmax, beta=2.0, seed=0, aniso=(1.0, 1.0)):
    """Tileable band-limited noise, unit variance."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n) * n
    kx, ky = np.meshgrid(f, f)
    k = np.sqrt((kx * aniso[0]) ** 2 + (ky * aniso[1]) ** 2)
    amp = np.where((k >= fmin) & (k <= fmax), (k + 1e-6) ** (-beta / 2), 0.0)
    F = amp * np.exp(1j * rng.uniform(0, TAU, (n, n)))
    out = np.real(np.fft.ifft2(F))
    return out / (out.std() + 1e-9)


def normal_map(h, strength):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * strength
    dy = (np.roll(h, 1, 0) - np.roll(h, -1, 0)) * strength  # image rows run down, v runs up
    nz = np.ones_like(h)
    l = np.sqrt(dx * dx + dy * dy + nz * nz)
    n = np.stack([-dx / l, -dy / l, nz / l], -1)
    return ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def save(arr, name, q=92):
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT, name), quality=q, subsampling=0)
    print("wrote", name, arr.shape)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def vein_family(n, a, b, width, warp, seed):
    """Parallel, wavy raised veins: lines where a*x + b*y is an integer (tileable)."""
    y, x = np.mgrid[0:n, 0:n] / n
    w = spec(n, 1, 7, 2.2, seed) * warp + spec(n, 6, 24, 1.6, seed + 1) * warp * 0.25
    ph = a * x + b * y + w
    fr = ph - np.floor(ph)
    d = np.minimum(fr, 1 - fr) / np.hypot(a, b)  # distance in tile units
    wid = width * (0.65 + 0.35 * np.tanh(spec(n, 2, 10, 2, seed + 2)))
    strength = 0.55 + 0.45 * np.tanh(1.2 * spec(n, 1, 6, 2, seed + 3))
    return np.exp(-(d / wid) ** 2) * strength


def leaf(name, n, seed, veins=1.0, blotch=1.0, coarse=False, tooth=False):
    v1 = vein_family(n, 3, 2, 0.0062 if not coarse else 0.008, 0.07, seed)
    v2 = vein_family(n, 7, -3, 0.0016, 0.05, seed + 10) * (spec(n, 2, 8, 2, seed + 11) > -0.2)
    v3 = vein_family(n, -5, 9, 0.0011, 0.04, seed + 20) * (spec(n, 2, 8, 2, seed + 21) > 0.3)
    net = 1 - np.abs(spec(n, 10, 70, 1.3, seed + 30))
    net = np.clip(net, 0, 1) ** 9
    wrinkle = spec(n, 2, 16, 2.2, seed + 40, aniso=(1.0, 0.55))
    grain = spec(n, 180, n // 2, 0.4, seed + 50)
    cells = spec(n, 60, 220, 0.8, seed + 60)

    h = (v1 * 0.55 + v2 * 0.25 + v3 * 0.16) * veins + net * 0.22 + wrinkle * 0.10 + cells * 0.035 + grain * 0.02
    if tooth:
        t = spec(n, 120, 400, 0.5, seed + 70)
        h += np.clip(t - 2.2, 0, None) * 0.35
    if coarse:
        h += spec(n, 4, 30, 1.8, seed + 80) * 0.08

    mott = spec(n, 1, 6, 2.4, seed + 90)
    mott2 = spec(n, 5, 30, 1.8, seed + 91)
    spots = np.clip((spec(n, 25, 90, 1.0, seed + 92) - 2.5) * 1.5, 0, 1)
    lum = 0.96 + 0.026 * mott * blotch + 0.016 * mott2 * blotch + 0.018 * (v1 + 0.5 * v2) * veins - 0.025 * net + 0.018 * grain - 0.06 * spots
    if tooth:
        lum -= np.clip(spec(n, 120, 400, 0.5, seed + 70) - 2.2, 0, None) * 0.25
    hue = spec(n, 1, 8, 2.2, seed + 93)
    rgb = np.stack([lum * (1 + 0.015 * hue), lum, lum * (1 - 0.025 * hue)], -1)
    small = lambda a: np.asarray(Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8)).resize((min(n, 1024),) * 2, Image.LANCZOS))
    save(small(rgb), f"{name}_albedo.jpg", 86)
    save(normal_map(h, 6.0 if not coarse else 7.5), f"{name}_normal.jpg", 86)
    if tooth:
        return
    rough = 0.74 + 0.06 * mott2 - 0.04 * (v1 + 0.5 * v2) * veins + 0.04 * grain + 0.1 * spots
    save(small(np.repeat(rough[..., None], 3, -1)), f"{name}_rough.jpg", 86)


def filler(name, n, seed):
    # long filler: crinkled, folded leaf, folds running along the length (y)
    folds = spec(n, 3, 30, 2.0, seed, aniso=(1.0, 0.25))
    ridge = 1 - np.abs(spec(n, 4, 40, 1.8, seed + 1, aniso=(1.0, 0.4)))
    crinkle = spec(n, 10, 70, 1.4, seed + 2)
    h = folds * 0.35 + ridge ** 3 * 0.5 + crinkle * 0.06
    mott = spec(n, 1, 10, 2.0, seed + 3)
    lum = 0.9 + 0.06 * mott + 0.08 * (ridge ** 3 - 0.3) + 0.012 * crinkle
    lum *= 0.82 + 0.18 * smoothstep(-1.2, 0.8, folds)  # shade the creases
    hue = spec(n, 1, 8, 2, seed + 4)
    rgb = np.stack([lum * (1 + 0.04 * hue), lum, lum * (1 - 0.07 * hue)], -1)
    save(rgb, f"{name}_albedo.jpg")
    save(normal_map(h, 4.5), f"{name}_normal.jpg")
    rough = 0.85 + 0.06 * spec(n, 10, 100, 1, seed + 5)
    save(np.repeat(rough[..., None], 3, -1), f"{name}_rough.jpg")


def foot(name, n, seed):
    """End-on view of the foot: bunched, accordion-folded filler leaves."""
    rng = np.random.default_rng(seed)
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 - 1
    r = np.hypot(xx, yy)
    # each leaf is a region with its own fold direction and spacing
    k = 34
    pts = []
    while len(pts) < k:
        p = rng.uniform(-1, 1, 2)
        if np.hypot(*p) < 0.97:
            pts.append(p)
    pts = np.array(pts)
    wx = spec(n, 2, 9, 2, seed + 1) * 0.07
    wy = spec(n, 2, 9, 2, seed + 2) * 0.07
    d = np.stack([np.hypot(xx + wx - px, yy + wy - py) for px, py in pts], 0)
    order = np.argsort(d, 0)
    nearest = order[0]
    d0 = np.take_along_axis(d, order[:1], 0)[0]
    d1 = np.take_along_axis(d, order[1:2], 0)[0]
    edge = smoothstep(0.0, 0.025, d1 - d0)  # 0 on the boundary between leaves

    ang = rng.uniform(0, np.pi, k)[nearest]
    freq = rng.uniform(110, 170, k)[nearest]
    shade = rng.uniform(-1, 1, k)[nearest]
    proj = xx * np.cos(ang) + yy * np.sin(ang)
    warp = spec(n, 3, 25, 1.8, seed + 3) * 0.9 + spec(n, 20, 90, 1.4, seed + 4) * 0.35 + spec(n, 60, 200, 1.0, seed + 7) * 0.06
    folds = np.sin(freq * proj + 4.0 * warp)
    # leaf edges seen end-on: thin bright lines where the folds cross zero
    edges = np.exp(-(folds / 0.32) ** 2)
    edges *= 0.6 + 0.4 * smoothstep(-1, 1, spec(n, 8, 60, 1.4, seed + 8))  # some edges sit deeper
    gap = smoothstep(0.0, 0.03, d1 - d0)

    h = edges * 0.9 + gap * 0.15 + spec(n, 40, 300, 1.0, seed + 5) * 0.06
    base = np.array([0.40, 0.26, 0.155])
    lighten = 0.22 * smoothstep(0.15, 0.95, r) + 0.09 * shade + 0.05 * spec(n, 4, 40, 1.6, seed + 6)
    leafcol = base[None, None, :] * (1 + 1.3 * lighten[..., None])
    dark = np.array([0.085, 0.055, 0.035])[None, None, :]
    mix = (0.22 + 0.78 * edges) * (0.55 + 0.45 * gap)
    col = dark + (leafcol - dark) * mix[..., None]
    col *= (1 - 0.4 * smoothstep(0.93, 1.0, r))[..., None]  # shadow under the binder
    rgb = np.clip(col * 1.35, 0, 1)
    save(rgb, f"{name}_albedo.jpg")
    save(normal_map(h, 10.0), f"{name}_normal.jpg")


def plant_leaf(name, w, h, seed):
    """A living tobacco leaf, laid flat: x runs base -> tip, y runs edge -> edge.
    Reference: light yellow-green, pale midrib, ~9 pairs of curving side veins,
    puckered (quilted) between the veins."""
    yy, xx = np.mgrid[0:h, 0:w]
    sx = xx / (w - 1)
    t = (yy / (h - 1)) * 2 - 1
    at = np.abs(t)
    n = max(w, h)
    big = lambda f0, f1, b, sd: np.asarray(Image.fromarray(((spec(n, f0, f1, b, sd) * 0.15 + 0.5).clip(0, 1) * 255).astype(np.uint8)).resize((w, h))) / 255.0 - 0.5
    mw = 0.05 * (1 - 0.8 * sx) + 0.006
    midrib = np.exp(-(t / mw) ** 2)
    nveins = 9
    ph = (sx - 0.34 * at ** 0.75) * nveins + big(1, 5, 2, seed) * 0.35
    fr = ph - np.floor(ph)
    d = np.minimum(fr, 1 - fr)
    veins = np.exp(-(d / 0.045) ** 2) * (1 - 0.75 * at) * (at > mw * 0.5)
    # quilting: the blade puffs up between side veins
    quilt = np.sin(np.pi * fr) ** 2 * (1 - at ** 2) * (at > mw)
    net = (1 - np.abs(big(25, 140, 1.2, seed + 1) * 6)).clip(0, 1) ** 8
    hgt = midrib * 1.4 + veins * 0.55 - quilt * 0.0 + quilt * 0.45 + net * 0.08 + big(10, 60, 1.6, seed + 2) * 0.25
    mott = big(2, 12, 2, seed + 3)
    base = np.array([0.50, 0.66, 0.22])              # light yellow-green
    col = base[None, None, :] * (1 + 0.12 * mott[..., None])
    col *= (0.88 + 0.16 * quilt)[..., None]          # puffed areas catch more light
    pale = np.array([0.80, 0.86, 0.60])
    col = col * (1 - midrib[..., None] * 0.9) + pale * midrib[..., None] * 0.9
    col = col * (1 - veins[..., None] * 0.45) + pale * veins[..., None] * 0.45
    col *= (1 - 0.05 * net)[..., None]
    save(np.clip(col, 0, 1), f"{name}_albedo.jpg", 88)
    save(normal_map(hgt * 2.5, 1.6), f"{name}_normal.jpg", 88)


if __name__ == "__main__":
    leaf("wrapper", 2048, 11)
    leaf("tooth", 2048, 11, tooth=True)
    leaf("binder", 1024, 211, veins=1.3, blotch=1.8, coarse=True)
    filler("filler", 1024, 311)
    foot("foot", 1024, 411)
    plant_leaf("plantleaf", 1024, 512, 511)
