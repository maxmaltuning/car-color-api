# server.py
import io, os, time, base64, traceback, requests
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2

# ML
from skimage.color import rgb2lab
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
from sklearn.neighbors import NearestCentroid

app = FastAPI(title="Car Color API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ----------------- helpers -----------------
def read_image(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))

def read_mask(file: UploadFile, target_hw) -> np.ndarray:
    data = file.file.read()
    m = np.array(Image.open(io.BytesIO(data)).convert("L"))
    if m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    return m.astype("uint8")

def mask_to_png_base64(mask_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(mask_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --------------- health/debug ---------------
@app.get("/env_check")
def env_check():
    return {"has_token": bool(os.environ.get("REPLICATE_API_TOKEN"))}

# ================== /refine ==================
@app.post("/refine")
async def refine(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    x: int = Form(0), y: int = Form(0),
    w: int = Form(0), h: int = Form(0),
):
    """Уточнення маски (GrabCut) в межах ROI або по всьому кадру."""
    img = read_image(image)
    rough = read_mask(mask, img.shape[:2])
    H, W = img.shape[:2]

    if w > 0 and h > 0:
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        img_roi = img[y0:y1, x0:x1].copy()
        rough_roi = rough[y0:y1, x0:x1].copy()
    else:
        x0 = y0 = 0; x1 = W; y1 = H
        img_roi = img; rough_roi = rough

    gc_mask = np.where(rough_roi > 0, 3, 2).astype("uint8")
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_roi, gc_mask, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)
    result = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')

    k = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,  k, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k, iterations=2)
    result = cv2.GaussianBlur(result, (3,3), 0)

    full = np.zeros((H, W), dtype='uint8')
    full[y0:y1, x0:x1] = result
    return {"mask_png_base64": mask_to_png_base64(full)}

# ============== /grow_similar (evristics) ==============
@app.post("/grow_similar")
async def grow_similar(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    add_mirror: int = Form(1),
    segments: int = Form(2500),
    lbp_radius: int = Form(1),
    lbp_points: int = Form(8),
    thresh: float = Form(0.85),
    band_margin: float = Form(0.18),
    keep_k: int = Form(4),
):
    img  = read_image(image)
    rough= read_mask(mask, img.shape[:2])
    H, W = rough.shape

    k_pos = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    k_neg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    seed_pos = cv2.erode(rough, k_pos)
    ring     = cv2.dilate(rough, k_neg)
    seed_neg = cv2.subtract(ring, rough)

    ys, xs = np.where(rough > 0)
    if len(xs) == 0:
        return {"mask_png_base64": mask_to_png_base64(rough)}
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()

    mx = int(W * float(band_margin)); my = int(H * float(band_margin))
    x0b = max(0, x0 - mx); x1b = min(W, x1 + mx)
    y0b = max(0, y0 - my); y1b = min(H, y1 + my)
    allowed = np.zeros((H, W), np.uint8)
    allowed[y0b:y1b, x0b:x1b] = 255
    if int(add_mirror) == 1:
        xm0 = max(0, W - x1b); xm1 = min(W, W - x0b)
        allowed[y0b:y1b, xm0:xm1] = 255

    seg = slic(img, n_segments=int(segments), compactness=25, start_label=0)
    n = seg.max() + 1

    lab  = rgb2lab(img).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, P=int(lbp_points), R=int(lbp_radius), method="uniform")

    feats = np.zeros((n, 3 + int(lbp_points)+2), np.float32)
    cents = np.zeros((n, 2), np.float32)
    areas = np.zeros(n, np.int32)
    for i in range(n):
        m = (seg == i)
        if not m.any(): continue
        areas[i] = m.sum()
        ys_i, xs_i = np.where(m)
        cents[i] = [xs_i.mean(), ys_i.mean()]
        L = lab[...,0][m].mean(); A = lab[...,1][m].mean(); B = lab[...,2][m].mean()
        hist, _ = np.histogram(lbp[m], bins=np.arange(int(lbp_points)+3),
                               range=(0, int(lbp_points)+2), density=True)
        feats[i] = np.concatenate([[L,A,B], hist.astype(np.float32)])

    pos_idx = [i for i in range(n) if (seed_pos[seg==i] > 0).sum() > 0.2*areas[i]]
    neg_idx = [i for i in range(n) if (seed_neg[seg==i] > 0).sum() > 0.2*areas[i]]
    if len(pos_idx) == 0:
        cx, cy = xs.mean(), ys.mean()
        d = ((cents[:,0]-cx)**2 + (cents[:,1]-cy)**2)
        pos_idx = list(np.argsort(d)[:max(5, n//50)])
    if len(neg_idx) == 0:
        neg_idx = [i for i in range(n) if (rough[seg==i] > 0).sum() < 0.01*areas[i]]

    Xpos = feats[pos_idx]
    Xneg = feats[neg_idx] if len(neg_idx)>0 else feats
    ylbl = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))]).astype(int)
    Xall = np.concatenate([Xpos, Xneg], axis=0)
    clf  = NearestCentroid().fit(Xall, ylbl)
    c0, c1 = clf.centroids_
    d0 = np.linalg.norm(feats - c0, axis=1) + 1e-6
    d1 = np.linalg.norm(feats - c1, axis=1) + 1e-6
    score = d0 / (d0 + d1)

    sel = (score > float(thresh))
    init = np.zeros((H, W), np.uint8)
    for i in np.where(sel)[0]:
        cx, cy = map(int, cents[i])
        if allowed[min(H-1, max(0, cy)), min(W-1, max(0, cx))] == 0:
            continue
        init[seg == i] = 255

    seed_area = (rough > 0).sum()
    seed_yc   = int(ys.mean())
    num, labimg, stats, centroids = cv2.connectedComponentsWithStats(init, connectivity=8)
    kept = np.zeros((H, W), np.uint8)
    candidates = []
    for label in range(1, num):
        x, y, w, h, a = stats[label]
        cx, cy = centroids[label]
        area_ok = (0.5*seed_area <= a <= 2.5*seed_area)
        y_ok    = (abs(cy - seed_yc) <= 0.18*H)
        if area_ok and y_ok:
            d = (cx - xs.mean())**2 + (cy - ys.mean())**2
            candidates.append((d, label))
    for _, label in sorted(candidates)[:int(keep_k)]:
        kept[labimg == label] = 255

    if int(add_mirror) == 1:
        kept = np.maximum(kept, np.flip(kept, axis=1))

    k = np.ones((3,3), np.uint8)
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, k, iterations=2)
    kept = cv2.morphologyEx(kept, cv2.MORPH_OPEN,  k, iterations=1)

    return {"mask_png_base64": mask_to_png_base64(kept)}

# ================== /sam_point (AI) ==================
@app.post("/sam_point")
async def sam_point(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    multimask: int = Form(0),
):
    """
    SAM через Replicate: фото + одна точка. Повертає PNG маску (base64).
    Потрібна змінна середовища REPLICATE_API_TOKEN.
    """
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        return {"error": "Missing REPLICATE_API_TOKEN (Render → Settings → Environment)"}

    try:
        # 1) Прочитати зображення і зменшити до макс 1280 по довшій стороні
        raw = image.file.read()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        W, H = im.size
        max_side = max(W, H)
        if max_side > 1280:
            scale = 1280 / max_side
            im = im.resize((int(W*scale), int(H*scale)), Image.LANCZOS)

        buf = io.BytesIO()
        im.save(buf, format="PNG"); buf.seek(0)

        # 2) Завантажити файл у Replicate
        up = requests.post(
            "https://api.replicate.com/v1/files",
            headers={"Authorization": f"Token {token}"},
            files={"file": ("image.png", buf.getvalue(), "image/png")},
            timeout=120,
        )
        if not up.ok:
            return {"error": f"Replicate file upload failed: {up.status_code} {up.text}"}
        img_url = up.json().get("url")
        if not img_url:
            return {"error": f"Replicate file upload returned no URL: {up.text}"}

        # 3) Запустити предікт (SAM-2 alias)
        payload = {
            "model": "meta/sam-2",
            "input": {
                "image": img_url,
                "points": [[int(x), int(y)]],
                "labels": [1],
                "multimask_output": bool(int(multimask)),
            },
        }
        run = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if not run.ok:
            return {"error": f"Replicate start failed: {run.status_code} {run.text}"}
        pred = run.json()
        get_url = pred.get("urls", {}).get("get")
        if not get_url:
            return {"error": f"Unexpected Replicate response (no get url): {pred}"}

        # 4) Полінг
        status = pred.get("status"); t0 = time.time()
        while status in ("starting", "processing"):
            if time.time() - t0 > 300:
                return {"error": "Timeout waiting for Replicate (5 min)"}
            time.sleep(1.5)
            rr = requests.get(get_url, headers={"Authorization": f"Token {token}"}, timeout=120)
            if not rr.ok:
                return {"error": f"Replicate poll failed: {rr.status_code} {rr.text}"}
            pred = rr.json()
            status = pred.get("status")

        if status != "succeeded":
            return {"error": f"SAM failed: status={status}, detail={pred}"}

        # 5) Дістати URL маски
        output = pred.get("output")
        mask_url = None
        if isinstance(output, dict):
            if "masks" in output:
                m = output["masks"]
                mask_url = m[0] if isinstance(m, list) else m
            elif "mask" in output:
                mask_url = output["mask"]
            elif "segmentation" in output:
                mask_url = output["segmentation"]
        if not mask_url and isinstance(output, list) and len(output) > 0:
            if isinstance(output[0], str) and output[0].startswith("http"):
                mask_url = output[0]
        if not mask_url:
            return {"error": f"Cannot find mask url in output: {output}"}

        mask_png = requests.get(mask_url, timeout=120)
        if not mask_png.ok:
            return {"error": f"Download mask failed: {mask_png.status_code} {mask_png.text}"}

        b64 = base64.b64encode(mask_png.content).decode("utf-8")
        return {"mask_png_base64": b64}

    except Exception as e:
        # повертаємо текст помилки і пишемо стек у логи Render
        print("=== ERROR in /sam_point ===")
        print(traceback.format_exc())
        return {"error": str(e)}

# ------------- local run -------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=10000, reload=True)
