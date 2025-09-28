# server.py
import io, os, time, base64, requests
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2

# ML / обробка
from skimage.color import rgb2lab
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
from sklearn.neighbors import NearestCentroid

# -------------------- FastAPI --------------------
app = FastAPI(title="Car Color API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# -------------------- Helpers --------------------
def read_image(file: UploadFile) -> np.ndarray:
    """Читання зображення як RGB np.array(H,W,3)"""
    data = file.file.read()
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))

def read_mask(file: UploadFile, target_hw) -> np.ndarray:
    """Читання маски як uint8 (0/255) розміру як у зображення"""
    data = file.file.read()
    m = np.array(Image.open(io.BytesIO(data)).convert("L"))
    if m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    return m

def mask_to_png_base64(mask_uint8: np.ndarray) -> str:
    """Повертає base64(PNG) з маски"""
    buf = io.BytesIO()
    Image.fromarray(mask_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# =================================================
# ===============   /refine (GrabCut)   ===========
# =================================================
@app.post("/refine")
async def refine(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
    w: int = Form(0),
    h: int = Form(0),
):
    """Уточнення маски в межах ROI (або всього кадру) через GrabCut."""
    img = read_image(image)                 # HxWx3 RGB
    rough = read_mask(mask, img.shape[:2])  # HxW (0/255)

    H, W = img.shape[:2]
    if w > 0 and h > 0:
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        img_roi = img[y0:y1, x0:x1].copy()
        rough_roi = rough[y0:y1, x0:x1].copy()
    else:
        x0 = y0 = 0; x1 = W; y1 = H
        img_roi = img; rough_roi = rough

    # 0/255 -> мітки GrabCut: 3=fg, 2=bg
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

# =================================================
# ==========  /grow_similar (без AI)  =============
# =================================================
@app.post("/grow_similar")
async def grow_similar(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    add_mirror: int = Form(1),          # 1 = додати дзеркальну пару
    segments: int = Form(2500),         # більше суперпікселів -> точніше
    lbp_radius: int = Form(1),
    lbp_points: int = Form(8),
    thresh: float = Form(0.85),         # суворіше
    band_margin: float = Form(0.18),    # смуга навколо зразка (0..0.5)
    keep_k: int = Form(4),              # скільки найближчих компонент лишати
):
    """
    «Розумно: схожі» — SLIC суперпікселі + колір (Lab) + текстура (LBP).
    Потім фільтр за площею/позицією і вибираємо K найближчих компонент.
    """
    img  = read_image(image)
    rough= read_mask(mask, img.shape[:2])
    H, W = rough.shape

    # Seeds
    k_pos = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    k_neg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    seed_pos = cv2.erode(rough, k_pos)
    ring     = cv2.dilate(rough, k_neg)
    seed_neg = cv2.subtract(ring, rough)

    ys, xs = np.where(rough > 0)
    if len(xs) == 0:
        return {"mask_png_base64": mask_to_png_base64(rough)}
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # Дозволена зона навколо вибраної + дзеркало
    mx = int(W * float(band_margin)); my = int(H * float(band_margin))
    x0b = max(0, x0 - mx); x1b = min(W, x1 + mx)
    y0b = max(0, y0 - my); y1b = min(H, y1 + my)
    allowed = np.zeros((H, W), np.uint8)
    allowed[y0b:y1b, x0b:x1b] = 255
    if int(add_mirror) == 1:
        xm0 = max(0, W - x1b); xm1 = min(W, W - x0b)
        allowed[y0b:y1b, xm0:xm1] = 255

    # SLIC
    seg = slic(img, n_segments=int(segments), compactness=25, start_label=0)
    n = seg.max() + 1

    # Ознаки
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

    # Позитивні/негативні суперпікселі
    pos_idx = [i for i in range(n) if (seed_pos[seg==i] > 0).sum() > 0.2*areas[i]]
    neg_idx = [i for i in range(n) if (seed_neg[seg==i] > 0).sum() > 0.2*areas[i]]
    if len(pos_idx) == 0:
        cx, cy = xs.mean(), ys.mean()
        d = ((cents[:,0]-cx)**2 + (cents[:,1]-cy)**2)
        pos_idx = list(np.argsort(d)[:max(5, n//50)])
    if len(neg_idx) == 0:
        neg_idx = [i for i in range(n) if (rough[seg==i] > 0).sum() < 0.01*areas[i]]

    # Класифікація
    Xpos = feats[pos_idx]
    Xneg = feats[neg_idx] if len(neg_idx)>0 else feats
    ylbl = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))]).astype(int)
    Xall = np.concatenate([Xpos, Xneg], axis=0)
    clf  = NearestCentroid().fit(Xall, ylbl)
    c0, c1 = clf.centroids_
    d0 = np.linalg.norm(feats - c0, axis=1) + 1e-6
    d1 = np.linalg.norm(feats - c1, axis=1) + 1e-6
    score = d0 / (d0 + d1)

    # Початкова маска з порогом і allowed-зоною
    sel = (score > float(thresh))
    init = np.zeros((H, W), np.uint8)
    for i in np.where(sel)[0]:
        cx, cy = map(int, cents[i])
        if allowed[min(H-1, max(0, cy)), min(W-1, max(0, cx))] == 0:
            continue
        init[seg == i] = 255

    # Постобробка по компонентам (схожа площа/рівень по вертикалі)
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

# =================================================
# ===============  /sam_point (AI)  ===============
# =================================================
@app.post("/sam_point")
async def sam_point(
    image: UploadFile = File(...),
    x: int = Form(...),    # клік у пікселях (у тому ж масштабі, що й фото з фронта)
    y: int = Form(...),
    multimask: int = Form(0),   # 0 = одна найкраща маска
):
    """
    Проксі до Replicate SAM-2: приймає фото + одну точку, повертає маску PNG (base64).
    Потрібна змінна середовища REPLICATE_API_TOKEN.
    """
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        return {"error": "Missing REPLICATE_API_TOKEN"}

    # 1) Читаємо фото і вантажимо його у тимчасовий файл на Replicate
    data = image.file.read()
    up = requests.post(
        "https://api.replicate.com/v1/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("image.png", data, "image/png")},
        timeout=60
    )
    up.raise_for_status()
    img_url = up.json()["url"]

    # 2) Стартуємо предікт
    payload = {
        # Узагальнена назва SAM-2; Replicate мапить на актуальну версію
        "model": "meta/sam-2",
        "input": {
            "image": img_url,
            "points": [[int(x), int(y)]],
            "labels": [1],  # 1 = foreground (виділити)
            "multimask_output": bool(int(multimask))
        }
    }
    run = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=60
    )
    run.raise_for_status()
    pred = run.json()
    get_url = pred["urls"]["get"]

    # 3) Чекаємо завершення
    status = pred.get("status")
    while status in ("starting", "processing"):
        time.sleep(1.5)
        rr = requests.get(get_url, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        rr.raise_for_status()
        pred = rr.json()
        status = pred.get("status")

    if status != "succeeded":
        return {"error": f"sam failed: {status}"}

    # 4) Витягуємо посилання на PNG з маскою
    output = pred.get("output", {})
    # можливі різні формати від моделей; обробимо кілька варіантів
    mask_url = None
    if isinstance(output, dict):
        if "masks" in output:
            m = output["masks"]
            mask_url = m[0] if isinstance(m, list) else m
        elif "mask" in output:
            mask_url = output["mask"]
        elif "segmentation" in output:
            mask_url = output["segmentation"]
    elif isinstance(output, list) and len(output) > 0:
        mask_url = output[0]

    if not mask_url:
        return {"error": "mask url not found in SAM output"}

    mask_png = requests.get(mask_url, timeout=60)
    mask_png.raise_for_status()
    b64 = base64.b64encode(mask_png.content).decode("utf-8")
    return {"mask_png_base64": b64}

# =================================================
# ==============  локальний запуск  ===============
# =================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=10000, reload=True)
