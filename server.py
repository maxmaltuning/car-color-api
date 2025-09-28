import io, base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2

# нове
from skimage.color import rgb2lab
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
from skimage.measure import regionprops
from sklearn.neighbors import NearestCentroid

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---------- helpers ----------
def read_image(file: UploadFile):
    data = file.file.read()
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return img

def read_mask(file: UploadFile, target_hw):
    data = file.file.read()
    m = np.array(Image.open(io.BytesIO(data)).convert("L"))
    if m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    return m

def mask_to_png_base64(mask_uint8):
    pil = Image.fromarray(mask_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------- існуюче уточнення (GrabCut + ROI) ----------
@app.post("/refine")
async def refine(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
    w: int = Form(0),
    h: int = Form(0),
):
    img = read_image(image)                # HxWx3
    rough = read_mask(mask, img.shape[:2]) # HxW (0/255)

    H, W = img.shape[:2]
    if w > 0 and h > 0:
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        img_roi = img[y0:y1, x0:x1].copy()
        rough_roi = rough[y0:y1, x0:x1].copy()
    else:
        x0 = y0 = 0; x1 = W; y1 = H
        img_roi = img; rough_roi = rough

    gc_mask = np.where(rough_roi > 0, 3, 2).astype('uint8')
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_roi, gc_mask, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)

    result = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')
    kernel = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    result = cv2.GaussianBlur(result, (3,3), 0)

    full = np.zeros((H, W), dtype='uint8')
    full[y0:y1, x0:x1] = result

    return {"mask_png_base64": mask_to_png_base64(full)}

# ---------- НОВЕ: розумне «знайти подібні» (SLIC + LBP + колір) ----------
@app.post("/grow_similar")
async def grow_similar(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    add_mirror: int = Form(1),   # 1 = пробувати віддзеркалення по вертикалі (для ліво/право сидінь)
    segments: int = Form(1200),  # кількість суперпікселів (більше = точніше, але довше)
    lbp_radius: int = Form(1),
    lbp_points: int = Form(8),
    thresh: float = Form(0.6),   # чим нижче — тим більше «схожих» додасть
):
    img = read_image(image)
    rough = read_mask(mask, img.shape[:2])  # 0/255
    H, W = rough.shape

    # 1) Суперпікселі
    seg = slic(img, n_segments=int(segments), compactness=20, start_label=0)
    n = seg.max() + 1

    # 2) Ознаки: колір Lab (середнє) + LBP-гістограма (текстура)
    lab = rgb2lab(img).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=int(lbp_points), R=int(lbp_radius), method="uniform")

    feats = []
    for i in range(n):
        mask_i = (seg == i)
        if not mask_i.any():
            feats.append(np.zeros(16, dtype=np.float32)); continue
        L = lab[...,0][mask_i].mean()
        A = lab[...,1][mask_i].mean()
        B = lab[...,2][mask_i].mean()
        hist, _ = np.histogram(lbp[mask_i], bins=np.arange(int(lbp_points)+3), range=(0, int(lbp_points)+2), density=True)
        feat = np.concatenate([ [L,A,B], hist.astype(np.float32) ])  # 3 + (P+2)
        feats.append(feat.astype(np.float32))
    feats = np.stack(feats, axis=0)

    # 3) Позитивні сегменти = ті, де rough перекриває значну частину
    pos = []
    neg = []
    for i in range(n):
        m = (seg == i)
        inter = (rough[m] > 0).sum()
        ratio = inter / (m.sum() + 1e-6)
        if ratio > 0.3:
            pos.append(i)
        elif ratio < 0.02:
            neg.append(i)
    if len(pos) == 0:
        # якщо користувач мало намалював — візьмемо топ-кілька сегментів поруч із маскою
        ys, xs = np.where(rough > 0)
        if len(xs) == 0:
            return {"mask_png_base64": mask_to_png_base64(rough)}
        cx, cy = int(xs.mean()), int(ys.mean())
        d = np.full(n, 1e9, dtype=np.float32)
        for i in range(n):
            rp = regionprops((seg==i).astype(np.uint8))
            if rp:
                y,x = rp[0].centroid
                d[i] = (x - cx)**2 + (y - cy)**2
        pos = list(np.argsort(d)[:max(5, n//50)])
        neg = list(np.argsort(d)[-max(50, n//10):])

    # 4) Класифікатор на ознаках (дуже легкий): NearestCentroid
    Xpos = feats[pos]
    Xneg = feats[neg] if len(neg)>0 else feats
    y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))]).astype(int)
    X = np.concatenate([Xpos, Xneg], axis=0)
    clf = NearestCentroid()
    clf.fit(X, y)

    # 5) Оцінюємо всі сегменти: ймовірність -> будуємо маску
    # (NearestCentroid не дає proba, тож використаємо відстань до центрів)
    c0, c1 = clf.centroids_
    d0 = np.linalg.norm(feats - c0, axis=1) + 1e-6
    d1 = np.linalg.norm(feats - c1, axis=1) + 1e-6
    score = d0 / (d0 + d1)  # ближче до "1" = схоже на позитив
    sel = (score > float(thresh)).astype(np.uint8)

    grown = np.zeros((H, W), dtype=np.uint8)
    for i in range(n):
        if sel[i]:
            grown[seg == i] = 255

    # 6) (опційно) Дзеркалимо по вертикалі (для парних елементів ліво/право)
    if int(add_mirror) == 1:
        grown_flip = np.flip(grown, axis=1)
        # залишимо тільки схожі за ознаками сегменти й на дзеркалі:
        # просте правило: якщо фрагмент на фліпі перекриває позитив > 30%, додаємо
        overlap = (grown_flip > 0).astype(np.uint8) * 255
        grown = np.maximum(grown, overlap)

    # 7) Підчистимо маску
    k = np.ones((3,3), np.uint8)
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, k, iterations=1)
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k, iterations=2)

    return {"mask_png_base64": mask_to_png_base64(grown)}
