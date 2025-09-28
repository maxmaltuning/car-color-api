import io, base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

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

@app.post("/refine")
async def refine(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
    w: int = Form(0),
    h: int = Form(0),
):
    img = read_image(image)              # HxWx3
    rough = read_mask(mask, img.shape[:2])  # HxW (0/255)

    # Якщо прийшов ROI — обріжемо до нього (швидше і точніше)
    H, W = img.shape[:2]
    if w > 0 and h > 0:
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        img_roi = img[y0:y1, x0:x1].copy()
        rough_roi = rough[y0:y1, x0:x1].copy()
    else:
        x0 = y0 = 0; x1 = W; y1 = H
        img_roi = img; rough_roi = rough

    # Ініціалізація GrabCut: 3 — ймовірний передній план, 2 — ймовірний фон
    gc_mask = np.where(rough_roi > 0, 3, 2).astype('uint8')

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_roi, gc_mask, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)  # iterations=8

    result = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')

    # Підчистимо краї (morph + легке згладжування)
    kernel = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)
    result = cv2.GaussianBlur(result, (3,3), 0)

    # Повернемо маску у повний розмір кадру
    full = np.zeros((H, W), dtype='uint8')
    full[y0:y1, x0:x1] = result

    pil = Image.fromarray(full)
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"mask_png_base64": b64}
