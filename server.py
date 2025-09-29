# server.py
# Простий API для роботи з масками: refine, grow_similar, sam_point (Replicate SAM2)
# Мова пояснень — максимально проста.

import os
import io
import time
import base64
from typing import Optional, List

import numpy as np
import cv2
from PIL import Image
import requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------- FastAPI ----------
app = FastAPI(title="Car Color API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Допоміжні функції ----------
def pil_to_cv2(im: Image.Image) -> np.ndarray:
    """PIL -> OpenCV BGR"""
    arr = np.array(im.convert("RGB"))
    return arr[:, :, ::-1].copy()

def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL"""
    rgb = img_bgr[:, :, ::-1]
    return Image.fromarray(rgb)

def read_image_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

def read_mask_upload(upload: UploadFile, size=None) -> np.ndarray:
    """Читаємо маску (PNG 0..255). Повертаємо uint8 [0..255]."""
    data = upload.file.read()
    im = Image.open(io.BytesIO(data)).convert("L")
    if size is not None:
        im = im.resize(size, Image.BILINEAR)
    return np.array(im, dtype=np.uint8)

def encode_mask_png_base64(mask_uint8: np.ndarray) -> str:
    """Кодуємо маску у PNG base64 (щоб легко передавати в JSON)."""
    pil = Image.fromarray(mask_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ---------- / ----------
@app.get("/")
def root():
    return {"ok": True, "msg": "Car Color API is live"}

# ---------- /refine ----------
@app.post("/refine")
def refine_mask(
    image: UploadFile = File(..., description="Оригінальне фото"),
    mask: UploadFile = File(..., description="Маска 0..255 (біле — виділено)"),
    close_ksize: int = Form(5, description="Розмір ядра для закриття дірок"),
    open_ksize: int = Form(3, description="Розмір ядра для прибирання шуму"),
    blur_ksize: int = Form(3, description="Згладження краю (блюр)"),
    thresh: int = Form(127, description="Бінаризація (0..255)")
):
    """
    Підчищає маску: морфологічне закриття/відкриття + легкий блюр.
    Повертає base64 PNG маски.
    """
    try:
        im = read_image_upload(image)  # не використовується тут, але залишимо для майбутнього
        w, h = im.size
        m = read_mask_upload(mask, size=(w, h))

        # Бінаризація
        _, mb = cv2.threshold(m, thresh, 255, cv2.THRESH_BINARY)

        # Морфологія
        if close_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            mb = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, k, iterations=1)
        if open_ksize > 1:
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
            mb = cv2.morphologyEx(mb, cv2.MORPH_OPEN, k2, iterations=1)

        # Згладити край
        if blur_ksize > 1 and blur_ksize % 2 == 1:
            mb = cv2.GaussianBlur(mb, (blur_ksize, blur_ksize), 0)

        b64 = encode_mask_png_base64(mb)
        return {"mask_png_base64": b64}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- /grow_similar ----------
@app.post("/grow_similar")
def grow_similar(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    thresh: int = Form(25, description="Дозволена різниця кольору (LAB)"),
    band_margin: float = Form(0.10, description="Ширина службової смуги навколо виділення"),
    keep_k: int = Form(4, description="Скільки найбільш підходящих компонент залишити"),
    add_mirror: int = Form(0, description="1 — пробуємо знайти дзеркальну пару")
):
    """
    Розширює маску на подібні ділянки:
    1) беремо пікселі з існуючої маски
    2) рахуємо їх середній LAB-колір
    3) виділяємо пікселі в межах порогу 'thresh'
    """
    try:
        im_pil = read_image_upload(image)
        w, h = im_pil.size
        m = read_mask_upload(mask, size=(w, h))
        img = pil_to_cv2(im_pil)

        # LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        ys, xs = np.where(m > 127)
        if len(xs) == 0:
            return {"mask_png_base64": encode_mask_png_base64(m)}

        sample = lab[ys, xs]
        mean = sample.mean(axis=0)
        dist = np.linalg.norm(lab.astype(np.float32) - mean.astype(np.float32), axis=2)
        grown = (dist < float(thresh)).astype(np.uint8) * 255

        # Залишимо лише найбільші компоненти
        num_labels, labels = cv2.connectedComponents(grown)
        areas = []
        for lbl in range(1, num_labels):
            areas.append((lbl, int((labels == lbl).sum())))
        areas.sort(key=lambda x: x[1], reverse=True)
        keep = [lbl for lbl, _ in areas[:max(1, keep_k)]]
        out = np.zeros_like(grown)
        for lbl in keep:
            out[labels == lbl] = 255

        # Невелике згладження краю
        out = cv2.GaussianBlur(out, (3, 3), 0)

        # Опціонально: дзеркало (дуже грубо — по горизонталі)
        if add_mirror == 1:
            out = np.maximum(out, cv2.flip(out, 1))

        b64 = encode_mask_png_base64(out)
        return {"mask_png_base64": b64}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- /sam_point (Replicate SAM2) ----------
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
SAM2_VERSION = os.getenv("SAM2_VERSION", "7b2c0e9f0dfeeddc8b20ffec90dd1df6acf9c8d3b0848bb0d9a7d5f2aa4e2d8c")  # lucataco/segment-anything-2

def replicate_upload_file(buf: bytes, mime: str = "image/png") -> str:
    """Завантажуємо файл у Replicate (v1/files) і отримуємо URL (replicate.delivery/...)."""
    fd = {
        # requests потребує (ім'я_поля: (ім'я_файла, байти, mime))
        "file": ("image.png", buf, mime),
    }
    r = requests.post(
        "https://api.replicate.com/v1/files",
        headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"},
        files=fd,
        timeout=120,
    )
    j = r.json()
    if not r.ok:
        raise RuntimeError(f"Replicate file upload failed: {j}")
    return j.get("url")

def replicate_start_prediction(image_url: str, x: Optional[int], y: Optional[int], multimask: int = 0) -> str:
    """Стартуємо предікт SAM2, повертаємо get-URL для опитування."""
    payload = {
        "version": SAM2_VERSION,
        "input": {
            "image": image_url
        }
    }
    # Якщо прийшли координати точки — додамо їх як підказку (prompt)
    if x is not None and y is not None:
        payload["input"]["point_coords"] = [[int(x), int(y)]]
        payload["input"]["point_labels"] = [1]  # 1 — “це всередині”
        payload["input"]["multimask_output"] = bool(multimask)

    r = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    j = r.json()
    if not r.ok:
        raise RuntimeError(f"Prediction start failed: {j}")
    return j["urls"]["get"]  # опитувальний URL

def replicate_poll(get_url: str, timeout_s: int = 180, step_s: float = 2.0) -> dict:
    """Опитуємо статус до готовності або помилки. Повертаємо повний JSON."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = requests.get(
            get_url,
            headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"},
            timeout=60,
        )
        j = r.json()
        st = j.get("status")
        if st in ("succeeded", "failed", "canceled"):
            return j
        time.sleep(step_s)
    raise RuntimeError("Prediction timeout")

@app.post("/sam_point")
def sam_point(
    image: UploadFile = File(..., description="Фото (jpg/png/webp)"),
    x: int = Form(300, description="x"),
    y: int = Form(300, description="y"),
    multimask: int = Form(0, description="0 або 1")
):
    """
    SAM через Replicate: завантажуємо PNG як файл у /v1/files, далі запускаємо предікт.
    Потрібно мати REPLICATE_API_TOKEN у змінних оточення (Render → Environment).
    Повертає JSON з output (URL або масив URLів масок).
    """
    try:
        if not REPLICATE_API_TOKEN:
            return JSONResponse(status_code=500, content={"error": "Missing REPLICATE_API_TOKEN"})

        # 1) Прочитати зображення і привести до PNG (менше проблем з типами)
        raw = image.file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")

        # Обмежимо довгу сторону ~1280 px, щоб не платити зайве
        W, H = pil.size
        max_side = max(W, H)
        if max_side > 1280:
            scale = 1280.0 / max_side
            pil = pil.resize((int(W*scale), int(H*scale)), Image.LANCZOS)

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)

        # 2) Завантажити файл у Replicate
        img_url = replicate_upload_file(buf.getvalue(), mime="image/png")

        # 3) Старт предікта
        get_url = replicate_start_prediction(img_url, x, y, multimask=int(multimask))

        # 4) Дочекатися готовності
        result = replicate_poll(get_url)

        return {
            "status": result.get("status"),
            "output": result.get("output"),
            "web": result.get("urls", {}).get("web"),   # сторінка перегляду на replicate.com
            "get": get_url
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- Запуск локально / на Render ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
