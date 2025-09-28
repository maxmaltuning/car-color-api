import io, base64
import numpy as np
from fastapi import FastAPI, UploadFile, File
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
    # приверстаємо до розміру фото, якщо треба
    if m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    # бінарна маска 0/255
    _, m = cv2.threshold(m, 1, 255, cv2.THRESH_BINARY)
    return m

@app.post("/refine")
async def refine(image: UploadFile = File(...), mask: UploadFile = File(...)):
    img = read_image(image)              # HxWx3 RGB
    rough = read_mask(mask, img.shape[:2])  # HxW 0/255

    # Підготовка для GrabCut
    # 0 - фон, 2 - ймовірний фон, 1 - передній план, 3 - ймовірний передній план
    gc_mask = np.where(rough > 0, 3, 2).astype('uint8')

    # Проганяємо GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # Отримуємо чисту маску переднього плану
    result = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')

    # Трошки зачистимо краї
    kernel = np.ones((3,3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Повертаємо PNG як base64
    pil = Image.fromarray(result)
    buf = io.BytesIO(); pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"mask_png_base64": b64}
