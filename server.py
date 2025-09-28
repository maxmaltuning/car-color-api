# ---------- НОВЕ: grow_similar з жорсткішим відбором і просторовим обмеженням ----------
@app.post("/grow_similar")
async def grow_similar(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    add_mirror: int = Form(1),          # 1 = додати дзеркальну зону (для парних деталей)
    segments: int = Form(2000),         # більше суперпікселів -> точніше
    lbp_radius: int = Form(1),
    lbp_points: int = Form(8),
    thresh: float = Form(0.8),          # вище поріг -> менше «сміття»
    band_margin: float = Form(0.22),    # ширина «дозволеної» смуги навколо зразка (0..0.5)
):
    img  = read_image(image)
    rough= read_mask(mask, img.shape[:2])  # 0/255
    H, W = rough.shape

    # --- 0) Підготовка позитивів/негативів із маски користувача ---
    k_pos = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    k_neg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    seed_pos = cv2.erode(rough, k_pos)                  # впевнені всередині
    ring     = cv2.dilate(rough, k_neg)
    seed_neg = cv2.subtract(ring, rough)                # кільце зовні

    # bbox обраної деталі + «дозволена» смуга навколо + дзеркало
    ys, xs = np.where(rough > 0)
    if len(xs) == 0:
        return {"mask_png_base64": mask_to_png_base64(rough)}
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # розширимо bbox
    mx = int(W * band_margin); my = int(H * band_margin)
    x0b = max(0, x0 - mx); x1b = min(W, x1 + mx)
    y0b = max(0, y0 - my); y1b = min(H, y1 + my)

    allowed = np.zeros((H, W), np.uint8)
    allowed[y0b:y1b, x0b:x1b] = 255
    if int(add_mirror) == 1:
        # просте дзеркало по центру кадру
        xm0 = max(0, W - x1b); xm1 = min(W, W - x0b)
        allowed[y0b:y1b, xm0:xm1] = 255

    # --- 1) Суперпікселі ---
    seg = slic(img, n_segments=int(segments), compactness=25, start_label=0)
    n = seg.max() + 1

    # --- 2) Ознаки: Lab + LBP (текстура) ---
    lab  = rgb2lab(img).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, P=int(lbp_points), R=int(lbp_radius), method="uniform")

    feats = np.zeros((n, 3 + int(lbp_points)+2), np.float32)
    cents = np.zeros((n, 2), np.float32)   # центроїди сегментів
    areas = np.zeros(n, np.int32)
    for i in range(n):
        m = (seg == i)
        if not m.any(): 
            continue
        areas[i] = m.sum()
        ys_i, xs_i = np.where(m)
        cents[i] = [xs_i.mean(), ys_i.mean()]
        L = lab[...,0][m].mean(); A = lab[...,1][m].mean(); B = lab[...,2][m].mean()
        hist, _ = np.histogram(lbp[m], bins=np.arange(int(lbp_points)+3), range=(0, int(lbp_points)+2), density=True)
        feats[i] = np.concatenate([[L,A,B], hist.astype(np.float32)])

    # --- 3) Позитивні/негативні сегменти ---
    pos_idx = [i for i in range(n) if (seed_pos[seg==i] > 0).sum() > 0.2*areas[i]]
    neg_idx = [i for i in range(n) if (seed_neg[seg==i] > 0).sum() > 0.2*areas[i]]
    # якщо користувач намалював мало — підхопимо кілька найближчих сегментів
    if len(pos_idx) == 0:
        cx, cy = xs.mean(), ys.mean()
        d = ((cents[:,0]-cx)**2 + (cents[:,1]-cy)**2)
        pos_idx = list(np.argsort(d)[:max(5, n//50)])
    if len(neg_idx) == 0:
        neg_idx = [i for i in range(n) if (rough[seg==i] > 0).sum() < 0.01*areas[i]]

    # --- 4) Навчання простого класифікатора ---
    Xpos = feats[pos_idx]
    Xneg = feats[neg_idx] if len(neg_idx)>0 else feats
    y    = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))]).astype(int)
    X    = np.concatenate([Xpos, Xneg], axis=0)
    clf  = NearestCentroid()
    clf.fit(X, y)
    c0, c1 = clf.centroids_
    d0 = np.linalg.norm(feats - c0, axis=1) + 1e-6
    d1 = np.linalg.norm(feats - c1, axis=1) + 1e-6
    score = d0 / (d0 + d1)

    # --- 5) Відбір + просторове обмеження (всередині allowed) + фільтр за розміром ---
    sel = (score > float(thresh))
    # лишаємо тільки сегменти, центр яких у дозволеній зоні
    idxs = np.where(sel)[0]
    for i in idxs:
        cx, cy = map(int, cents[i])
        if allowed[min(H-1, max(0, cy)), min(W-1, max(0, cx))] == 0:
            sel[i] = False

    # відфільтруємо дуже малі/великі відносно середнього позитиву
    if len(pos_idx) > 0:
        a_mean = areas[pos_idx].mean()
        for i in np.where(sel)[0]:
            if areas[i] < 0.3*a_mean or areas[i] > 3.0*a_mean:
                sel[i] = False

    grown = np.zeros((H, W), np.uint8)
    for i in np.where(sel)[0]:
        grown[seg == i] = 255

    # --- 6) Підчистити ---
    k = np.ones((3,3), np.uint8)
    grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, k, iterations=1)
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, k, iterations=2)

    return {"mask_png_base64": mask_to_png_base64(grown)}
