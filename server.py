@app.post("/grow_similar")
async def grow_similar(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    add_mirror: int = Form(1),          # 1 = додавати дзеркальну пару
    segments: int = Form(2500),         # більше суперпікселів -> точніше
    lbp_radius: int = Form(1),
    lbp_points: int = Form(8),
    thresh: float = Form(0.85),         # суворіше
    band_margin: float = Form(0.18),    # смуга навколо зразка (0..0.5)
    keep_k: int = Form(4),              # скільки найближчих компонент лишити
):
    img  = read_image(image)
    rough= read_mask(mask, img.shape[:2])  # 0/255
    H, W = rough.shape

    # --- Seeds: впевнені плюси/мінуси
    k_pos = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    k_neg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    seed_pos = cv2.erode(rough, k_pos)
    ring     = cv2.dilate(rough, k_neg)
    seed_neg = cv2.subtract(ring, rough)

    # bbox вибраного + дозволена зона + дзеркало
    ys, xs = np.where(rough > 0)
    if len(xs) == 0:
        return {"mask_png_base64": mask_to_png_base64(rough)}
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    mx = int(W * band_margin); my = int(H * band_margin)
    x0b = max(0, x0 - mx); x1b = min(W, x1 + mx)
    y0b = max(0, y0 - my); y1b = min(H, y1 + my)
    allowed = np.zeros((H, W), np.uint8)
    allowed[y0b:y1b, x0b:x1b] = 255
    if int(add_mirror) == 1:
        xm0 = max(0, W - x1b); xm1 = min(W, W - x0b)
        allowed[y0b:y1b, xm0:xm1] = 255

    # --- SLIC
    seg = slic(img, n_segments=int(segments), compactness=25, start_label=0)
    n = seg.max() + 1

    # --- Ознаки: Lab + LBP
    from skimage.color import rgb2lab
    from skimage.feature import local_binary_pattern
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

    # --- Позитивні/негативні суперпікселі
    pos_idx = [i for i in range(n) if (seed_pos[seg==i] > 0).sum() > 0.2*areas[i]]
    neg_idx = [i for i in range(n) if (seed_neg[seg==i] > 0).sum() > 0.2*areas[i]]
    if len(pos_idx) == 0:
        cx, cy = xs.mean(), ys.mean()
        d = ((cents[:,0]-cx)**2 + (cents[:,1]-cy)**2)
        pos_idx = list(np.argsort(d)[:max(5, n//50)])
    if len(neg_idx) == 0:
        neg_idx = [i for i in range(n) if (rough[seg==i] > 0).sum() < 0.01*areas[i]]

    # --- Класифікація
    from sklearn.neighbors import NearestCentroid
    Xpos = feats[pos_idx]
    Xneg = feats[neg_idx] if len(neg_idx)>0 else feats
    ylbl = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))]).astype(int)
    Xall = np.concatenate([Xpos, Xneg], axis=0)
    clf  = NearestCentroid().fit(Xall, ylbl)
    c0, c1 = clf.centroids_
    d0 = np.linalg.norm(feats - c0, axis=1) + 1e-6
    d1 = np.linalg.norm(feats - c1, axis=1) + 1e-6
    score = d0 / (d0 + d1)

    # --- Початкова маска з відбором за порогом і allowed-зоною
    sel = (score > float(thresh))
    init = np.zeros((H, W), np.uint8)
    for i in np.where(sel)[0]:
        cx, cy = map(int, cents[i])
        if allowed[min(H-1,max(0,cy)), min(W-1,max(0,cx))] == 0:
            continue
        init[seg == i] = 255

    # --- Постобробка: працюємо по зв’язаних компонентах
    # параметри референсу з твоєї деталі
    seed_area = (rough>0).sum()
    seed_yc   = int(ys.mean())
    # збережемо тільки компоненти з близькою площею та вертикальною позицією
    num, labimg, stats, centroids = cv2.connectedComponentsWithStats(init, connectivity=8)
    kept = np.zeros((H, W), np.uint8)
    candidates = []
    for label in range(1, num):
        x, y, w, h, a = stats[label]
        cx, cy = centroids[label]
        area_ok = (0.5*seed_area <= a <= 2.5*seed_area)
        y_ok    = (abs(cy - seed_yc) <= 0.18*H)  # ±18% по вертикалі
        if area_ok and y_ok:
            # відстань до центру зразка — для сортування
            d = (cx - xs.mean())**2 + (cy - ys.mean())**2
            candidates.append((d, label))
    # беремо K найближчих
    for _, label in sorted(candidates)[:int(keep_k)]:
        kept[labimg == label] = 255

    # опційно додати дзеркальну пару цілком
    if int(add_mirror) == 1:
        kept = np.maximum(kept, np.flip(kept, axis=1))

    k = np.ones((3,3), np.uint8)
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, k, iterations=2)
    kept = cv2.morphologyEx(kept, cv2.MORPH_OPEN,  k, iterations=1)

    return {"mask_png_base64": mask_to_png_base64(kept)}
