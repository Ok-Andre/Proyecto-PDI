#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación experimental de Watershed controlada por marcadores
para segmentación de fuentes luminosas en escenas urbanas nocturnas.

Uso básico:
    python watershed_eval.py imagen1.jpg carpeta_de_imagenes/ -o resultados
    python watershed_eval.py *.jpg --method kapur --thresholds 0.3 0.5 0.7 --version v02

Todos los tamaños de kernel y áreas mínimas son RELATIVOS a las dimensiones de
cada imagen (no hay coordenadas ni máscaras fijas), por lo que el mismo código
procesa fotografías de distinta resolución sin editarlo.

NOTA: los valores son intensidades digitales (niveles de gris), no iluminancia
física (lux). Una región "brillante" no implica una cantidad medida de luz.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RED_BGR = (0, 0, 255)


# ----------------------------------------------------------------------------
# Parámetros experimentales (todos modificables por línea de comandos)
# ----------------------------------------------------------------------------
@dataclass
class Params:
    thresholds: tuple = (0.3, 0.5, 0.7)  # fracción del máximo de la DT
    method: str = "otsu"                 # otsu | kapur
    max_side: int = 1600                 # reescala si el lado mayor lo excede (0 = no)
    blur_frac: float = 0.002             # sigma-kernel del suavizado / min(H,W)
    open_frac: float = 0.004             # kernel de apertura / min(H,W)
    close_frac: float = 0.0              # kernel de cierre / min(H,W) (0 = desactivado)
    bg_dilate_frac: float = 0.008        # kernel de dilatación (fondo seguro) / min(H,W)
    min_area_frac: float = 2e-5          # área mínima de componente / (H*W)
    landscape: str = "image"             # image | distance (relieve que inunda watershed)
    # --- criterios de evaluación (heurísticos, ver evaluar_veredicto) ---
    frag_frac: float = 0.05              # fragmento "diminuto": < frag_frac del área padre
    solidity_thr: float = 0.80           # solidez < thr => candidata a subsegmentación
    min_solidity_area: int = 30          # no evaluar solidez en regiones menores (px)
    lost_area_max: float = 0.05          # máx. fracción de área base perdida
    lost_comp_max: float = 0.20          # máx. fracción de componentes base perdidas
    tiny_frag_max: float = 0.20          # máx. fracción de fragmentos diminutos
    version: str = "v01"


# ----------------------------------------------------------------------------
# Utilidades: verificación de tipos, E/S, kernels
# ----------------------------------------------------------------------------
def check_dtype(arr: np.ndarray, dtype, name: str) -> None:
    if arr.dtype != dtype:
        raise TypeError(f"'{name}' debe ser {np.dtype(dtype)}, pero es {arr.dtype}")


def read_image(path: Path, max_side: int) -> np.ndarray:
    """Lee una imagen BGR uint8 (robusto con rutas con acentos en Windows)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    check_dtype(img, np.uint8, "imagen")
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"No se pudo codificar {path}")
    buf.tofile(str(path))


def odd(n: float) -> int:
    k = max(3, int(round(n)))
    return k + (k % 2 == 0)


def ellipse_kernel(frac: float, shape) -> np.ndarray:
    k = odd(frac * min(shape[:2]))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def collect_inputs(paths) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT and p.exists():
            files.append(p)
        else:
            print(f"[AVISO] Se ignora: {p}", file=sys.stderr)
    return files


# ----------------------------------------------------------------------------
# 1) Segmentación inicial
# ----------------------------------------------------------------------------
def kapur_threshold(gray: np.ndarray) -> int:
    """Umbral de Kapur: maximiza H_fondo(t) + H_objeto(t) (entropías de Shannon)."""
    p = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p /= p.sum()
    P = np.cumsum(p)
    best_t, best_h = 0, -np.inf
    for t in range(255):
        pb, pf = P[t], 1.0 - P[t]
        if pb < 1e-12 or pf < 1e-12:
            continue
        a, b = p[: t + 1] / pb, p[t + 1:] / pf
        a, b = a[a > 0], b[b > 0]
        h = -(a * np.log(a)).sum() - (b * np.log(b)).sum()
        if h > best_h:
            best_h, best_t = h, t
    return best_t


def initial_mask(img_bgr: np.ndarray, P: Params):
    """Gris -> suavizado leve -> umbral global (Otsu o Kapur). Devuelve (gris, máscara, info)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = odd(P.blur_frac * min(gray.shape))
    smooth = cv2.GaussianBlur(gray, (k, k), 0)
    t_otsu, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_kapur = kapur_threshold(smooth)
    T = int(t_otsu) if P.method == "otsu" else int(t_kapur)
    mask = np.where(smooth > T, 255, 0).astype(np.uint8)
    info = {"T_usado": T, "T_otsu": int(t_otsu), "T_kapur": int(t_kapur),
            "media_gris": float(gray.mean()), "mediana_gris": float(np.median(gray)),
            "pct_saturado(>=250)": float((gray >= 250).mean() * 100),
            "pct_oscuro(<50)": float((gray < 50).mean() * 100)}
    check_dtype(gray, np.uint8, "gris")
    check_dtype(mask, np.uint8, "máscara inicial")
    return gray, mask, info


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return (keep[lab] * 255).astype(np.uint8)


def clean_mask(mask: np.ndarray, P: Params) -> np.ndarray:
    """Apertura (quita ruido/puentes finos) [+ cierre opcional] + filtro de área mínima."""
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(P.open_frac, mask.shape))
    if P.close_frac > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ellipse_kernel(P.close_frac, mask.shape))
    out = remove_small(out, max(1, int(P.min_area_frac * mask.size)))
    check_dtype(out, np.uint8, "máscara limpia")
    return out


def label_components(mask: np.ndarray):
    """Referencia previa a Watershed: componentes conexas 8-conectadas."""
    n, lab, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return lab.astype(np.int32), n - 1


# ----------------------------------------------------------------------------
# 2) Watershed controlada por marcadores
# ----------------------------------------------------------------------------
def distance_transform(mask: np.ndarray) -> np.ndarray:
    """Distancia euclidiana exacta de cada píxel de objeto al fondo más cercano."""
    return cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def build_markers(mask: np.ndarray, dt: np.ndarray, ratio: float, bg_kernel: np.ndarray):
    dilated = cv2.dilate(mask, bg_kernel)                 # "posible objeto"
    sure_bg = cv2.bitwise_not(dilated)                    # fondo seguro
    dmax = float(dt.max())
    sure_fg = (dt > ratio * dmax).astype(np.uint8) * 255 if dmax > 0 else np.zeros_like(mask)
    unknown = cv2.subtract(dilated, sure_fg)              # región por decidir
    _, markers = cv2.connectedComponents(sure_fg, connectivity=8)
    markers = markers.astype(np.int32) + 1                # fondo=1, objetos>=2
    markers[unknown == 255] = 0                           # 0 = desconocido
    markers = markers.astype(np.int32)
    for a, n in ((sure_bg, "fondo seguro"), (sure_fg, "primer plano seguro"), (unknown, "desconocida")):
        check_dtype(a, np.uint8, n)
    check_dtype(markers, np.int32, "marcadores")
    return sure_bg, sure_fg, unknown, markers


def run_watershed(img_bgr, markers, dt, landscape: str):
    """cv2.watershed exige imagen uint8 de 3 canales y marcadores int32."""
    if landscape == "distance":
        dmax = dt.max() if dt.max() > 0 else 1.0
        inv = (255 - np.clip(dt / dmax * 255, 0, 255)).astype(np.uint8)
        canvas = cv2.merge([inv, inv, inv])
    else:
        canvas = img_bgr.copy()
    check_dtype(canvas, np.uint8, "imagen para watershed")
    m = markers.copy()
    check_dtype(m, np.int32, "marcadores")
    cv2.watershed(canvas, m)
    return m   # -1 = frontera, 1 = fondo, >=2 = objetos


def finalize_labels(ws: np.ndarray, mask: np.ndarray):
    """
    Etiquetas finales de objetos: 0 = fondo/frontera, 1..N = regiones.
    Se restringe a la máscara limpia para comparar ANTES y DESPUÉS sobre el
    mismo dominio (el anillo de dilatación no cuenta como área segmentada).
    """
    obj = (ws >= 2) & (mask > 0)
    final = np.zeros(ws.shape, np.int32)
    ids = np.unique(ws[obj])
    lut = np.zeros(int(ids.max()) + 1 if ids.size else 1, np.int32)
    lut[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    final[obj] = lut[ws[obj]]
    extra = {"px_frontera": int(((ws == -1) & (mask > 0)).sum()),
             "px_mascara_a_fondo": int(((ws == 1) & (mask > 0)).sum())}
    return final, int(ids.size), extra


# ----------------------------------------------------------------------------
# 3) Métricas
# ----------------------------------------------------------------------------
def solidity_per_region(labels: np.ndarray, min_area: int) -> dict:
    """Solidez = área / área de la envolvente convexa (heurística de no-convexidad)."""
    h, w = labels.shape
    flat = labels.ravel()
    idx = np.flatnonzero(flat)
    if idx.size == 0:
        return {}
    lab = flat[idx]
    order = np.argsort(lab, kind="stable")
    idx, lab = idx[order], lab[order]
    ids, starts, counts = np.unique(lab, return_index=True, return_counts=True)
    out = {}
    for i, s, c in zip(ids, starts, counts):
        if c < min_area:
            continue
        pix = idx[s:s + c]
        ys, xs = pix // w, pix % w
        y0, x0 = ys.min(), xs.min()
        m = np.zeros((ys.max() - y0 + 3, xs.max() - x0 + 3), np.uint8)
        m[ys - y0 + 1, xs - x0 + 1] = 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = cv2.contourArea(cv2.convexHull(np.vstack(cnts)))
        out[int(i)] = min(1.0, c / hull_area) if hull_area > 0 else 1.0
    return out


def evaluate(base_lab, n_base, final, n_final, extra, mask, ratio, P: Params) -> dict:
    H, W = mask.shape
    base_areas = np.bincount(base_lab.ravel(), minlength=n_base + 1)
    fin_areas = np.bincount(final.ravel(), minlength=n_final + 1)
    area_b, area_a = int(base_areas[1:].sum()), int(fin_areas[1:].sum())

    # Relación componente base <-> región watershed (clave int64 = base*(n+1)+final)
    valid = final > 0
    keys = np.unique(base_lab[valid].astype(np.int64) * (n_final + 1) + final[valid])
    pb, pf = keys // (n_final + 1), keys % (n_final + 1)
    per_base = np.bincount(pb, minlength=n_base + 1)        # regiones WS dentro de cada comp. base
    per_final = np.bincount(pf, minlength=n_final + 1)      # comps. base que toca cada región WS
    parent = np.zeros(n_final + 1, np.int64)
    parent[pf] = pb

    lost = np.flatnonzero(per_base[1:] == 0) + 1
    n_split = int((per_base[1:] >= 2).sum())
    n_keep = int((per_base[1:] == 1).sum())
    regs = np.arange(1, n_final + 1)
    in_split = per_base[parent[regs]] >= 2
    tiny = in_split & (fin_areas[regs] < P.frag_frac * base_areas[parent[regs]])
    n_frag = int(in_split.sum())

    sol_b = solidity_per_region(base_lab, P.min_solidity_area)
    sol_a = solidity_per_region(final, P.min_solidity_area)
    low_b = sum(v < P.solidity_thr for v in sol_b.values())
    low_a = sum(v < P.solidity_thr for v in sol_a.values())

    lost_area = int(base_areas[lost].sum())
    return {
        "umbral_DT": ratio,
        "regiones_antes": n_base, "regiones_despues": n_final,
        "variacion_regiones": n_final - n_base,
        "area_antes_px": area_b, "area_despues_px": area_a,
        "pct_area_antes": round(100 * area_b / (H * W), 3),
        "pct_area_despues": round(100 * area_a / (H * W), 3),
        "px_frontera": extra["px_frontera"],
        "px_mascara_asignados_a_fondo": extra["px_mascara_a_fondo"],
        "comp_perdidas": int(lost.size),
        "comp_perdidas_pct": round(100 * lost.size / max(n_base, 1), 2),
        "area_perdida_pct": round(100 * lost_area / max(area_b, 1), 2),
        "comp_divididas": n_split, "comp_sin_cambio": n_keep,
        "fragmentos_en_divididas": n_frag,
        "fragmentos_diminutos": int(tiny.sum()),
        "regiones_ws_que_cruzan_comps": int((per_final[1:] > 1).sum()),
        "baja_solidez_antes": int(low_b), "baja_solidez_despues": int(low_a),
    }


def verdict(r: dict, P: Params) -> str:
    """
    Criterio heurístico y transparente (NO sustituye la inspección visual):
      CONTRAPRODUCENTE : pierde demasiada área/componentes o genera muchos fragmentos diminutos.
      SIN MEJORA       : no separa ninguna componente conexa.
      MEJORA CANDIDATA : separa componentes, sin pérdidas relevantes y reduce regiones no convexas.
      REVISAR          : separa componentes pero sin evidencia numérica clara -> decidir visualmente.
    """
    if r["area_perdida_pct"] > 100 * P.lost_area_max or r["comp_perdidas_pct"] > 100 * P.lost_comp_max:
        return "CONTRAPRODUCENTE (pérdida de regiones)"
    if r["comp_divididas"] == 0:
        return "SIN MEJORA RELEVANTE (no separa)"
    frac_tiny = r["fragmentos_diminutos"] / max(r["fragmentos_en_divididas"], 1)
    if frac_tiny > P.tiny_frag_max:
        return "CONTRAPRODUCENTE (posible sobresegmentación)"
    if r["baja_solidez_despues"] < r["baja_solidez_antes"]:
        return "MEJORA CANDIDATA"
    return "REVISAR VISUALMENTE"


# ----------------------------------------------------------------------------
# 4) Visualización
# ----------------------------------------------------------------------------
def colorize(labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    lut = rng.integers(60, 255, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[np.clip(labels, 0, None)]


def overlay_boundaries(img_bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Dibuja en rojo los límites entre regiones y contra el fondo."""
    lf = labels.astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(lf, k) != cv2.erode(lf, k)
    t = max(1, int(round(min(labels.shape) / 600)))
    if t > 1:
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((t, t), np.uint8)) > 0
    out = img_bgr.copy()
    out[edge] = RED_BGR
    return out


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def show(ax, img, title):
    ax.imshow(rgb(img), cmap="gray" if img.ndim == 2 else None)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def fig_pipeline(path, img, gray, info, m0, m1, sure_bg, dt, base_lab, n_base, ov_base, P):
    fig, ax = plt.subplots(2, 4, figsize=(18, 9))
    show(ax[0, 0], img, "Imagen original")
    ax[0, 1].hist(gray.ravel(), bins=256, range=(0, 255), color="gray")
    ax[0, 1].set_yscale("log")
    ax[0, 1].axvline(info["T_otsu"], color="b", ls="--", label=f"Otsu={info['T_otsu']}")
    ax[0, 1].axvline(info["T_kapur"], color="g", ls=":", label=f"Kapur={info['T_kapur']}")
    ax[0, 1].set_title(f"Histograma (log) — usado: {P.method} T={info['T_usado']}", fontsize=9)
    ax[0, 1].legend(fontsize=8)
    show(ax[0, 2], m0, "Máscara binaria inicial")
    show(ax[0, 3], m1, "Máscara tras morfología")
    show(ax[1, 0], sure_bg, "Fondo seguro")
    im = ax[1, 1].imshow(dt, cmap="magma")
    ax[1, 1].set_title(f"Transformada de distancia (máx={dt.max():.1f}px)", fontsize=9)
    ax[1, 1].axis("off")
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    show(ax[1, 2], colorize(base_lab), f"Componentes conexas (antes): {n_base}")
    show(ax[1, 3], ov_base, "Límites ANTES de Watershed (rojo)")
    fig.suptitle(f"{path.stem} — pipeline previo a Watershed", fontsize=12)
    fig.tight_layout()
    return fig


def fig_watershed(path, img, per_thr, base_n):
    n = len(per_thr)
    fig, ax = plt.subplots(n, 5, figsize=(20, 4 * n), squeeze=False)
    for i, (ratio, d) in enumerate(per_thr.items()):
        show(ax[i, 0], d["sure_fg"], f"Primer plano seguro (t={ratio})")
        show(ax[i, 1], d["unknown"], "Región desconocida")
        show(ax[i, 2], colorize(np.clip(d["markers"] - 1, 0, None)),
             f"Marcadores etiquetados ({d['n_markers']})")
        show(ax[i, 3], colorize(d["final"]), f"Watershed: {d['n_final']} regiones (antes {base_n})")
        show(ax[i, 4], d["overlay"], "Límites DESPUÉS (rojo)")
    fig.suptitle(f"{path.stem} — Watershed por umbral de DT", fontsize=12)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 5) Procesamiento de una imagen
# ----------------------------------------------------------------------------
def process_image(path: Path, P: Params, out_root: Path, display: bool) -> list[dict]:
    out = out_root / path.stem
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{path.stem}_{P.version}"

    img = read_image(path, P.max_side)
    gray, m0, info = initial_mask(img, P)
    m1 = clean_mask(m0, P)
    base_lab, n_base = label_components(m1)
    ov_base = overlay_boundaries(img, base_lab)
    dt = distance_transform(m1)
    bg_k = ellipse_kernel(P.bg_dilate_frac, m1.shape)

    per_thr, rows = {}, []
    for ratio in P.thresholds:
        sure_bg, sure_fg, unknown, markers = build_markers(m1, dt, ratio, bg_k)
        ws = run_watershed(img, markers, dt, P.landscape)
        final, n_final, extra = finalize_labels(ws, m1)
        ov = overlay_boundaries(img, final)
        row = evaluate(base_lab, n_base, final, n_final, extra, m1, ratio, P)
        row.update(imagen=path.name, version=P.version, metodo_umbral=P.method,
                   T_gris=info["T_usado"], alto=img.shape[0], ancho=img.shape[1],
                   veredicto=verdict(row, P))
        rows.append(row)
        per_thr[ratio] = dict(sure_fg=sure_fg, unknown=unknown, markers=markers,
                              n_markers=int(markers.max()) - 1, final=final,
                              n_final=n_final, overlay=ov)
        t = f"{ratio:.2f}"
        write_image(out / f"{tag}_watershed_t{t}_etiquetas.png", colorize(final))
        write_image(out / f"{tag}_watershed_t{t}_limites.png", ov)

    # Salidas comunes (independientes del umbral)
    write_image(out / f"{tag}_mascara_inicial.png", m0)
    write_image(out / f"{tag}_mascara_morfologia.png", m1)
    write_image(out / f"{tag}_baseline_cc_limites.png", ov_base)
    dt8 = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    write_image(out / f"{tag}_distancia.png", dt8)
    sure_bg = cv2.bitwise_not(cv2.dilate(m1, bg_k))
    f1 = fig_pipeline(path, img, gray, info, m0, m1, sure_bg, dt, base_lab, n_base, ov_base, P)
    f1.savefig(out / f"{tag}_fig_pipeline.png", dpi=110)
    f2 = fig_watershed(path, img, per_thr, n_base)
    f2.savefig(out / f"{tag}_fig_watershed.png", dpi=110)
    (out / f"{tag}_histograma_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if display:
        plt.show()
    plt.close("all")
    return rows


# ----------------------------------------------------------------------------
# 6) Tabla y resumen
# ----------------------------------------------------------------------------
COLS = ["imagen", "umbral_DT", "regiones_antes", "regiones_despues", "variacion_regiones",
        "area_antes_px", "area_despues_px", "pct_area_antes", "pct_area_despues",
        "comp_perdidas", "area_perdida_pct", "comp_divididas", "fragmentos_diminutos",
        "baja_solidez_antes", "baja_solidez_despues", "veredicto"]


def print_table(rows):
    w = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in COLS}
    print(" | ".join(c.ljust(w[c]) for c in COLS))
    print("-+-".join("-" * w[c] for c in COLS))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(w[c]) for c in COLS))


def save_outputs(rows, P: Params, out_root: Path):
    csv_path = out_root / f"metricas_{P.version}_{P.method}.csv"
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    (out_root / f"parametros_{P.version}.json").write_text(
        json.dumps(asdict(P), indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [f"# Resumen {P.version} (umbral {P.method})", "",
             "| Umbral DT | Imágenes | CONTRAPRODUCENTE | MEJORA CANDIDATA | REVISAR | SIN MEJORA |",
             "|---|---|---|---|---|---|"]
    for t in P.thresholds:
        sub = [r for r in rows if r["umbral_DT"] == t]
        c = lambda key: sum(r["veredicto"].startswith(key) for r in sub)
        lines.append(f"| {t} | {len(sub)} | {c('CONTRA')} | {c('MEJORA')} | {c('REVISAR')} | {c('SIN MEJORA')} |")
    lines += ["", "Los veredictos son heurísticos; confirmar con las figuras `*_fig_watershed.png`.",
              "Las intensidades son niveles de gris, no lux."]
    (out_root / f"resumen_{P.version}_{P.method}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCSV: {csv_path}")


# ----------------------------------------------------------------------------
def parse_args():
    d = Params()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Imágenes y/o carpetas")
    ap.add_argument("-o", "--output", default="resultados")
    ap.add_argument("--show", action="store_true", help="Mostrar figuras en pantalla")
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(d.thresholds))
    ap.add_argument("--method", choices=["otsu", "kapur"], default=d.method)
    ap.add_argument("--landscape", choices=["image", "distance"], default=d.landscape)
    for name in ("max_side", "frag_frac", "solidity_thr", "min_solidity_area", "lost_area_max",
                 "lost_comp_max", "tiny_frag_max", "blur_frac", "open_frac", "close_frac",
                 "bg_dilate_frac", "min_area_frac"):
        ap.add_argument("--" + name.replace("_", "-"), type=type(getattr(d, name)), default=getattr(d, name))
    ap.add_argument("--version", default=d.version)
    return ap.parse_args()


def main():
    a = parse_args()
    if not a.show:
        plt.switch_backend("Agg")
    P = Params(**{k: v for k, v in vars(a).items() if k in Params.__dataclass_fields__})
    P.thresholds = tuple(a.thresholds)
    files = collect_inputs(a.inputs)
    if not files:
        sys.exit("No se encontraron imágenes válidas.")
    out_root = Path(a.output)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        print(f"Procesando {f.name} ...")
        rows += process_image(f, P, out_root, a.show)
    print_table(rows)
    save_outputs(rows, P, out_root)


if __name__ == "__main__":
    main()
