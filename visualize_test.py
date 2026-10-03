import os
import sys
import json
import yaml
import random
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import architecture if needed for checkpoint mode
try:
    from SAM2UNet import SAM2UNet
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualisasi Perbandingan Hasil Test Segmentasi Longsor: [RGB, DTM, Ground Truth, RGB-DTM 0%, RGB-DTM 100%]"
    )
    # Dataset and input paths
    parser.add_argument(
        "--test_root",
        type=str,
        required=True,
        help="Path ke folder dataset test (berisi folder IMAGE, DTM_NORM/DTM, dan LABEL)",
    )
    # Mode A: Precomputed prediction results (from test.py)
    parser.add_argument(
        "--pred_0_dir",
        type=str,
        default=None,
        help="Path ke folder hasil prediksi RGB-DTM 0%% (bisa root folder test_results atau folder binary_masks)",
    )
    parser.add_argument(
        "--pred_100_dir",
        type=str,
        default=None,
        help="Path ke folder hasil prediksi RGB-DTM 100%% (bisa root folder test_results atau folder binary_masks)",
    )
    # Mode B: Direct Checkpoint inference (end-to-end)
    parser.add_argument(
        "--checkpoint_0",
        type=str,
        default=None,
        help="(Opsional) Path ke file .pth checkpoint model RGB-DTM 0%%",
    )
    parser.add_argument(
        "--checkpoint_100",
        type=str,
        default=None,
        help="(Opsional) Path ke file .pth checkpoint model RGB-DTM 100%%",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path ke file konfigurasi YAML (digunakan untuk inferensi checkpoint)",
    )

    # Visualization and selection parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Direktori untuk menyimpan gambar hasil visualisasi",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Jumlah sampel bertumpuk yang ditampilkan (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold binarisasi probabilitas prediksi (default: 0.5)",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=10,
        help="Jumlah piksel minimum landslide agar sampel dianggap positif (default: 10)",
    )
    parser.add_argument(
        "--blacklist_path",
        type=str,
        default=None,
        help="Path ke file teks blacklist untuk melewati data rusak (format 1 ID sampel per baris, contoh: datasets/landslide/black_list.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed untuk reproducibility pemilihan sampel (opsional)",
    )
    parser.add_argument(
        "--sample_names",
        type=str,
        default=None,
        help="Daftar nama sampel spesifik dipisahkan koma (opsional, jika ingin menentukan sampel manual)",
    )
    parser.add_argument(
        "--dtm_colormap",
        type=str,
        default="elevation",
        help="Colormap untuk raster DTM: 'elevation' (biru-hijau-kuning-coklat-putih), 'terrain', 'viridis', dll",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolusi DPI gambar output (default: 300)",
    )
    parser.add_argument(
        "--save_individual",
        nargs="?",
        const=True,
        type=lambda x: (str(x).lower() in ("true", "1", "yes")),
        default=True,
        help="Simpan juga masing-masing dari 5 sampel ke file terpisah (default: True)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_blacklist(blacklist_path: Optional[str]) -> set:
    """
    Memuat daftar ID sampel yang di-blacklist (data rusak / anomali sensor) dari file teks.
    Mendukung format file dengan 1 ID per baris (seperti datasets/landslide/black_list.txt).
    Mengabaikan baris kosong, komentar (#), dan ekstensi file jika ada.
    """
    if not blacklist_path or str(blacklist_path).lower() in ("none", "null", ""):
        return set()
    if not os.path.exists(blacklist_path):
        print(f"[!] Peringatan: File blacklist tidak ditemukan di '{blacklist_path}'")
        return set()
    blacklist = set()
    with open(blacklist_path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            stem, _ = os.path.splitext(clean)
            blacklist.add(stem)
    return blacklist


def find_dir(root: str, candidates: List[str]) -> str:
    """Mencari direktori yang ada berdasarkan daftar kandidat nama folder."""
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    return ""


def find_sample_file(folder: str, stem: str, extensions: Tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg")) -> Optional[str]:
    """Mencari file sampel berdasarkan stem dan ekstensi yang didukung."""
    for ext in extensions:
        candidate = os.path.join(folder, f"{stem}{ext}")
        if os.path.isfile(candidate):
            return candidate
        # Coba juga ekstensi huruf kapital
        candidate_upper = os.path.join(folder, f"{stem}{ext.upper()}")
        if os.path.isfile(candidate_upper):
            return candidate_upper
    return None


def resolve_prediction_file(pred_dir: str, stem: str) -> str:
    """
    Mencari file prediksi untuk sampel tertentu di berbagai kemungkinan struktur folder:
    1. pred_dir/binary_masks/{stem}.png (.tif)
    2. pred_dir/probability_maps/{stem}.png (.tif)
    3. pred_dir/{stem}.png (.tif)
    4. pred_dir/test_results/binary_masks/{stem}.png
    """
    candidates = [
        os.path.join(pred_dir, "binary_masks"),
        os.path.join(pred_dir, "probability_maps"),
        pred_dir,
        os.path.join(pred_dir, "test_results", "binary_masks"),
        os.path.join(pred_dir, "test_results", "probability_maps"),
    ]

    for c in candidates:
        if os.path.isdir(c):
            found = find_sample_file(c, stem)
            if found is not None:
                return found

    raise FileNotFoundError(
        f"File prediksi untuk sampel '{stem}' tidak ditemukan di '{pred_dir}' "
        f"(sudah dicek di binary_masks/, probability_maps/, dan root folder)."
    )


def read_dtm_raster(file_path: str) -> Tuple[np.ndarray, Optional[float], str]:
    """
    Membaca raster DTM (.tif) dengan multi-tier fallback:
    1. rasterio (pembaca GeoTIFF standar GIS)
    2. cv2 (OpenCV IMREAD_UNCHANGED - membaca Float32 TIFF secara akurat)
    3. tifffile
    4. imageio
    5. PIL.Image
    Mengembalikan (arr, nodata_value, engine_name).
    """
    arr = None
    nodata = None
    engine = ""

    # 1. Coba rasterio
    try:
        import rasterio
        with rasterio.open(file_path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
            engine = f"rasterio (nodata={nodata})"
    except Exception:
        pass

    # 2. Coba cv2 IMREAD_UNCHANGED
    if arr is None:
        try:
            cv_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if cv_img is not None:
                if cv_img.ndim == 3:
                    cv_img = cv_img[:, :, 0]
                arr = cv_img.astype(np.float32)
                engine = "OpenCV (cv2)"
        except Exception:
            pass

    # 3. Coba tifffile
    if arr is None:
        try:
            import tifffile
            t_arr = tifffile.imread(file_path).astype(np.float32)
            if t_arr.ndim == 3:
                t_arr = t_arr[0] if t_arr.shape[0] < t_arr.shape[2] else t_arr[:, :, 0]
            arr = t_arr
            engine = "tifffile"
        except Exception:
            pass

    # 4. Coba imageio
    if arr is None:
        try:
            import imageio.v2 as imageio
            i_img = imageio.imread(file_path)
            if i_img.ndim == 3:
                i_img = i_img[:, :, 0]
            arr = np.array(i_img, dtype=np.float32)
            engine = "imageio"
        except Exception:
            pass

    # 5. Coba PIL
    if arr is None:
        try:
            im = Image.open(file_path)
            p_arr = np.array(im, dtype=np.float32)
            if p_arr.ndim == 3:
                p_arr = p_arr[:, :, 0]
            arr = p_arr
            engine = "PIL"
        except Exception as e:
            raise RuntimeError(f"Gagal membaca raster DTM '{file_path}': {e}")

    return arr, nodata, engine


def format_elevation_tick(val: float, val_range: float) -> str:
    """Format label elevasi secara adaptif sesuai rentang data."""
    if val_range >= 100:
        return f"{val:.0f}m"
    elif val_range >= 10:
        return f"{val:.1f}m"
    elif val_range >= 1:
        return f"{val:.2f}"
    else:
        return f"{val:.3f}"


def inspect_and_normalize_dtm(
    dtm_raw: np.ndarray,
    nodata: Optional[float],
    sample_name: str,
    engine: str,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    Memeriksa rentang data DTM, membersihkan NoData / anomali ekstrem,
    mencetak ringkasan rentang data ke terminal, dan menghasilkan
    array display ternormalisasi [0.0, 1.0] yang dipetakan ke colormap.
    Mengembalikan (dtm_display, disp_min, disp_max, valid_mask).
    """
    H, W = dtm_raw.shape[:2]
    finite_mask = np.isfinite(dtm_raw)

    valid_mask = finite_mask.copy()
    if nodata is not None and not np.isnan(nodata):
        valid_mask = valid_mask & (~np.isclose(dtm_raw, nodata, atol=1e-2))

    # Masking nilai sentinel NoData GIS yang umum (-9999, 50000, 65535, 1e38, dll)
    nodata_candidates = [-9999.0, -99999.0, -32768.0, -32767.0, 9999.0, 32767.0, 50000.0, 65535.0, -3.4028235e+38, 1e38]
    for sentinel in nodata_candidates:
        if np.any(np.isclose(dtm_raw[finite_mask], sentinel, atol=1e-1)):
            valid_mask = valid_mask & (~np.isclose(dtm_raw, sentinel, atol=1e-1))

    # Jika ada nilai ekstrem di atas 10.000m (sentinel NoData seperti di dataset.py)
    if valid_mask.any():
        cur_max = float(np.max(dtm_raw[valid_mask]))
        if cur_max > 10000.0:
            sensible_mask = (dtm_raw > -1000.0) & (dtm_raw < 10000.0)
            if sensible_mask.any():
                valid_mask = valid_mask & sensible_mask

    if valid_mask.any():
        valid_vals = dtm_raw[valid_mask]
        raw_min = float(np.min(valid_vals))
        raw_max = float(np.max(valid_vals))
        mean_val = float(np.mean(valid_vals))
        std_val = float(np.std(valid_vals))
        p1 = float(np.percentile(valid_vals, 1))
        p2 = float(np.percentile(valid_vals, 2))
        p98 = float(np.percentile(valid_vals, 98))
        p99 = float(np.percentile(valid_vals, 99))
    else:
        raw_min, raw_max, mean_val, std_val = 0.0, 0.0, 0.0, 0.0
        p1, p2, p98, p99 = 0.0, 0.0, 0.0, 0.0

    raw_range = raw_max - raw_min
    p_range = p99 - p1

    # Tampilkan inspeksi data ke terminal secara rinci
    print(f"  [DTM Data Range: {sample_name}] Reader: {engine}")
    print(f"    - Resolusi : {W}x{H} piksel | Valid: {np.count_nonzero(valid_mask):,}/{valid_mask.size:,} ({np.count_nonzero(valid_mask)/valid_mask.size*100:.1f}%)")
    print(f"    - Rentang  : Min = {raw_min:.4f} | Max = {raw_max:.4f} | Range = {raw_range:.4f}")
    print(f"    - Statistik: Mean = {mean_val:.4f} | Std = {std_val:.4f}")
    print(f"    - Persentil: P1 = {p1:.4f} | P2 = {p2:.4f} | P98 = {p98:.4f} | P99 = {p99:.4f}")

    # Deteksi outlier ekstrem (misal nilai batas frame peta)
    if p_range > 0 and (raw_range > 5 * p_range):
        disp_min = p1
        disp_max = p99
        print(f"    [!] Outlier batas terdeteksi! Menggunakan rentang kontras P1-P99 ({disp_min:.4f} s/d {disp_max:.4f})")
    else:
        disp_min = raw_min
        disp_max = raw_max

    # Normalisasi untuk pewarnaan
    if disp_max > disp_min:
        norm_arr = np.clip((dtm_raw - disp_min) / (disp_max - disp_min), 0.0, 1.0)
    else:
        print(f"    [!] PERINGATAN: Rentang DTM bernilai 0 (semua bernilai {disp_min:.4f})!")
        norm_arr = np.zeros_like(dtm_raw)

    if not valid_mask.all():
        dtm_display = np.ma.masked_array(norm_arr, mask=~valid_mask)
    else:
        dtm_display = norm_arr

    return dtm_display, disp_min, disp_max, valid_mask


def get_custom_dtm_colormap():
    """
    Membuat colormap khusus ketinggian DTM bergradasi:
    Biru -> Hijau -> Kuning -> Coklat -> Putih
    """
    colors = [
        (0.00, "#08306b"),  # Biru tua (lembah terendah)
        (0.15, "#2171b5"),  # Biru muda
        (0.32, "#31a354"),  # Hijau (dataran rendah)
        (0.52, "#fee391"),  # Kuning (perbukitan)
        (0.70, "#fe9929"),  # Kuning kecoklatan
        (0.85, "#8c510a"),  # Coklat (pegunungan terjal)
        (1.00, "#ffffff"),  # Putih (puncak tertinggi)
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dtm_elevation",
        [(pos, col) for pos, col in colors],
        N=256
    )
    cmap.set_bad(color="#d9d9d9")  # NoData pixel ditampilkan sebagai abu-abu netral
    return cmap


def get_dtm_colormap(cmap_name: str = "elevation"):
    """Mengembalikan colormap untuk DTM."""
    if not cmap_name or str(cmap_name).lower() in ("elevation", "dtm", "terrain_custom", "default"):
        return get_custom_dtm_colormap()
    try:
        return plt.get_cmap(cmap_name)
    except Exception:
        return get_custom_dtm_colormap()


def apply_edge_scale(ax, H: int, W: int, step: Optional[int] = None):
    """
    Menambahkan scale koordinat piksel pada tepian tiap gambar
    (sumbu x di bagian bawah dan sumbu y di bagian kiri).
    """
    if step is None:
        max_dim = max(H, W)
        if max_dim <= 150:
            step = 25
        elif max_dim <= 300:
            step = 50
        elif max_dim <= 600:
            step = 100
        else:
            step = 200

    xticks = np.arange(0, W, step)
    if len(xticks) == 0 or (W - xticks[-1]) > step // 2:
        xticks = np.append(xticks, W)
    yticks = np.arange(0, H, step)
    if len(yticks) == 0 or (H - yticks[-1]) > step // 2:
        yticks = np.append(yticks, H)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=8,
        direction="out",
        length=3.5,
        width=0.8,
        colors="#333333",
        top=False,
        right=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.8)


def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Dict[str, float]:
    """Menghitung metrik segmentasi per sampel: IoU, Dice/F1, Precision, Recall."""
    pred = (pred_bin > 0).astype(bool)
    gt = (gt_bin > 0).astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    tp = intersection
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~gt, gt).sum() if False else np.logical_and(~pred, gt).sum()

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (pred.sum() + gt.sum() + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def load_prediction_mask(file_path: str, threshold: float = 0.5) -> np.ndarray:
    """Memuat masker prediksi dari file gambar (bisa biner atau peta probabilitas)."""
    img = Image.open(file_path)
    arr = np.array(img, dtype=np.float32)

    # Jika gambar RGB/RGBA, ambil channel pertama
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    if arr.max() > 1.0:
        # Jika nilai dalam rentang 0-255
        prob = arr / 255.0
    else:
        prob = arr

    binary_mask = (prob >= threshold).astype(np.float32)
    return binary_mask


def create_overlay_image(rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    """Menghasilkan citra overlay RGB dengan masker berwarna semi-transparan."""
    overlay = rgb.copy().astype(np.float32)
    pos_mask = (mask > 0)
    for c in range(3):
        overlay[pos_mask, c] = overlay[pos_mask, c] * (1.0 - alpha) + color[c] * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def get_model(checkpoint_path: str, device: torch.device, config: Dict[str, Any]):
    """Memuat model SAM2UNet dari checkpoint untuk inferensi on-the-fly."""
    if not SAM2_AVAILABLE:
        raise ImportError("Modul SAM2UNet tidak dapat diimpor. Pastikan file SAM2UNet.py ada.")

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    use_ssf = any(k.startswith("proj1.fuse") or k.startswith("proj1.conv1") for k in state_dict.keys())
    use_kan = any("kan." in k for k in state_dict.keys())
    topo_backbone = config.get("model", {}).get("topo_backbone", "convnext_tiny")

    model = SAM2UNet(
        checkpoint_path=None,
        topo_in_chans=1,
        topo_backbone=topo_backbone,
        pretrained_topo=False,
        use_kan=use_kan,
        use_ssf=use_ssf,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def run_inference_on_sample(
    model: torch.nn.Module,
    rgb_arr: np.ndarray,
    dtm_norm: np.ndarray,
    is_zero_dtm: bool,
    device: torch.device,
    target_size: int = 512,
) -> np.ndarray:
    """Melakukan forward pass inferensi untuk 1 sampel."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Resize dan normalisasi RGB
    rgb_resized = cv2.resize(rgb_arr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb_norm = (rgb_resized.astype(np.float32) / 255.0 - mean) / std
    rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).unsqueeze(0).to(device, dtype=torch.float32)

    # Topo raster
    if is_zero_dtm:
        topo_tensor = torch.zeros((1, 1, target_size, target_size), device=device, dtype=torch.float32)
    else:
        dtm_resized = cv2.resize(dtm_norm, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        topo_tensor = torch.from_numpy(dtm_resized).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

    input_tensor = torch.cat([rgb_tensor, topo_tensor], dim=1)

    with torch.no_grad():
        preds, *rest = model(input_tensor)
        prob = torch.sigmoid(preds)[0, 0].cpu().numpy()

    # Resize kembali ke ukuran asli jika berbeda
    if prob.shape != rgb_arr.shape[:2]:
        prob = cv2.resize(prob, (rgb_arr.shape[1], rgb_arr.shape[0]), interpolation=cv2.INTER_LINEAR)

    return prob


def main():
    args = parse_args()
    print("=" * 80)
    print("      VISUALISASI HASIL TESTING: RGB, DTM, GT, RGB-DTM 0%, RGB-DTM 100%")
    print("=" * 80)

    # 1. Validasi direktori dataset test
    test_root = args.test_root
    if not os.path.exists(test_root):
        print(f"ERROR: test_root tidak ditemukan: {test_root}")
        sys.exit(1)

    image_dir = find_dir(test_root, ["IMAGE", "image", "images", "test/IMAGE", "test/image"])
    dtm_dir = find_dir(test_root, ["DTM_NORM", "dtm_norm", "DTM", "dtm", "test/DTM_NORM", "test/DTM", "test/dtm"])
    label_dir = find_dir(test_root, ["LABEL", "label", "labels", "masks", "test/LABEL", "test/label"])

    if not image_dir:
        print(f"ERROR: Subfolder IMAGE tidak ditemukan di {test_root}")
        sys.exit(1)
    if not dtm_dir:
        print(f"ERROR: Subfolder DTM_NORM/DTM tidak ditemukan di {test_root}")
        sys.exit(1)
    if not label_dir:
        print(f"ERROR: Subfolder LABEL tidak ditemukan di {test_root}")
        sys.exit(1)

    print(f"[*] Lokasi Dataset:")
    print(f"    - IMAGE : {image_dir}")
    print(f"    - DTM   : {dtm_dir}")
    print(f"    - LABEL : {label_dir}")

    # 2. Tentukan Mode: Dari File Hasil Test vs Inferensi Checkpoint Langsung
    use_checkpoint_mode = bool(args.checkpoint_0 and args.checkpoint_100)
    if use_checkpoint_mode:
        print(f"[*] Mode Inferensi Langsung dari Checkpoint:")
        print(f"    - Checkpoint 0%   : {args.checkpoint_0}")
        print(f"    - Checkpoint 100% : {args.checkpoint_100}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    - Device Komputasi: {device}")
        config = load_config(args.config)
        model_0 = get_model(args.checkpoint_0, device, config)
        model_100 = get_model(args.checkpoint_100, device, config)
    else:
        if not args.pred_0_dir or not args.pred_100_dir:
            print("ERROR: Harap tentukan salah satu dari:")
            print("  1. Folder hasil test (--pred_0_dir dan --pred_100_dir), ATAU")
            print("  2. Checkpoint model (--checkpoint_0 dan --checkpoint_100)")
            sys.exit(1)
        print(f"[*] Mode Membaca Hasil Test (Cepat, Tanpa Beban GPU):")
        print(f"    - Pred 0%   : {args.pred_0_dir}")
        print(f"    - Pred 100% : {args.pred_100_dir}")

    # 3. Muat file blacklist jika disediakan atau ada di lokasi standar
    blacklist_path = args.blacklist_path
    if not blacklist_path:
        # Coba auto-detect dari folder test_root atau default config
        for candidate in [
            os.path.join(test_root, "black_list.txt"),
            os.path.join(test_root, "blacklist.txt"),
            "datasets/landslide/black_list.txt",
        ]:
            if os.path.isfile(candidate):
                blacklist_path = candidate
                break

    blacklist = load_blacklist(blacklist_path)
    if blacklist:
        print(f"[*] Blacklist aktif: {len(blacklist)} sampel data rusak dari '{blacklist_path}' akan dilewati (skip).")

    # 4. Temukan semua sampel dan seleksi hanya yang POSITIF punya landslide & TIDAK di-blacklist
    all_label_files = sorted([
        f for f in os.listdir(label_dir)
        if f.lower().endswith((".png", ".tif", ".tiff", ".jpg"))
    ])

    print(f"\n[*] Memindai {len(all_label_files)} sampel untuk mendeteksi sampel positif landslide...")
    positive_samples: List[Tuple[str, int, float]] = []
    skipped_blacklist = 0

    for f in all_label_files:
        stem, _ = os.path.splitext(f)
        if stem in blacklist:
            skipped_blacklist += 1
            continue
        lbl_file = os.path.join(label_dir, f)
        try:
            lbl_arr = np.array(Image.open(lbl_file))
            pos_pixels = int(np.count_nonzero(lbl_arr > 0))
            if pos_pixels >= args.min_pixels:
                pct = (pos_pixels / float(lbl_arr.size)) * 100.0
                positive_samples.append((stem, pos_pixels, pct))
        except Exception as e:
            continue

    msg_skip = f" ({skipped_blacklist} sampel data rusak dilewati via blacklist)." if skipped_blacklist > 0 else "."
    print(f"[*] Ditemukan {len(positive_samples)} sampel POSITIF landslide (>= {args.min_pixels} piksel){msg_skip}")

    if len(positive_samples) == 0:
        print("ERROR: Tidak ditemukan sampel dengan label positif landslide!")
        sys.exit(1)

    # 5. Pilih N sampel positif secara acak (atau gunakan daftar spesifik)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"[*] Menggunakan random seed: {args.seed}")
    else:
        print(f"[*] Pemilihan sampel acak murni (tanpa seed tetap).")

    if args.sample_names:
        specified = [s.strip() for s in args.sample_names.split(",") if s.strip()]
        for s in specified:
            if s in blacklist:
                print(f"[!] PERINGATAN: Sampel '{s}' yang Anda minta tercatat di blacklist data rusak!")
        selected_stems = [s for s in specified if any(s == p[0] for p in positive_samples)]
        if len(selected_stems) < len(specified):
            print(f"[!] Catatan: Beberapa nama sampel yang diminta tidak ditemukan di daftar positif.")
    else:
        num_to_pick = min(args.num_samples, len(positive_samples))
        picked = random.sample(positive_samples, num_to_pick)
        selected_stems = [p[0] for p in picked]

    print("\n" + "=" * 80)
    print(f"[*] {len(selected_stems)} SAMPEL POSITIF TERPILIH UNTUK DITUMPUK ATAS-BAWAH:")
    for idx, stem in enumerate(selected_stems, 1):
        info = next((p for p in positive_samples if p[0] == stem), None)
        detail = f"({info[1]:,} px - {info[2]:.2f}%)" if info else ""
        print(f"    {idx}. {stem} {detail}")
    print("=" * 80 + "\n")

    # 5. Siapkan folder output
    os.makedirs(args.output_dir, exist_ok=True)
    indiv_dir = os.path.join(args.output_dir, "individual_samples")
    if args.save_individual:
        os.makedirs(indiv_dir, exist_ok=True)

    # 6. Muat data dan prediksi untuk setiap sampel
    processed_samples = []

    for stem in selected_stems:
        # A. Muat citra RGB asli
        img_file = find_sample_file(image_dir, stem)
        if not img_file:
            print(f"[!] Peringatan: Citra RGB tidak ditemukan untuk {stem}, lewati.")
            continue
        rgb_raw = np.array(Image.open(img_file).convert("RGB"))

        # B. Muat raster DTM asli dengan reader multi-tier (rasterio/cv2/tifffile)
        dtm_file = find_sample_file(dtm_dir, stem)
        if not dtm_file:
            print(f"[!] Peringatan: File DTM tidak ditemukan untuk {stem}, lewati.")
            continue
        dtm_raw, dtm_nodata, engine = read_dtm_raster(dtm_file)
        dtm_norm, val_min, val_max, valid_mask = inspect_and_normalize_dtm(dtm_raw, dtm_nodata, stem, engine)

        # C. Muat Ground Truth
        lbl_file = find_sample_file(label_dir, stem)
        gt_raw = np.array(Image.open(lbl_file))
        gt_bin = (gt_raw > 0).astype(np.float32)

        H, W = gt_bin.shape[:2]

        # Pastikan ukuran RGB & DTM sinkron dengan Ground Truth
        if rgb_raw.shape[:2] != (H, W):
            rgb_raw = cv2.resize(rgb_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        if dtm_norm.shape[:2] != (H, W):
            dtm_norm = cv2.resize(dtm_norm, (W, H), interpolation=cv2.INTER_LINEAR)

        # D. Dapatkan Prediksi RGB-DTM 0% dan RGB-DTM 100%
        if use_checkpoint_mode:
            target_size = config.get("dataset", {}).get("size", 512)
            prob_0 = run_inference_on_sample(model_0, rgb_raw, dtm_norm, is_zero_dtm=True, device=device, target_size=target_size)
            prob_100 = run_inference_on_sample(model_100, rgb_raw, dtm_norm, is_zero_dtm=False, device=device, target_size=target_size)
            pred0_bin = (prob_0 >= args.threshold).astype(np.float32)
            pred100_bin = (prob_100 >= args.threshold).astype(np.float32)
        else:
            file_0 = resolve_prediction_file(args.pred_0_dir, stem)
            file_100 = resolve_prediction_file(args.pred_100_dir, stem)
            pred0_bin = load_prediction_mask(file_0, threshold=args.threshold)
            pred100_bin = load_prediction_mask(file_100, threshold=args.threshold)

        # Pastikan resolusi prediksi selaras dengan GT
        if pred0_bin.shape != (H, W):
            pred0_bin = cv2.resize(pred0_bin, (W, H), interpolation=cv2.INTER_NEAREST)
        if pred100_bin.shape != (H, W):
            pred100_bin = cv2.resize(pred100_bin, (W, H), interpolation=cv2.INTER_NEAREST)

        # E. Hitung metrik perbandingan
        m0 = compute_metrics(pred0_bin, gt_bin)
        m100 = compute_metrics(pred100_bin, gt_bin)

        processed_samples.append({
            "stem": stem,
            "rgb": rgb_raw,
            "dtm": dtm_norm,
            "val_min": val_min,
            "val_max": val_max,
            "gt": gt_bin,
            "pred0": pred0_bin,
            "pred100": pred100_bin,
            "m0": m0,
            "m100": m100,
            "pos_pixels": int(gt_bin.sum()),
            "pct": float(gt_bin.sum() / gt_bin.size * 100.0),
        })

    if len(processed_samples) == 0:
        print("ERROR: Tidak ada sampel yang berhasil diproses.")
        sys.exit(1)

    num_rows = len(processed_samples)

    # =========================================================================
    # 7. RENDERING 1: COMPOSITE FIGURE BERTUMPUK ATAS-BAWAH (CRISP BINARY MASKS)
    # Susunan kolom: [RGB, DTM, Ground Truth, RGB-DTM 0%, RGB-DTM 100%]
    # =========================================================================
    print(f"\n[*] Merender figure 5-baris bertumpuk atas-bawah...")
    fig, axes = plt.subplots(
        num_rows, 5,
        figsize=(23, 4.6 * num_rows),
        facecolor="white",
        squeeze=False
    )

    dtm_cmap = get_dtm_colormap(args.dtm_colormap)

    column_titles = [
        "RGB",
        "DTM",
        "Ground Truth",
        "RGB-DTM 0%",
        "RGB-DTM 100%",
    ]

    for col_idx, col_title in enumerate(column_titles):
        axes[0, col_idx].set_title(col_title, fontsize=15, fontweight="bold", pad=14)

    for row_idx, item in enumerate(processed_samples):
        stem = item["stem"]
        rgb = item["rgb"]
        dtm = item["dtm"]
        gt = item["gt"]
        pred0 = item["pred0"]
        pred100 = item["pred100"]
        m0 = item["m0"]
        m100 = item["m100"]
        H, W = gt.shape[:2]

        # Kolom 0: RGB
        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_ylabel(f"{stem}", fontsize=12, fontweight="bold", labelpad=10)
        apply_edge_scale(axes[row_idx, 0], H, W)

        # Kolom 1: DTM (Colormap Biru -> Hijau -> Kuning -> Coklat -> Putih)
        dtm_im = axes[row_idx, 1].imshow(dtm, cmap=dtm_cmap, vmin=0.0, vmax=1.0)
        apply_edge_scale(axes[row_idx, 1], H, W)

        # Scale warna / colorbar di samping DTM
        divider = make_axes_locatable(axes[row_idx, 1])
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar = fig.colorbar(dtm_im, cax=cax)
        cbar_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])
        cbar.ax.tick_params(labelsize=8)

        # Kolom 2: Ground Truth
        axes[row_idx, 2].imshow(gt, cmap="gray", vmin=0.0, vmax=1.0)
        apply_edge_scale(axes[row_idx, 2], H, W)

        # Kolom 3: RGB-DTM 0%
        axes[row_idx, 3].imshow(pred0, cmap="gray", vmin=0.0, vmax=1.0)
        apply_edge_scale(axes[row_idx, 3], H, W)
        axes[row_idx, 3].set_xlabel(
            f"IoU: {m0['iou']:.4f} | F1: {m0['dice']:.4f}",
            fontsize=11,
            fontweight="bold",
            labelpad=8,
            color="#a80000" if m0['iou'] < m100['iou'] else "#005500",
        )

        # Kolom 4: RGB-DTM 100%
        axes[row_idx, 4].imshow(pred100, cmap="gray", vmin=0.0, vmax=1.0)
        apply_edge_scale(axes[row_idx, 4], H, W)
        axes[row_idx, 4].set_xlabel(
            f"IoU: {m100['iou']:.4f} | F1: {m100['dice']:.4f}",
            fontsize=11,
            fontweight="bold",
            labelpad=8,
            color="#006600" if m100['iou'] >= m0['iou'] else "#a80000",
        )

    plt.tight_layout()
    fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.32, wspace=0.24)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_stacked_binary = os.path.join(args.output_dir, f"test_comparison_stacked_{timestamp}.png")
    out_latest_binary = os.path.join(args.output_dir, "test_comparison_stacked_latest.png")

    plt.savefig(out_stacked_binary, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(out_latest_binary, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Berhasil disimpan: {out_stacked_binary}")
    print(f"[*] Link versi latest : {out_latest_binary}")

    # =========================================================================
    # 8. RENDERING 2: COMPOSITE FIGURE BERTUMPUK ATAS-BAWAH DENGAN OVERLAY RGB
    # =========================================================================
    print(f"\n[*] Merender figure 5-baris bertumpuk atas-bawah (OVERLAY MODE)...")
    fig_ov, axes_ov = plt.subplots(
        num_rows, 5,
        figsize=(23, 4.6 * num_rows),
        facecolor="white",
        squeeze=False
    )

    column_titles_ov = [
        "RGB",
        "DTM",
        "Ground Truth",
        "RGB-DTM 0%",
        "RGB-DTM 100%",
    ]

    for col_idx, col_title in enumerate(column_titles_ov):
        axes_ov[0, col_idx].set_title(col_title, fontsize=15, fontweight="bold", pad=14)

    for row_idx, item in enumerate(processed_samples):
        stem = item["stem"]
        rgb = item["rgb"]
        dtm = item["dtm"]
        gt = item["gt"]
        pred0 = item["pred0"]
        pred100 = item["pred100"]
        m0 = item["m0"]
        m100 = item["m100"]
        H, W = gt.shape[:2]

        # Kolom 0: RGB
        axes_ov[row_idx, 0].imshow(rgb)
        axes_ov[row_idx, 0].set_ylabel(f"{stem}", fontsize=12, fontweight="bold", labelpad=10)
        apply_edge_scale(axes_ov[row_idx, 0], H, W)

        # Kolom 1: DTM
        dtm_im_ov = axes_ov[row_idx, 1].imshow(dtm, cmap=dtm_cmap, vmin=0.0, vmax=1.0)
        apply_edge_scale(axes_ov[row_idx, 1], H, W)

        divider_ov = make_axes_locatable(axes_ov[row_idx, 1])
        cax_ov = divider_ov.append_axes("right", size="5%", pad=0.08)
        cbar_ov = fig_ov.colorbar(dtm_im_ov, cax=cax_ov)
        cbar_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
        cbar_ov.set_ticks(cbar_ticks)
        cbar_ov.set_ticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])
        cbar_ov.ax.tick_params(labelsize=8)

        # Kolom 2: GT Overlay (Hijau)
        gt_overlay = create_overlay_image(rgb, gt, color=(0, 230, 0), alpha=0.45)
        axes_ov[row_idx, 2].imshow(gt_overlay)
        apply_edge_scale(axes_ov[row_idx, 2], H, W)

        # Kolom 3: Pred 0% Overlay (Merah)
        pred0_overlay = create_overlay_image(rgb, pred0, color=(255, 30, 30), alpha=0.45)
        axes_ov[row_idx, 3].imshow(pred0_overlay)
        apply_edge_scale(axes_ov[row_idx, 3], H, W)
        axes_ov[row_idx, 3].set_xlabel(
            f"IoU: {m0['iou']:.4f} | F1: {m0['dice']:.4f}",
            fontsize=11,
            fontweight="bold",
            labelpad=8,
            color="#a80000" if m0['iou'] < m100['iou'] else "#005500",
        )

        # Kolom 4: Pred 100% Overlay (Merah)
        pred100_overlay = create_overlay_image(rgb, pred100, color=(255, 30, 30), alpha=0.45)
        axes_ov[row_idx, 4].imshow(pred100_overlay)
        apply_edge_scale(axes_ov[row_idx, 4], H, W)
        axes_ov[row_idx, 4].set_xlabel(
            f"IoU: {m100['iou']:.4f} | F1: {m100['dice']:.4f}",
            fontsize=11,
            fontweight="bold",
            labelpad=8,
            color="#006600" if m100['iou'] >= m0['iou'] else "#a80000",
        )

    plt.tight_layout()
    fig_ov.subplots_adjust(top=0.96, bottom=0.05, hspace=0.32, wspace=0.24)

    out_stacked_overlay = os.path.join(args.output_dir, f"test_comparison_overlay_{timestamp}.png")
    out_latest_overlay = os.path.join(args.output_dir, "test_comparison_overlay_latest.png")

    plt.savefig(out_stacked_overlay, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(out_latest_overlay, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_ov)
    print(f"[*] Berhasil disimpan: {out_stacked_overlay}")
    print(f"[*] Link versi latest : {out_latest_overlay}")

    # =========================================================================
    # 9. SIMPAN FILE INDIVIDUAL JIKA DIMINTA
    # =========================================================================
    if args.save_individual:
        for item in processed_samples:
            stem = item["stem"]
            H, W = item["gt"].shape[:2]
            fig_single, ax_s = plt.subplots(1, 5, figsize=(23, 5.0), facecolor="white")
            for col_idx, col_title in enumerate(column_titles):
                ax_s[col_idx].set_title(col_title, fontsize=13, fontweight="bold", pad=12)

            ax_s[0].imshow(item["rgb"])
            ax_s[0].set_ylabel(f"{stem}", fontsize=12, fontweight="bold", labelpad=10)
            apply_edge_scale(ax_s[0], H, W)

            dtm_single = ax_s[1].imshow(item["dtm"], cmap=dtm_cmap, vmin=0.0, vmax=1.0)
            apply_edge_scale(ax_s[1], H, W)
            divider_s = make_axes_locatable(ax_s[1])
            cax_s = divider_s.append_axes("right", size="5%", pad=0.08)
            cbar_s = fig_single.colorbar(dtm_single, cax=cax_s)
            cbar_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
            cbar_s.set_ticks(cbar_ticks)
            cbar_s.set_ticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])
            cbar_s.ax.tick_params(labelsize=8)

            ax_s[2].imshow(item["gt"], cmap="gray", vmin=0.0, vmax=1.0)
            apply_edge_scale(ax_s[2], H, W)

            ax_s[3].imshow(item["pred0"], cmap="gray", vmin=0.0, vmax=1.0)
            apply_edge_scale(ax_s[3], H, W)
            ax_s[3].set_xlabel(f"IoU: {item['m0']['iou']:.4f} | F1: {item['m0']['dice']:.4f}", fontsize=11, fontweight="bold", labelpad=8)

            ax_s[4].imshow(item["pred100"], cmap="gray", vmin=0.0, vmax=1.0)
            apply_edge_scale(ax_s[4], H, W)
            ax_s[4].set_xlabel(f"IoU: {item['m100']['iou']:.4f} | F1: {item['m100']['dice']:.4f}", fontsize=11, fontweight="bold", labelpad=8)

            plt.tight_layout()
            out_single = os.path.join(indiv_dir, f"{stem}_comparison.png")
            plt.savefig(out_single, dpi=200, bbox_inches="tight")
            plt.close(fig_single)

        print(f"[*] 5 sampel individual tersimpan di: {indiv_dir}")

    # =========================================================================
    # 10. SIMPAN SUMMARY METRICS KE JSON
    # =========================================================================
    summary_data = {
        "timestamp": timestamp,
        "num_samples": num_rows,
        "threshold": args.threshold,
        "selected_samples": [
            {
                "sample_name": item["stem"],
                "positive_pixels": item["pos_pixels"],
                "positive_percentage": item["pct"],
                "metrics_rgb_dtm_0_percent": item["m0"],
                "metrics_rgb_dtm_100_percent": item["m100"],
                "iou_gain": float(item["m100"]["iou"] - item["m0"]["iou"]),
            }
            for item in processed_samples
        ],
        "mean_metrics": {
            "mean_iou_0%": float(np.mean([item["m0"]["iou"] for item in processed_samples])),
            "mean_iou_100%": float(np.mean([item["m100"]["iou"] for item in processed_samples])),
            "mean_dice_0%": float(np.mean([item["m0"]["dice"] for item in processed_samples])),
            "mean_dice_100%": float(np.mean([item["m100"]["dice"] for item in processed_samples])),
        }
    }

    json_path = os.path.join(args.output_dir, f"test_comparison_metrics_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # =========================================================================
    # 11. CETAK TABEL RINGKASAN METRIK KE TERMINAL
    # =========================================================================
    print("\n" + "=" * 80)
    print("RINGKASAN METRIK PERBANDINGAN PADA 5 SAMPEL POSITIF TERPILIH:")
    print("=" * 80)
    print(f"{'SAMPEL':<22} | {'IoU 0%':<10} | {'IoU 100%':<10} | {'GAIN IoU':<10} | {'F1 0%':<10} | {'F1 100%':<10}")
    print("-" * 80)
    for item in processed_samples:
        gain = item["m100"]["iou"] - item["m0"]["iou"]
        gain_str = f"+{gain:.4f}" if gain >= 0 else f"{gain:.4f}"
        print(
            f"{item['stem']:<22} | "
            f"{item['m0']['iou']:<10.4f} | "
            f"{item['m100']['iou']:<10.4f} | "
            f"{gain_str:<10} | "
            f"{item['m0']['dice']:<10.4f} | "
            f"{item['m100']['dice']:<10.4f}"
        )
    print("-" * 80)
    mean_gain = summary_data["mean_metrics"]["mean_iou_100%"] - summary_data["mean_metrics"]["mean_iou_0%"]
    print(
        f"{'RATA-RATA (MEAN)':<22} | "
        f"{summary_data['mean_metrics']['mean_iou_0%']:<10.4f} | "
        f"{summary_data['mean_metrics']['mean_iou_100%']:<10.4f} | "
        f"{'+' if mean_gain>=0 else ''}{mean_gain:<9.4f} | "
        f"{summary_data['mean_metrics']['mean_dice_0%']:<10.4f} | "
        f"{summary_data['mean_metrics']['mean_dice_100%']:<10.4f}"
    )
    print("=" * 80)
    print(f"\n[SELESAI] Semua visualisasi berhasil dibuat di folder: {args.output_dir}\n")


if __name__ == "__main__":
    main()
