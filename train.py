"""
MVAC v3.0  —  Military Vehicle AI Classifier
Lightning AI training pipeline
────────────────────────────────────────────────────────────────
WHAT'S NEW vs v2.0
  • Focal Loss  (handles class imbalance far better than CE)
  • MixUp regularisation  (prevents overfitting on augmented data)
  • Mosaic augmentation   (synthesises new views for tank/low-data classes)
  • 3-tier augmentation   (ultra / heavy / light based on image count)
  • Differential learning rates  (backbone vs head)
  • Linear warmup + cosine restart scheduler
  • drop_rate=0.3, drop_path_rate=0.2  (stochastic depth)
  • RandomErasing in torchvision transforms
  • TARGET raised to 500 images per class
  • Early stop patience raised to 15 epochs
  • 4 extra Kaggle datasets pulled for more tank/ship coverage

DATASETS USED  (all verified on Kaggle as of 2025)
  Core:
    crsuthikshnkumar/fighter-aircraft-data-set
    crsuthikshnkumar/military-helicopter-data-set
    a2015003713/militaryaircraftdetectiondataset
    antoreepjana/military-tanks-dataset-images
    oleksandershevchenko/ship-classification-dataset
  Supplementary  (tried; skipped gracefully if unavailable):
    iamsouravbanerjee/tank-image-dataset
    farialmahmod/military-vehicles
    amanrajbose/millitary-vechiles
    mexwell/militarycivilian-vehicles-image-classification

NOTE ON IFV / ARTILLERY / AIR-DEFENCE CLASSES
  No public Kaggle dataset has per-class folders for BMP, BTR,
  Bradley, 2S19, HIMARS, Pantsir, S-300 etc.
  Those classes are in CLASS_MAP with broad keyword lists so they
  WILL be picked up if any of the supplementary datasets happen to
  contain matching folder names.  Classes with 0 images at scan
  time are automatically skipped — the model is never trained on
  phantom labels.
────────────────────────────────────────────────────────────────
"""

import subprocess, sys, os

# ================================================================
# INSTALL
# ================================================================
print("Installing packages...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'kagglehub', 'albumentations', 'timm', 'onnx',
    'onnxsim', 'onnxruntime', 'matplotlib',
    'opencv-python-headless', 'pillow', 'numpy'],
    capture_output=False)
print("All packages installed\n")

# ================================================================
# IMPORTS
# ================================================================
import random, json, time, zipfile, cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import albumentations as A
import kagglehub

# ================================================================
# PATHS
# ================================================================
WORK_DIR     = os.path.expanduser('~/MilitaryAI')
DATASET      = f'{WORK_DIR}/dataset'
CKPT_PATH    = f'{WORK_DIR}/best_model.pth'
LAST_PATH    = f'{WORK_DIR}/last_model.pth'
HISTORY_PATH = f'{WORK_DIR}/history.json'
ONNX_PATH    = f'{WORK_DIR}/military_classifier.onnx'
META_PATH    = f'{WORK_DIR}/metadata.json'
ZIP_PATH     = f'{WORK_DIR}/mvac_outputs.zip'
ACC_CURVE_PATH = f'{WORK_DIR}/accuracy_curve.png'
LOSS_CURVE_PATH = f'{WORK_DIR}/loss_curve.png'
LR_CURVE_PATH   = f'{WORK_DIR}/lr_schedule.png'

os.makedirs(WORK_DIR, exist_ok=True)
for split in ['train', 'val', 'test']:
    os.makedirs(f'{DATASET}/{split}', exist_ok=True)
print(f"Working directory: {WORK_DIR}")

# ================================================================
# KAGGLE API AUTHENTICATION
# ================================================================

os.environ['KAGGLE_USERNAME'] = 'YOUR_KAGGLE_USERNAME'
os.environ['KAGGLE_KEY']      = 'YOUR_API_KEY'

if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
    print("WARNING: KAGGLE_USERNAME / KAGGLE_KEY are not set in the environment.")
    print("         Set them as Lightning AI secrets (or export them in your")
    print("         shell) before running, or place a kaggle.json in ~/.kaggle/.")
    print("         Dataset downloads below will fail without credentials.\n")

# ================================================================
# BLOCK 1 — DOWNLOAD DATASETS
# ================================================================
print("\n" + "="*60)
print("DOWNLOADING DATASETS")
print("="*60)

def safe_download(slug):
    try:
        path = kagglehub.dataset_download(slug)
        print(f"  OK   {slug}")
        return path
    except Exception as e:
        print(f"  SKIP {slug}  ({e})")
        return None

fighter_path    = safe_download('crsuthikshnkumar/fighter-aircraft-data-set')
helicopter_path = safe_download('crsuthikshnkumar/military-helicopter-data-set')
aircraft_path   = safe_download('a2015003713/militaryaircraftdetectiondataset')
tanks_path      = safe_download('antoreepjana/military-tanks-dataset-images')
ships_path      = safe_download('oleksandershevchenko/ship-classification-dataset')

# Supplementary — tried; skipped gracefully if unavailable
tanks2_path    = safe_download('iamsouravbanerjee/tank-image-dataset')
milv_path      = safe_download('farialmahmod/military-vehicles')
milv2_path     = safe_download('amanrajbose/millitary-vechiles')
milciv_path    = safe_download('mexwell/militarycivilian-vehicles-image-classification')

# ================================================================
# BLOCK 2 — SCAN
# ================================================================
print("\n" + "="*60)
print("SCANNING FOLDERS")
print("="*60)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

DATASET_PATHS = {k: v for k, v in {
    'fighter':    fighter_path,
    'helicopter': helicopter_path,
    'aircraft':   aircraft_path,
    'tanks':      tanks_path,
    'ships':      ships_path,
    'tanks2':     tanks2_path,
    'milv':       milv_path,
    'milv2':      milv2_path,
    'milciv':     milciv_path,
}.items() if v and os.path.exists(v)}

for ds_name, ds_path in DATASET_PATHS.items():
    print(f"\n--- {ds_name} ---")
    for root, dirs, files in os.walk(ds_path):
        depth = root.replace(ds_path, '').count(os.sep)
        if depth > 3:
            dirs.clear()
            continue
        dirs.sort()
        n = len([f for f in files if Path(f).suffix.lower() in IMG_EXTS])
        if n > 0:
            print(f"  {os.path.basename(root):<40} {n:>5} imgs")

# ================================================================
# BLOCK 3 — CLASS MAP
# ================================================================
print("\n" + "="*60)
print("BUILDING DATASET")
print("="*60)

TARGET = 500
IMG_SZ = 224
SPLITS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEED   = 42
random.seed(SEED)

CLASS_MAP = {
    # ── Jets ──────────────────────────────────────────────────
    'a10_warthog':      ['a10'],
    'b1_lancer':        ['b1'],
    'b2_spirit':        ['b2'],
    'b21_raider':       ['b21'],
    'b52':              ['b52'],
    'c130_hercules':    ['c130'],
    'c17_globemaster':  ['c17'],
    'eurofighter':      ['ef2000'],
    'f117':             ['f117'],
    'f14_tomcat':       ['f14'],
    'f15':              ['f15'],
    'f16':              ['f16'],
    'f18_hornet':       ['f18'],
    'f22_raptor':       ['f22'],
    'f35':              ['f35'],
    'f4_phantom':       ['f4'],
    'gripen':           ['jas39'],
    'j10':              ['j10'],
    'j20':              ['j20'],
    'jf17':             ['jf17'],
    'mig29':            ['mig29'],
    'mig31':            ['mig31'],
    'mirage2000':       ['mirage2000'],
    'mq9_reaper':       ['mq9'],
    'osprey':           ['v22'],
    'rafale':           ['rafale'],
    'rq4_globalhawk':   ['rq4'],
    'sr71_blackbird':   ['sr71'],
    'su24':             ['su24'],
    'su25':             ['su25'],
    'su34':             ['su34'],
    'su57':             ['su57'],
    'tb2_drone':        ['tb2'],
    'tejas':            ['tejas'],
    'tornado':          ['tornado'],
    'tu160':            ['tu160'],
    'tu22m':            ['tu22m'],
    'tu95':             ['tu95'],
    'u2_spyplane':      ['u2'],

    # ── Helicopters ───────────────────────────────────────────
    'apache':           ['ah64', 'apache'],
    'blackhawk':        ['uh60', 'blackhawk'],
    'chinook':          ['ch47', 'chinook'],
    'ka52_alligator':   ['ka52'],
    'mi24_hind':        ['mi24'],
    'mi28_havoc':       ['mi28'],
    'mi8_hip':          ['mi8'],

    # ── Tanks ─────────────────────────────────────────────────
    'm1_abrams':        ['m1_abrams', 'm1a1_abrams', 'm1a2_abrams',
                         'm1a2_sep', 'tusk', 'm1a2', 'm1a1', 'abrams'],
    't90':              ['t90', 't90m', 't90ms_tagil', 't-90'],
    'leopard2':         ['leopard_2', 'leopard_2a4m_can', 'leopard_2a5',
                         'leopard_2a6', 'leopard_2a7', 'leopard_2ng',
                         'leopard_2pl', 'leopard_c2', 'leopard', 'leopard2'],
    'challenger2':      ['challenger_2', 'challenger2'],
    't72':              ['t72', 't72b3', 't72b4', 't72m2_moderna',
                         't72m4', 't72ua1', 't-72'],
    't80':              ['t80', 't80b', 't80bvm', 't80u', 't80ud', 't-80'],
    't64':              ['t64', 't64b1m', 't64bm_bulat', 't64e', 't-64'],
    't55':              ['t55', 't55_enigma', 't-55'],
    't62':              ['t62', 't-62'],
    'merkava':          ['merkava_mk1', 'merkava_mk2', 'merkava_mk3',
                         'merkava_mk4', 'merkava_mk4_meil_ruach', 'merkava'],
    'leclerc':          ['leclerc'],
    'k2_blackpanther':  ['k2_black_panther_mbt', 'k2'],
    'type99':           ['type_99', 'type_99g', 'type99'],
    'type96':           ['type_96', 'type96'],
    'type90_japan':     ['type_90', 'type_90_II', 'type90'],
    'ariete':           ['ariete'],
    'armata':           ['armata', 't14', 't-14'],
    'k1_rok':           ['k1', 'k1a1'],
    'oplot':            ['oplot', 'oplot_m'],
    'arjun':            ['arjun', 'arjun_mk2'],
    'pt91_twardy':      ['pt91_twardy', 'pt91'],
    'sabra':            ['sabra'],

    # ── Naval ─────────────────────────────────────────────────
    'aircraft_carrier': ['aircraft carrier', 'carrier'],
    'destroyer_ddg':    ['ddg', 'destroyer'],
    'submarine':        ['submarine'],

    # ── IFV / APC  ────────────────────────────────────────────
    # Picked up if supplementary datasets have these folder names
    'bmp_ifv':          ['bmp', 'bmp-1', 'bmp-2', 'bmp-3',
                         'bmp1', 'bmp2', 'bmp3', 'bmp_ifv'],
    'btr_apc':          ['btr', 'btr-60', 'btr-70', 'btr-80',
                         'btr60', 'btr70', 'btr80', 'btr_apc'],
    'bradley':          ['bradley', 'm2_bradley', 'm2bradley',
                         'm2 bradley', 'ifv'],
    'stryker':          ['stryker', 'm1126', 'm1126_stryker'],

    # ── Artillery / MLRS  ─────────────────────────────────────
    '2s19_msta':        ['2s19', 'msta', '2s19_msta', '2s19msta'],
    'bm21_grad':        ['bm21', 'bm-21', 'grad', 'bm_21'],
    'bm30_smerch':      ['bm30', 'bm-30', 'smerch'],
    'm109_paladin':     ['m109', 'paladin', 'm109a6', 'm109a7',
                         'm109_paladin'],
    'himars':           ['himars', 'm142', 'm142_himars'],

    # ── Air Defence  ──────────────────────────────────────────
    'pantsir_s1':       ['pantsir', 'pantsir_s1', 'pantsir-s1',
                         'pantsir_s2'],
    's300_sam':         ['s300', 's-300', 'sa-10', 's300_sam'],
    's400_triumf':      ['s400', 's-400', 'triumf', 's400_triumf'],
    'patriot_sam':      ['patriot', 'mim104', 'mim-104',
                         'patriot_sam'],
}

# ================================================================
# BLOCK 4 — AUGMENTATION PIPELINES
# ================================================================

aug_ultra = A.Compose([            # < 50 images
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=60, p=0.9),
    A.RandomScale(scale_limit=0.4, p=0.6),
    A.ColorJitter(brightness=0.5, contrast=0.5,
                  saturation=0.4, hue=0.2, p=0.9),
    A.CLAHE(clip_limit=6.0, p=0.6),
    # NOTE: current albumentations expresses GaussNoise strength as
    # std_range (std-dev as a fraction of 255), not the old var_limit
    # (variance in raw 0-255 units). Converted so the noise strength
    # matches the original var_limit=(30.0, 120.0) intent:
    #   std = sqrt(var) / 255  ->  sqrt(30)/255=.0215 , sqrt(120)/255=.0430
    A.GaussNoise(std_range=(0.0215, 0.0430), p=0.6),
    A.GaussianBlur(blur_limit=(3, 9), p=0.4),
    A.CoarseDropout(num_holes_range=(3, 8),
                    hole_height_range=(20, 60),
                    hole_width_range=(20, 60),
                    fill=0, p=0.5),
    A.RandomBrightnessContrast(p=0.6),
    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3,
                       rotate_limit=45, p=0.7),
    A.Perspective(scale=(0.05, 0.15), p=0.4),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    A.Resize(IMG_SZ, IMG_SZ),
])

aug_heavy = A.Compose([            # 50–150 images
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=45, p=0.8),
    A.RandomScale(scale_limit=0.3, p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4,
                  saturation=0.3, hue=0.15, p=0.8),
    A.CLAHE(clip_limit=4.0, p=0.5),
    # Converted from var_limit=(20.0, 80.0): sqrt(var)/255
    A.GaussNoise(std_range=(0.0175, 0.0351), p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.CoarseDropout(num_holes_range=(2, 6),
                    hole_height_range=(15, 50),
                    hole_width_range=(15, 50),
                    fill=0, p=0.4),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                       rotate_limit=30, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.Resize(IMG_SZ, IMG_SZ),
])

aug_light = A.Compose([            # > 150 images
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.1, p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.2),
    # Converted from var_limit=(5.0, 30.0): sqrt(var)/255
    A.GaussNoise(std_range=(0.0088, 0.0215), p=0.2),
    A.CoarseDropout(num_holes_range=(1, 3),
                    hole_height_range=(10, 30),
                    hole_width_range=(10, 30),
                    fill=0, p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.Resize(IMG_SZ, IMG_SZ),
])

resize_only = A.Resize(IMG_SZ, IMG_SZ)

def read_rgb(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

def save_jpg(img_rgb, path):
    cv2.imwrite(path,
                cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

def make_mosaic(imgs, size=IMG_SZ):
    """4-tile mosaic: combines 4 images into one synthetic sample."""
    half   = size // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for (r, c), img in zip([(0,0),(0,half),(half,0),(half,half)], imgs[:4]):
        canvas[r:r+half, c:c+half] = cv2.resize(img, (half, half))
    return canvas

# ================================================================
# BLOCK 5 — COLLECT + BUILD SPLITS
# ================================================================
print("Scanning datasets...\n")
class_images = {cls: [] for cls in CLASS_MAP}

for ds_name, ds_root in DATASET_PATHS.items():
    for root, dirs, files in os.walk(ds_root):
        fname  = os.path.basename(root)
        flower = fname.lower()
        for cls, keywords in CLASS_MAP.items():
            if any(flower == kw.lower() for kw in keywords):
                imgs = [os.path.join(root, f) for f in files
                        if Path(f).suffix.lower() in IMG_EXTS]
                if imgs:
                    class_images[cls].extend(imgs)
                    print(f"  [{ds_name:>10}]  {cls:<22}  +{len(imgs):>4}"
                          f"  ('{fname}')")
        dirs.sort()

for cls in class_images:
    class_images[cls] = list(set(class_images[cls]))

# Tier assignment
print(f"\n{'─'*70}")
print(f"  {'CLASS':<28} {'IMGS':>6}  {'TIER':>8}")
print(f"{'─'*70}")
good_classes = []
class_tier   = {}
TIER_PIPELINE = {'ultra': aug_ultra, 'heavy': aug_heavy, 'light': aug_light}

for cls in sorted(CLASS_MAP):
    n = len(class_images[cls])
    if n == 0:
        tier = 'skip'
    elif n < 50:
        tier = 'ultra'
    elif n < 150:
        tier = 'heavy'
    else:
        tier = 'light'
    class_tier[cls] = tier
    if tier != 'skip':
        good_classes.append(cls)
        print(f"  {cls:<28} {n:>6}  {tier:>8}")
    else:
        print(f"  {cls:<28} {n:>6}  {'SKIPPED — no images':>20}")

print(f"{'─'*70}")
print(f"  Active classes: {len(good_classes)}\n")

# Build splits
class_names_final = sorted(good_classes)
class_counts      = {}

for cls in class_names_final:
    imgs = class_images[cls][:]
    random.shuffle(imgs)
    n       = len(imgs)
    n_train = max(1, int(n * SPLITS['train']))
    n_val   = max(1, int(n * SPLITS['val']))
    split_data = {
        'train': imgs[:n_train],
        'val':   imgs[n_train:n_train+n_val],
        'test':  imgs[n_train+n_val:],
    }
    tier     = class_tier[cls]
    pipeline = TIER_PIPELINE[tier]
    counts   = {}

    for sname, slist in split_data.items():
        if not slist:
            counts[sname] = 0
            continue
        out_dir = f'{DATASET}/{sname}/{cls}'
        os.makedirs(out_dir, exist_ok=True)
        saved = 0
        for p in slist:
            img = read_rgb(p)
            if img is None:
                continue
            save_jpg(resize_only(image=img)['image'],
                     f'{out_dir}/{saved:05d}.jpg')
            saved += 1
        if sname == 'train':
            attempts = 0
            while saved < TARGET and attempts < TARGET * 15:
                attempts += 1
                img = read_rgb(random.choice(slist))
                if img is None:
                    continue
                try:
                    # Mosaic every 4th augmented sample for ultra/heavy
                    if (tier in ('ultra', 'heavy')
                            and saved % 4 == 0
                            and len(slist) >= 4):
                        imgs4 = [i for _ in range(4)
                                 if (i := read_rgb(random.choice(slist))) is not None]
                        result = make_mosaic(imgs4) if len(imgs4) == 4 \
                                 else pipeline(image=img)['image']
                    else:
                        result = pipeline(image=img)['image']
                    save_jpg(result, f'{out_dir}/{saved:05d}a.jpg')
                    saved += 1
                except Exception:
                    continue
        counts[sname] = saved

    class_counts[cls] = counts.get('train', 0)
    print(f"  {cls:<28}  train={counts.get('train',0):>4}"
          f"  val={counts.get('val',0):>4}  [{tier}]")

with open(f'{WORK_DIR}/class_names.json', 'w') as f:
    json.dump(class_names_final, f, indent=2)
with open(f'{WORK_DIR}/class_map.json', 'w') as f:
    json.dump({c: i for i, c in enumerate(class_names_final)}, f, indent=2)
with open(f'{WORK_DIR}/class_counts.json', 'w') as f:
    json.dump(class_counts, f, indent=2)

print(f"\nDataset ready!  Classes={len(class_names_final)}  Path={DATASET}")

# ================================================================
# BLOCK 6 — FOCAL LOSS
# ================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss with per-class alpha and label smoothing.
    gamma=2 focuses on hard/rare examples (tanks, IFV, AD).
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        with torch.no_grad():
            smooth = torch.zeros_like(logits)
            smooth.fill_(self.label_smoothing / (num_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)
        ce        = -(smooth * log_probs).sum(dim=1)
        p_t       = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_w   = (1.0 - p_t) ** self.gamma
        loss      = focal_w * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()

# ================================================================
# BLOCK 7 — MIXUP
# ================================================================
def mixup_data(x, y, alpha=0.3, device='cpu'):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0)).to(device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)

# ================================================================
# BLOCK 8 — DATASET + TRANSFORMS
# ================================================================
with open(f'{WORK_DIR}/class_names.json') as f:
    CLASS_NAMES = json.load(f)
with open(f'{WORK_DIR}/class_counts.json') as f:
    CLASS_COUNTS = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)
print(f"\nTraining on {NUM_CLASSES} classes")

class MilDataset(Dataset):
    def __init__(self, folder, transform):
        self.samples   = []
        self.transform = transform
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(folder, cls)
            if not os.path.isdir(cls_dir):
                continue
            idx = CLASS_NAMES.index(cls)
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp')):
                    self.samples.append((os.path.join(cls_dir, fn), idx))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert('RGB')), label
    def get_labels(self):
        return [s[1] for s in self.samples]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SZ, IMG_SZ)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomRotation(20),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SZ, IMG_SZ)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = MilDataset(f'{DATASET}/train', train_tf)
val_ds   = MilDataset(f'{DATASET}/val',   val_tf)
print(f"Train: {len(train_ds)}   Val: {len(val_ds)}")

# ================================================================
# BLOCK 9 — WEIGHTED SAMPLER
# ================================================================
labels              = train_ds.get_labels()
class_sample_counts = np.array(
    [CLASS_COUNTS.get(CLASS_NAMES[i], 1) for i in range(NUM_CLASSES)])
weights_per_class   = 1.0 / class_sample_counts
sample_weights      = torch.from_numpy(
    np.array([weights_per_class[l] for l in labels])).float()

sampler = WeightedRandomSampler(
    weights=sample_weights.tolist(),
    num_samples=len(sample_weights),
    replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True)

# ================================================================
# BLOCK 10 — MODEL
# ================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

model = timm.create_model(
    'tf_efficientnetv2_m',
    pretrained=True,
    num_classes=NUM_CLASSES,
    drop_rate=0.3,
    drop_path_rate=0.2,
)
model = model.to(device)
print(f"Model: tf_efficientnetv2_m  |  Classes: {NUM_CLASSES}")

# ================================================================
# BLOCK 11 — LOSS / OPTIMISER / SCHEDULER
# ================================================================
class_weights = torch.tensor(
    [1.0 / max(CLASS_COUNTS.get(c, 1), 1) for c in CLASS_NAMES],
    dtype=torch.float32)
class_weights = (class_weights / class_weights.mean()).to(device)

criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)

# Differential learning rates: backbone slower, head faster.
#
# timm's tf_efficientnetv2_m (EfficientNet class) attribute chain is:
#   conv_stem -> bn1 -> blocks -> conv_head -> bn2 -> global_pool -> classifier
#
# The previous version of this optimizer covered conv_stem, bn1, blocks,
# conv_head and classifier, but never included bn2 (the BatchNorm layer
# that sits between conv_head and the pooled classifier). Any parameter
# that is never handed to the optimizer simply never receives a gradient
# update, so bn2's weight/bias would have stayed frozen at their
# pretrained/initialised values for the entire run. Fixed by adding a
# dedicated bn2 group at the same LR as the rest of the head, since it
# functionally belongs to the head/classifier stage.
conv_stem_params  = list(model.conv_stem.parameters())
bn1_params        = list(model.bn1.parameters())
blocks_params     = list(model.blocks.parameters())
conv_head_params  = list(model.conv_head.parameters())
bn2_params        = list(model.bn2.parameters())
classifier_params = list(model.classifier.parameters())

BASE_LR = 3e-4
param_groups = [
    {'params': conv_stem_params,  'lr': BASE_LR * 0.1},
    {'params': bn1_params,        'lr': BASE_LR * 0.1},
    {'params': blocks_params,     'lr': BASE_LR * 0.5},
    {'params': conv_head_params,  'lr': BASE_LR},
    {'params': bn2_params,        'lr': BASE_LR},
    {'params': classifier_params, 'lr': BASE_LR},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

# ── Sanity check: every trainable parameter is covered exactly once ──
_opt_ids = [id(p) for grp in param_groups for p in grp['params']]
assert len(_opt_ids) == len(set(_opt_ids)), \
    "Duplicate parameter found across optimizer groups!"
_model_ids = {id(p) for p in model.parameters() if p.requires_grad}
_missing   = _model_ids - set(_opt_ids)
assert not _missing, \
    f"{len(_missing)} trainable parameters are missing from the optimizer!"
print(f"Optimizer parameter check passed: {len(_opt_ids)} tensors across "
      f"{len(param_groups)} groups, 0 missing, 0 duplicated.")

EPOCHS        = 60
EARLY_STOP    = 15
WARMUP_EPOCHS = 5
MIXUP_ALPHA   = 0.3

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=1, eta_min=1e-6)
warmup_sched = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda ep: (ep + 1) / WARMUP_EPOCHS if ep < WARMUP_EPOCHS else 1.0)

# ================================================================
# BLOCK 12 — TRAINING LOOP
# ================================================================
best_val_acc     = 0.0
no_improve_count = 0
history          = []

print(f"\nStarting training — max {EPOCHS} epochs")
print(f"Focal Loss  |  MixUp={MIXUP_ALPHA}  |  EarlyStop={EARLY_STOP}")
print(f"{'─'*100}")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ── Train ────────────────────────────────────────────────
    model.train()
    tr_loss = tr_correct = tr_total = 0

    for imgs, lbls in train_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        use_mixup = (MIXUP_ALPHA > 0) and (random.random() > 0.5)
        if use_mixup:
            imgs, ya, yb, lam = mixup_data(imgs, lbls, MIXUP_ALPHA, device)

        optimizer.zero_grad()
        out = model(imgs)

        if use_mixup:
            loss    = mixup_criterion(criterion, out, ya, yb, lam)
            pred    = out.argmax(1)
            correct = (lam*(pred==ya).float() + (1-lam)*(pred==yb).float()).sum().item()
        else:
            loss    = criterion(out, lbls)
            correct = (out.argmax(1) == lbls).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tr_loss    += loss.item() * imgs.size(0)
        tr_correct += correct
        tr_total   += imgs.size(0)

    if epoch <= WARMUP_EPOCHS:
        warmup_sched.step()
    else:
        scheduler.step()

    # ── Validate ─────────────────────────────────────────────
    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            out  = model(imgs)
            loss = criterion(out, lbls)
            v_loss    += loss.item() * imgs.size(0)
            v_correct += (out.argmax(1) == lbls).sum().item()
            v_total   += imgs.size(0)

    t_acc   = 100 * tr_correct / tr_total
    v_acc   = 100 * v_correct  / v_total
    t_loss  = tr_loss / tr_total
    v_loss_ = v_loss  / v_total
    cur_lr  = optimizer.param_groups[-1]['lr']
    elapsed = time.time() - t0

    print(f"Epoch {epoch:02d}/{EPOCHS}"
          f"  Train {t_acc:6.2f}%  Loss {t_loss:.4f}"
          f"  Val {v_acc:6.2f}%  Loss {v_loss_:.4f}"
          f"  LR {cur_lr:.6f}  {elapsed:.1f}s")

    history.append({'epoch': epoch, 'train_acc': round(t_acc, 4),
                    'val_acc': round(v_acc, 4), 'train_loss': round(t_loss, 6),
                    'val_loss': round(v_loss_, 6), 'lr': cur_lr})

    if v_acc > best_val_acc:
        best_val_acc     = v_acc
        no_improve_count = 0
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'val_acc': v_acc, 'class_names': CLASS_NAMES}, CKPT_PATH)
        print(f"  >>> BEST MODEL SAVED  val_acc={v_acc:.2f}%")
    else:
        no_improve_count += 1
        print(f"  --- No improvement {no_improve_count}/{EARLY_STOP}")

    if epoch % 5 == 0:
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'val_acc': v_acc, 'class_names': CLASS_NAMES}, LAST_PATH)
        print(f"  --- Safety checkpoint epoch {epoch}")

    if no_improve_count >= EARLY_STOP:
        print(f"\nEarly stopping at epoch {epoch}")
        break

with open(HISTORY_PATH, 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE")
print(f"  Best val accuracy : {best_val_acc:.2f}%")
print(f"  Stopped at epoch  : {history[-1]['epoch']}")
print(f"{'='*60}")
for h in history:
    mark = ' <<< BEST' if abs(h['val_acc'] - best_val_acc) < 0.001 else ''
    print(f"  Ep {h['epoch']:02d}  Train {h['train_acc']:6.2f}%"
          f"  Val {h['val_acc']:6.2f}%{mark}")

# ================================================================
# BLOCK 12.5 — TRAINING CURVES (for GitHub README)
# ================================================================
print("\n" + "="*60)
print("GENERATING TRAINING CURVES")
print("="*60)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

epochs_x  = [h['epoch']      for h in history]
train_acc = [h['train_acc']  for h in history]
val_acc   = [h['val_acc']    for h in history]
train_ls  = [h['train_loss'] for h in history]
val_ls    = [h['val_loss']   for h in history]
lrs       = [h['lr']         for h in history]

plt.figure(figsize=(8, 5))
plt.plot(epochs_x, train_acc, label='Train Accuracy')
plt.plot(epochs_x, val_acc,   label='Val Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
plt.title('MVAC v3.0 — Training / Validation Accuracy')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ACC_CURVE_PATH, dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_x, train_ls, label='Train Loss')
plt.plot(epochs_x, val_ls,   label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Focal Loss')
plt.title('MVAC v3.0 — Training / Validation Loss')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(LOSS_CURVE_PATH, dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_x, lrs)
plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
plt.title('MVAC v3.0 — LR Schedule (Warmup + Cosine Restarts)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(LR_CURVE_PATH, dpi=150)
plt.close()

print(f"Saved: {ACC_CURVE_PATH}")
print(f"Saved: {LOSS_CURVE_PATH}")
print(f"Saved: {LR_CURVE_PATH}")

# ================================================================
# BLOCK 13 — EXPORT ONNX + METADATA + ZIP
# ================================================================
print("\n" + "="*60)
print("EXPORTING ONNX")
print("="*60)

import onnx
import onnxruntime as ort

ckpt_file = CKPT_PATH if os.path.exists(CKPT_PATH) else LAST_PATH
if not os.path.exists(ckpt_file):
    raise FileNotFoundError("No checkpoint found.")

ckpt        = torch.load(ckpt_file, map_location='cpu')
CLASS_NAMES = ckpt['class_names']
NUM_CLASSES = len(CLASS_NAMES)
print(f"Epoch {ckpt['epoch']}  |  Val acc {ckpt['val_acc']:.2f}%  |  {NUM_CLASSES} classes")

model_ex = timm.create_model('tf_efficientnetv2_m',
                              pretrained=False, num_classes=NUM_CLASSES)
model_ex.load_state_dict(ckpt['model_state'])
model_ex.eval()

dummy = torch.randn(1, 3, 224, 224)
try:
    torch.onnx.export(model_ex, dummy, ONNX_PATH,
                      input_names=['image'], output_names=['logits'],
                      dynamic_axes={'image': {0:'batch'}, 'logits': {0:'batch'}},
                      opset_version=14, do_constant_folding=True)
except Exception as e:
    print(f"Export warning: {e} — retrying with tuple input")
    torch.onnx.export(model_ex, (dummy,), ONNX_PATH,
                      input_names=['image'], output_names=['logits'],
                      dynamic_axes={'image': {0:'batch'}, 'logits': {0:'batch'}},
                      opset_version=14, do_constant_folding=True)

print(f"ONNX exported  ({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)")

try:
    from onnxsim import simplify
    m = onnx.load(ONNX_PATH)
    onnx.checker.check_model(m)
    m_sim, ok = simplify(m)
    if ok:
        onnx.save(m_sim, ONNX_PATH)
    print(f"ONNX simplified + verified  ({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)")
except Exception as e:
    print(f"Simplification skipped: {e}")

# NUM_CLASSES here is derived from the actual trained checkpoint
# (ckpt['class_names']), and model_ex was built with num_classes=NUM_CLASSES,
# so CLASS_NAMES / NUM_CLASSES / model output dim / ONNX output dim all
# agree by construction. The sanity check below re-verifies the ONNX side.
meta = {
    "model_name":     "tf_efficientnetv2_m",
    "input_size":     [1, 3, 224, 224],
    "class_names":    CLASS_NAMES,
    "num_classes":    NUM_CLASSES,
    "val_accuracy":   round(ckpt['val_acc'], 4),
    "trained_epoch":  ckpt['epoch'],
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std":  [0.229, 0.224, 0.225],
    "onnx_opset":     14,
    "project":        "MVAC v3.0",
}
with open(META_PATH, 'w') as f:
    json.dump(meta, f, indent=2)
print("metadata.json saved")

# Sanity check
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
out  = sess.run(['logits'], {'image': np.zeros((1,3,224,224), np.float32)})[0]
print(f"Sanity check: output {out.shape}  — "
      + ("PASSED" if out.shape[1] == NUM_CLASSES else "WARNING shape mismatch"))

# Zip
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(ONNX_PATH, 'military_classifier.onnx')
    zf.write(META_PATH, 'metadata.json')
    if os.path.exists(CKPT_PATH):
        zf.write(CKPT_PATH, 'best_model.pth')
    if os.path.exists(HISTORY_PATH):
        zf.write(HISTORY_PATH, 'history.json')
    if os.path.exists(ACC_CURVE_PATH):
        zf.write(ACC_CURVE_PATH, 'accuracy_curve.png')
    if os.path.exists(LOSS_CURVE_PATH):
        zf.write(LOSS_CURVE_PATH, 'loss_curve.png')
    if os.path.exists(LR_CURVE_PATH):
        zf.write(LR_CURVE_PATH, 'lr_schedule.png')

print(f"Zip: {ZIP_PATH}  ({os.path.getsize(ZIP_PATH)/1e6:.1f} MB)")
print(f"\n{'='*55}")
print(f"  ALL DONE — MVAC v3.0")
print(f"  Classes      : {NUM_CLASSES}")
print(f"  Val accuracy : {ckpt['val_acc']:.2f}%")
print(f"  ONNX         : {os.path.getsize(ONNX_PATH)/1e6:.1f} MB")
print(f"{'='*55}")
print(f"\n  Download: right-click mvac_outputs.zip → Download")
print(f"  Then upload to Google Drive and transfer to PC")
print(f"  Curves for GitHub README: accuracy_curve.png, loss_curve.png, lr_schedule.png")