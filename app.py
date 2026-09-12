# -*- coding: utf-8 -*-
"""
MVAC v3.0 — Military Vehicle AI Classifier
app.py — offline inference desktop app
Aligned with train.py metadata.json keys (normalize_mean / normalize_std)
"""

import tkinter as tk
from tkinter import filedialog
import json, numpy as np
import onnxruntime as ort
from PIL import Image, ImageTk, ImageDraw
import os, math, datetime

# ==============================================================
# MODEL + DATA
# ==============================================================
try:
    with open('metadata.json') as f:
        meta = json.load(f)
    CLASS_NAMES = meta['class_names']
    # ── train.py saves normalize_mean / normalize_std ──
    MEAN        = np.array(meta['normalize_mean'], dtype=np.float32)
    STD         = np.array(meta['normalize_std'],  dtype=np.float32)
    IMG_SIZE    = meta.get('input_size', [1, 3, 224, 224])[2]   # [1,3,H,W]
    VAL_ACC     = meta.get('val_accuracy', 0.0)
    session     = ort.InferenceSession(
                    'military_classifier.onnx',
                    providers=['CPUExecutionProvider'])
    MODEL_LOADED = True
    print(f"Model loaded  |  {len(CLASS_NAMES)} classes  |  val acc {VAL_ACC*100:.2f}%")
except Exception as e:
    print(f"[WARN] Model not loaded: {e}")
    CLASS_NAMES  = ['unknown']
    MEAN         = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD          = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    IMG_SIZE     = 224
    VAL_ACC      = 0.0
    session      = None
    MODEL_LOADED = False

# ==============================================================
# VEHICLE INFO  (all classes from train.py CLASS_MAP)
# ==============================================================
VEHICLE_INFO = {
    # ── Jets / Fixed Wing ─────────────────────────────────────
    'a10_warthog'     : {'country': 'USA',            'role': 'Ground Attack',            'speed': '706 km/h',   'weapons': 'GAU-8 Cannon, Maverick Missiles'},
    'b1_lancer'       : {'country': 'USA',            'role': 'Strategic Bomber',         'speed': '1448 km/h',  'weapons': 'JASSM, JDAM, Mk-82 Bombs'},
    'b2_spirit'       : {'country': 'USA',            'role': 'Stealth Bomber',           'speed': '1010 km/h',  'weapons': 'Nuclear / Conventional Bombs'},
    'b21_raider'      : {'country': 'USA',            'role': 'Stealth Bomber',           'speed': 'Classified', 'weapons': 'Nuclear / Conventional Bombs'},
    'b52'             : {'country': 'USA',            'role': 'Strategic Bomber',         'speed': '1000 km/h',  'weapons': 'Cruise Missiles, Bombs'},
    'c130_hercules'   : {'country': 'USA',            'role': 'Transport Aircraft',       'speed': '643 km/h',   'weapons': 'Defensive Systems'},
    'c17_globemaster' : {'country': 'USA',            'role': 'Strategic Transport',      'speed': '833 km/h',   'weapons': 'None - Transport'},
    'eurofighter'     : {'country': 'Europe',         'role': 'Multirole Fighter',        'speed': '2495 km/h',  'weapons': 'AMRAAM, IRIS-T, Mauser Cannon'},
    'f117'            : {'country': 'USA',            'role': 'Stealth Attack',           'speed': '993 km/h',   'weapons': 'Laser-Guided Bombs'},
    'f14_tomcat'      : {'country': 'USA',            'role': 'Air Superiority',          'speed': '2485 km/h',  'weapons': 'AIM-54 Phoenix, AIM-9, M61 Cannon'},
    'f15'             : {'country': 'USA',            'role': 'Air Superiority',          'speed': '2655 km/h',  'weapons': 'AIM-9, AIM-7, M61 Cannon'},
    'f16'             : {'country': 'USA',            'role': 'Multirole Fighter',        'speed': '2120 km/h',  'weapons': 'AIM-9, AIM-120, M61 Cannon'},
    'f18_hornet'      : {'country': 'USA',            'role': 'Multirole Fighter',        'speed': '1915 km/h',  'weapons': 'AIM-120, AIM-9, M61 Cannon'},
    'f22_raptor'      : {'country': 'USA',            'role': 'Stealth Fighter',          'speed': '1960 km/h',  'weapons': 'AIM-9X, AIM-120, M61A2'},
    'f35'             : {'country': 'USA',            'role': 'Stealth Multirole',        'speed': '1960 km/h',  'weapons': 'AIM-120, AIM-9X, GAU-22'},
    'f4_phantom'      : {'country': 'USA',            'role': 'Multirole Fighter',        'speed': '2370 km/h',  'weapons': 'AIM-7, AIM-9, Bombs'},
    'gripen'          : {'country': 'Sweden',         'role': 'Multirole Fighter',        'speed': '2204 km/h',  'weapons': 'AMRAAM, IRIS-T, Mauser Cannon'},
    'j10'             : {'country': 'China',          'role': 'Multirole Fighter',        'speed': '2200 km/h',  'weapons': 'PL-8, PL-12 Missiles'},
    'j20'             : {'country': 'China',          'role': 'Stealth Fighter',          'speed': '2100 km/h',  'weapons': 'PL-12, PL-15 Missiles'},
    'jf17'            : {'country': 'Pakistan/China', 'role': 'Multirole Fighter',        'speed': '1960 km/h',  'weapons': 'SD-10, PL-5E, CM-400AKG'},
    'mig29'           : {'country': 'Russia',         'role': 'Air Superiority',          'speed': '2400 km/h',  'weapons': 'R-60, R-73, GSh-30 Cannon'},
    'mig31'           : {'country': 'Russia',         'role': 'Interceptor',              'speed': '3000 km/h',  'weapons': 'R-33, R-37, GSh-6-23 Cannon'},
    'mirage2000'      : {'country': 'France',         'role': 'Multirole Fighter',        'speed': '2336 km/h',  'weapons': 'MICA, Magic 2, DEFA Cannon'},
    'mq9_reaper'      : {'country': 'USA',            'role': 'Attack Drone',             'speed': '482 km/h',   'weapons': 'Hellfire Missiles, GBU-12 Bombs'},
    'osprey'          : {'country': 'USA',            'role': 'Tiltrotor Aircraft',       'speed': '565 km/h',   'weapons': 'M240, GAU-17 Minigun'},
    'rafale'          : {'country': 'France',         'role': 'Multirole Fighter',        'speed': '1912 km/h',  'weapons': 'MICA, Exocet, GIAT Cannon'},
    'rq4_globalhawk'  : {'country': 'USA',            'role': 'Reconnaissance Drone',     'speed': '629 km/h',   'weapons': 'None - Surveillance Only'},
    'sr71_blackbird'  : {'country': 'USA',            'role': 'Strategic Reconnaissance', 'speed': '3540 km/h',  'weapons': 'None - Reconnaissance Only'},
    'su24'            : {'country': 'Russia',         'role': 'Strike Aircraft',          'speed': '1700 km/h',  'weapons': 'Kh-23, GSh-6-23 Cannon'},
    'su25'            : {'country': 'Russia',         'role': 'Ground Attack',            'speed': '950 km/h',   'weapons': 'GSh-30 Cannon, S-24 Rockets'},
    'su34'            : {'country': 'Russia',         'role': 'Strike Fighter',           'speed': '1900 km/h',  'weapons': 'Kh-29, Kh-59, GSh-30 Cannon'},
    'su57'            : {'country': 'Russia',         'role': 'Stealth Fighter',          'speed': '2600 km/h',  'weapons': 'R-77, R-74, GSh-30 Cannon'},
    'tb2_drone'       : {'country': 'Turkey',         'role': 'Combat Drone',             'speed': '220 km/h',   'weapons': 'MAM-L, MAM-C Smart Munitions'},
    'tejas'           : {'country': 'India',          'role': 'Multirole Fighter',        'speed': '1920 km/h',  'weapons': 'Derby, Astra, R-73 Missiles'},
    'tornado'         : {'country': 'UK / Germany',   'role': 'Strike Fighter',           'speed': '2417 km/h',  'weapons': 'ALARM, Brimstone, Mauser Cannon'},
    'tu160'           : {'country': 'Russia',         'role': 'Strategic Bomber',         'speed': '2220 km/h',  'weapons': 'Kh-55, Kh-101 Cruise Missiles'},
    'tu22m'           : {'country': 'Russia',         'role': 'Strategic Bomber',         'speed': '2000 km/h',  'weapons': 'Kh-22, Kh-32 Missiles'},
    'tu95'            : {'country': 'Russia',         'role': 'Strategic Bomber',         'speed': '920 km/h',   'weapons': 'Kh-55, Kh-101 Cruise Missiles'},
    'u2_spyplane'     : {'country': 'USA',            'role': 'Strategic Reconnaissance', 'speed': '805 km/h',   'weapons': 'None - Surveillance Only'},
    # ── Helicopters ───────────────────────────────────────────
    'apache'          : {'country': 'USA',            'role': 'Attack Helicopter',        'speed': '293 km/h',   'weapons': 'Hellfire Missiles, M230 Cannon'},
    'blackhawk'       : {'country': 'USA',            'role': 'Utility Helicopter',       'speed': '294 km/h',   'weapons': 'M60 Machine Gun, Rockets'},
    'chinook'         : {'country': 'USA',            'role': 'Transport Helicopter',     'speed': '315 km/h',   'weapons': 'M134 Minigun'},
    'ka52_alligator'  : {'country': 'Russia',         'role': 'Attack Helicopter',        'speed': '300 km/h',   'weapons': 'Vikhr Missiles, GSh-30 Cannon'},
    'mi24_hind'       : {'country': 'Russia',         'role': 'Attack Helicopter',        'speed': '335 km/h',   'weapons': 'AT-6 Spirals, S-8 Rockets, YaKB Cannon'},
    'mi28_havoc'      : {'country': 'Russia',         'role': 'Attack Helicopter',        'speed': '324 km/h',   'weapons': 'Ataka Missiles, 2A42 Cannon'},
    'mi8_hip'         : {'country': 'Russia',         'role': 'Transport Helicopter',     'speed': '250 km/h',   'weapons': 'S-8 Rockets, PKT Machine Gun'},
    # ── Tanks ─────────────────────────────────────────────────
    'm1_abrams'       : {'country': 'USA',            'role': 'Main Battle Tank',         'speed': '67 km/h',    'weapons': '120mm Smoothbore Gun, M2HB'},
    'armata'          : {'country': 'Russia',         'role': 'Main Battle Tank',         'speed': '80 km/h',    'weapons': '125mm 2A82 Gun, Afganit APS'},
    't90'             : {'country': 'Russia',         'role': 'Main Battle Tank',         'speed': '65 km/h',    'weapons': '125mm Smoothbore Gun'},
    't80'             : {'country': 'Russia',         'role': 'Main Battle Tank',         'speed': '70 km/h',    'weapons': '125mm Smoothbore Gun, Refleks ATGM'},
    't72'             : {'country': 'Russia',         'role': 'Main Battle Tank',         'speed': '60 km/h',    'weapons': '125mm Smoothbore Gun, PKT MG'},
    't64'             : {'country': 'USSR / Ukraine', 'role': 'Main Battle Tank',         'speed': '60 km/h',    'weapons': '125mm Smoothbore Gun'},
    't62'             : {'country': 'USSR / Russia',  'role': 'Main Battle Tank',         'speed': '50 km/h',    'weapons': '115mm Smoothbore Gun'},
    't55'             : {'country': 'USSR',           'role': 'Main Battle Tank',         'speed': '50 km/h',    'weapons': '100mm Rifled Gun'},
    'leopard2'        : {'country': 'Germany',        'role': 'Main Battle Tank',         'speed': '72 km/h',    'weapons': '120mm Smoothbore Gun'},
    'challenger2'     : {'country': 'UK',             'role': 'Main Battle Tank',         'speed': '59 km/h',    'weapons': '120mm Rifled Gun'},
    'merkava'         : {'country': 'Israel',         'role': 'Main Battle Tank',         'speed': '64 km/h',    'weapons': '120mm Smoothbore Gun, Trophy APS'},
    'leclerc'         : {'country': 'France',         'role': 'Main Battle Tank',         'speed': '71 km/h',    'weapons': '120mm Smoothbore Gun, 12.7mm MG'},
    'k2_blackpanther' : {'country': 'South Korea',    'role': 'Main Battle Tank',         'speed': '70 km/h',    'weapons': '120mm Smoothbore Gun, KAPS APS'},
    'type99'          : {'country': 'China',          'role': 'Main Battle Tank',         'speed': '80 km/h',    'weapons': '125mm Smoothbore Gun, HJ-8 ATGM'},
    'type96'          : {'country': 'China',          'role': 'Main Battle Tank',         'speed': '65 km/h',    'weapons': '125mm Smoothbore Gun'},
    'type90_japan'    : {'country': 'Japan',          'role': 'Main Battle Tank',         'speed': '70 km/h',    'weapons': '120mm Smoothbore Gun'},
    'ariete'          : {'country': 'Italy',          'role': 'Main Battle Tank',         'speed': '65 km/h',    'weapons': '120mm Smoothbore Gun, Otobreda MG'},
    'k1_rok'          : {'country': 'South Korea',    'role': 'Main Battle Tank',         'speed': '65 km/h',    'weapons': '105mm / 120mm Rifled Gun'},
    'oplot'           : {'country': 'Ukraine',        'role': 'Main Battle Tank',         'speed': '70 km/h',    'weapons': '125mm KBA3 Gun, Zaslon APS'},
    'arjun'           : {'country': 'India',          'role': 'Main Battle Tank',         'speed': '70 km/h',    'weapons': '120mm Rifled Gun, 12.7mm MG'},
    'pt91_twardy'     : {'country': 'Poland',         'role': 'Main Battle Tank',         'speed': '65 km/h',    'weapons': '125mm Smoothbore Gun'},
    'sabra'           : {'country': 'Israel / Turkey','role': 'Main Battle Tank',         'speed': '55 km/h',    'weapons': '120mm Smoothbore Gun'},
    # ── Naval ─────────────────────────────────────────────────
    'aircraft_carrier': {'country': 'Multi-Nation',   'role': 'Naval Capital Ship',       'speed': '56 km/h',    'weapons': '90 Aircraft, CIWS, Missiles'},
    'destroyer_ddg'   : {'country': 'Multi-Nation',   'role': 'Guided Missile Destroyer', 'speed': '56 km/h',    'weapons': 'Tomahawk, SM-2, MK-45 Gun'},
    'submarine'       : {'country': 'Multi-Nation',   'role': 'Attack Submarine',         'speed': '46 km/h',    'weapons': 'Torpedoes, Cruise Missiles'},
}

# ==============================================================
# DESIGN TOKENS
# ==============================================================
C_BG        = '#111418'
C_SURFACE   = '#181d24'
C_SURFACE2  = '#1e2530'
C_BORDER    = '#2c3848'
C_BORDER_LT = '#3a4d60'
C_ACCENT    = '#3d85b0'
C_ACCENT_HV = '#4d9acc'
C_CONFIRM   = '#4a9e6b'
C_WARN      = '#b8862a'
C_CRITICAL  = '#9b3535'
C_TEXT_PRI  = '#dce5ef'
C_TEXT_SEC  = '#8fa8c0'
C_TEXT_DIM  = '#57728a'
C_TEXT_INV  = '#111418'

SAN  = 'Segoe UI'
MONO = 'Courier New'

def F(family, size, weight='normal'):
    return (family, size, 'bold') if weight == 'bold' else (family, size)

FN_HDR_TITLE  = F(SAN, 13, 'bold')
FN_HDR_SUB    = F(SAN, 10)
FN_HDR_BADGE  = F(SAN, 9,  'bold')
FN_SECTION    = F(SAN, 10, 'bold')
FN_LABEL      = F(SAN, 10)
FN_VALUE      = F(SAN, 10)
FN_VALUE_B    = F(SAN, 10, 'bold')
FN_RESULT     = F(SAN, 17, 'bold')
FN_RESULT_SUB = F(SAN, 10)
FN_CONF_LBL   = F(SAN, 9)
FN_CONF_VAL   = F(SAN, 12, 'bold')
FN_BAR_NAME   = F(SAN, 10)
FN_BAR_PCT    = F(SAN, 10, 'bold')
FN_STATUS     = F(SAN, 9,  'bold')
FN_BTN        = F(SAN, 10, 'bold')
FN_DROP_HEAD  = F(SAN, 11, 'bold')
FN_DROP_HINT  = F(SAN, 10)
FN_LOG        = F(MONO, 9)
FN_FOOTER     = F(SAN, 9)

# ==============================================================
# INFERENCE
# ==============================================================
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize(
            (IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis]          # (1, 3, H, W)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(img_path):
    if not MODEL_LOADED or session is None:
        probs = np.random.dirichlet(np.ones(len(CLASS_NAMES)) * 0.5)
        top5  = probs.argsort()[::-1][:5]
        return [(CLASS_NAMES[i], float(probs[i])) for i in top5]
    inp        = preprocess(img_path)
    input_name = session.get_inputs()[0].name           # reads actual name from ONNX
    raw        = session.run(None, {input_name: inp})[0][0]
    probs      = softmax(raw)
    top5       = probs.argsort()[::-1][:5]
    return [(CLASS_NAMES[i], float(probs[i])) for i in top5]

# ==============================================================
# APP EMBLEM  (generic mark, no external branding)
# ==============================================================
def make_app_emblem(size=52):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    cx = cy = size // 2
    r_outer = size // 2 - 2
    r_inner = r_outer - 7
    d.ellipse([cx-r_outer, cy-r_outer, cx+r_outer, cy+r_outer],
              outline=C_ACCENT, width=2)
    d.ellipse([cx-r_inner, cy-r_inner, cx+r_inner, cy+r_inner],
              outline=C_ACCENT, width=1)
    for i in range(8):
        angle = math.radians(i * 45 - 90)
        x1 = cx + (r_inner + 1) * math.cos(angle)
        y1 = cy + (r_inner + 1) * math.sin(angle)
        x2 = cx + (r_outer - 1) * math.cos(angle)
        y2 = cy + (r_outer - 1) * math.sin(angle)
        d.line([x1, y1, x2, y2], fill=C_ACCENT, width=2)
    aw = 4
    ah = r_inner - 4
    tip_y  = cy - ah
    base_y = cy + ah // 2
    pts = [
        (cx,        tip_y),
        (cx + aw,   cy - 2),
        (cx + aw-2, cy - 2),
        (cx + aw-2, base_y),
        (cx - aw+2, base_y),
        (cx - aw+2, cy - 2),
        (cx - aw,   cy - 2),
    ]
    d.polygon(pts, fill=C_ACCENT)
    return img

def hdivider(parent, color=C_BORDER, thickness=1, padx=0, pady=(0, 0)):
    tk.Frame(parent, bg=color, height=thickness).pack(
        fill='x', padx=padx, pady=pady)

# ==============================================================
# MAIN WINDOW
# ==============================================================
root = tk.Tk()
root.title("MVAC v3.0  —  Military Vehicle AI Classifier")
root.configure(bg=C_BG)
root.geometry('980x860')
root.minsize(900, 800)
root.resizable(True, True)
try:
    root.tk.call('tk', 'scaling', 1.0)
except Exception:
    pass

# ==============================================================
# HEADER
# ==============================================================
header = tk.Frame(root, bg=C_SURFACE2, height=68)
header.pack(fill='x', side='top')
header.pack_propagate(False)

hdr_left = tk.Frame(header, bg=C_SURFACE2)
hdr_left.pack(side='left', padx=20, pady=10)

app_pil    = make_app_emblem(48)
app_photo  = ImageTk.PhotoImage(app_pil)
app_icon   = tk.Label(hdr_left, image=app_photo, bg=C_SURFACE2)
app_icon.image = app_photo
app_icon.pack(side='left', padx=(0, 10))

wordmark = tk.Frame(hdr_left, bg=C_SURFACE2)
wordmark.pack(side='left')
tk.Label(wordmark, text="MVAC v3.0",
         font=FN_HDR_TITLE, fg=C_TEXT_PRI, bg=C_SURFACE2).pack(anchor='w')
tk.Label(wordmark, text="Military Vehicle AI Classifier — Offline Inference",
         font=FN_HDR_SUB, fg=C_TEXT_SEC, bg=C_SURFACE2).pack(anchor='w')

hdr_right = tk.Frame(header, bg=C_SURFACE2)
hdr_right.pack(side='right', padx=20, pady=10)

_ts_var = tk.StringVar()
def _update_clock():
    _ts_var.set(datetime.datetime.now().strftime('%d %b %Y   %H:%M:%S'))
    root.after(1000, _update_clock)

tk.Label(hdr_right, textvariable=_ts_var,
         font=FN_HDR_SUB, fg=C_TEXT_SEC, bg=C_SURFACE2, anchor='e').pack(anchor='e')
tk.Label(hdr_right, text="Vehicle Identification  -  v3.0",
         font=FN_HDR_BADGE, fg=C_TEXT_DIM, bg=C_SURFACE2, anchor='e').pack(anchor='e')

_update_clock()
tk.Frame(root, bg=C_ACCENT, height=2).pack(fill='x')

# ==============================================================
# STATUS BAR
# ==============================================================
status_bar = tk.Frame(root, bg=C_SURFACE2, height=28)
status_bar.pack(fill='x')
status_bar.pack_propagate(False)

def _chip(parent, text, fg, dot=None):
    f = tk.Frame(parent, bg=C_SURFACE2)
    f.pack(side='left', padx=(0, 16))
    if dot:
        tk.Label(f, text='*', font=FN_STATUS, fg=dot,
                 bg=C_SURFACE2).pack(side='left', padx=(0, 4))
    tk.Label(f, text=text, font=FN_STATUS, fg=fg, bg=C_SURFACE2).pack(side='left')

si = tk.Frame(status_bar, bg=C_SURFACE2)
si.pack(side='left', padx=16, pady=5)

acc_display = f'{VAL_ACC*100:.2f}%' if VAL_ACC > 0 else 'N/A'
_chip(si, 'SYSTEM ONLINE',    C_TEXT_SEC, C_CONFIRM)
_chip(si, 'ONNX: ' + ('ACTIVE' if MODEL_LOADED else 'DEMO MODE'),
      C_TEXT_SEC, C_CONFIRM if MODEL_LOADED else C_WARN)
_chip(si, f'{len(CLASS_NAMES)} VEHICLE CLASSES', C_TEXT_DIM)
_chip(si, f'ACCURACY  {acc_display}',            C_TEXT_DIM)
_chip(si, 'OFFLINE MODE',                        C_TEXT_DIM)

hdivider(root, color=C_BORDER)

# ==============================================================
# BODY  (two columns)
# ==============================================================
body = tk.Frame(root, bg=C_BG)
body.pack(fill='both', expand=True, padx=14, pady=10)

left_col  = tk.Frame(body, bg=C_BG)
left_col.pack(side='left', fill='both', expand=True, padx=(0, 7))

right_col = tk.Frame(body, bg=C_BG)
right_col.pack(side='right', fill='both', expand=True, padx=(7, 0))

def section_header(parent, title, right_text=''):
    bar = tk.Frame(parent, bg=C_SURFACE2)
    bar.pack(fill='x')
    tk.Label(bar, text=title,
             font=FN_SECTION, fg=C_TEXT_SEC, bg=C_SURFACE2,
             padx=12, pady=7).pack(side='left')
    if right_text:
        tk.Label(bar, text=right_text,
                 font=FN_CONF_LBL, fg=C_TEXT_DIM, bg=C_SURFACE2,
                 padx=12).pack(side='right')
    hdivider(parent, color=C_BORDER)
    return bar

# ==============================================================
# LEFT — IMAGE INPUT PANEL
# ==============================================================
img_panel = tk.Frame(left_col, bg=C_SURFACE,
                     highlightbackground=C_BORDER, highlightthickness=1)
img_panel.pack(fill='x', pady=(0, 10))
section_header(img_panel, 'Image Input', 'JPG · PNG · BMP · WEBP')

img_canvas = tk.Canvas(img_panel, bg=C_BG, width=440, height=250,
                        highlightthickness=0, cursor='hand2')
img_canvas.pack(fill='x', padx=12, pady=(10, 12))

_img_refs = {}

def render_placeholder():
    img_canvas.delete('all')
    w = img_canvas.winfo_width() or 440
    h = img_canvas.winfo_height() or 250
    pad = 10
    img_canvas.create_rectangle(pad, pad, w-pad, h-pad,
                                  outline=C_BORDER_LT, width=1, dash=(6, 4))
    img_canvas.create_text(w//2, h//2 - 24,
                            text='^', font=(SAN, 26, 'bold'), fill=C_TEXT_DIM)
    img_canvas.create_text(w//2, h//2 + 10,
                            text='Drop image here  or  click  Load Image',
                            font=FN_DROP_HEAD, fill=C_TEXT_SEC)
    img_canvas.create_text(w//2, h//2 + 32,
                            text='Ctrl+O to open   |   Ctrl+V to paste path',
                            font=FN_DROP_HINT, fill=C_TEXT_DIM)

root.after(80, render_placeholder)

def display_image(path):
    img_canvas.delete('all')
    w = img_canvas.winfo_width() or 440
    h = img_canvas.winfo_height() or 250
    img = Image.open(path).convert('RGB')
    iw, ih = img.size
    scale  = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img    = img.resize((nw, nh), Image.LANCZOS)
    photo  = ImageTk.PhotoImage(img)
    _img_refs['main'] = photo
    ox = (w - nw) // 2
    oy = (h - nh) // 2
    img_canvas.create_rectangle(0, 0, w, h, fill=C_BG, outline='')
    img_canvas.create_image(ox, oy, anchor='nw', image=photo)
    img_canvas.create_rectangle(ox, oy, ox+nw, oy+nh,
                                  outline=C_BORDER_LT, width=1)

# ==============================================================
# BUTTONS
# ==============================================================
btn_row = tk.Frame(left_col, bg=C_BG)
btn_row.pack(fill='x', pady=(0, 10))

_last_path = [None]

def load_image(path=None):
    if path is None:
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
    if not path or not os.path.isfile(path):
        return
    _last_path[0] = path
    display_image(path)
    run_prediction(path)

load_btn = tk.Button(btn_row, text='Load Image', command=load_image,
    font=FN_BTN, bg=C_ACCENT, fg=C_TEXT_INV,
    activebackground=C_ACCENT_HV, activeforeground=C_TEXT_INV,
    relief='flat', bd=0, padx=20, pady=9, cursor='hand2')
load_btn.pack(side='left', expand=True, fill='x', padx=(0, 6))
load_btn.bind('<Enter>', lambda e: load_btn.config(bg=C_ACCENT_HV))
load_btn.bind('<Leave>', lambda e: load_btn.config(bg=C_ACCENT))

clear_btn = tk.Button(btn_row, text='Clear',
    command=lambda: (render_placeholder(), clear_results()),
    font=FN_BTN, bg=C_SURFACE2, fg=C_TEXT_SEC,
    activebackground=C_BORDER, activeforeground=C_TEXT_PRI,
    relief='flat', bd=0, padx=20, pady=9, cursor='hand2',
    highlightbackground=C_BORDER, highlightthickness=1)
clear_btn.pack(side='right')

# ==============================================================
# VEHICLE SPECIFICATIONS
# ==============================================================
spec_panel = tk.Frame(left_col, bg=C_SURFACE,
                       highlightbackground=C_BORDER, highlightthickness=1)
spec_panel.pack(fill='both', expand=True)
section_header(spec_panel, 'Vehicle Specifications')

spec_body = tk.Frame(spec_panel, bg=C_SURFACE)
spec_body.pack(fill='both', expand=True)

spec_vars   = {}
spec_fields = [
    ('Designation', 'designation'),
    ('Country',     'country'),
    ('Role',        'role'),
    ('Max Speed',   'speed'),
    ('Armament',    'weapons'),
    ('Confidence',  'confidence'),
]

for idx, (label, key) in enumerate(spec_fields):
    row_bg = C_SURFACE if idx % 2 == 0 else C_SURFACE2
    row = tk.Frame(spec_body, bg=row_bg)
    row.pack(fill='x')
    tk.Label(row, text=label, font=FN_LABEL, fg=C_TEXT_DIM, bg=row_bg,
             width=13, anchor='w', padx=12, pady=7).pack(side='left')
    tk.Frame(row, bg=C_BORDER, width=1).pack(side='left', fill='y', pady=4)
    v = tk.StringVar(value='--')
    spec_vars[key] = v
    if key == 'designation':
        fg, fn = C_TEXT_PRI, FN_VALUE_B
    elif key == 'confidence':
        fg, fn = C_CONFIRM,  FN_VALUE_B
    else:
        fg, fn = C_TEXT_PRI, FN_VALUE
    tk.Label(row, textvariable=v, font=fn, fg=fg, bg=row_bg,
             anchor='w', padx=12).pack(side='left', fill='x', expand=True)

# ==============================================================
# RIGHT — IDENTIFICATION RESULT
# ==============================================================
result_card = tk.Frame(right_col, bg=C_SURFACE,
                        highlightbackground=C_BORDER, highlightthickness=1)
result_card.pack(fill='x', pady=(0, 10))

rc_hdr = tk.Frame(result_card, bg=C_SURFACE2)
rc_hdr.pack(fill='x')
tk.Label(rc_hdr, text='Identification Result',
         font=FN_SECTION, fg=C_TEXT_SEC, bg=C_SURFACE2,
         padx=12, pady=7).pack(side='left')
status_var   = tk.StringVar(value='Standby')
status_badge = tk.Label(rc_hdr, textvariable=status_var,
                         font=FN_STATUS, fg=C_TEXT_DIM, bg=C_SURFACE2,
                         padx=12, pady=7)
status_badge.pack(side='right')
hdivider(result_card, color=C_BORDER)

result_name_var = tk.StringVar(value='Awaiting Input')
tk.Label(result_card, textvariable=result_name_var,
         font=FN_RESULT, fg=C_TEXT_PRI, bg=C_SURFACE,
         pady=14, padx=14, anchor='w').pack(fill='x')

role_var = tk.StringVar(value='')
tk.Label(result_card, textvariable=role_var,
         font=FN_RESULT_SUB, fg=C_TEXT_SEC, bg=C_SURFACE,
         padx=14, pady=0, anchor='w').pack(fill='x')

hdivider(result_card, color=C_BORDER, pady=(8, 0))

conf_frame = tk.Frame(result_card, bg=C_SURFACE)
conf_frame.pack(fill='x', padx=14, pady=10)

conf_hdr = tk.Frame(conf_frame, bg=C_SURFACE)
conf_hdr.pack(fill='x')
tk.Label(conf_hdr, text='Confidence Score',
         font=FN_CONF_LBL, fg=C_TEXT_DIM, bg=C_SURFACE).pack(side='left')
conf_pct_var = tk.StringVar(value='--')
tk.Label(conf_hdr, textvariable=conf_pct_var,
         font=FN_CONF_VAL, fg=C_TEXT_PRI, bg=C_SURFACE).pack(side='right')

conf_track = tk.Canvas(conf_frame, height=8, bg=C_SURFACE2, highlightthickness=0)
conf_track.pack(fill='x', pady=(6, 0))

def update_confidence(prob):
    conf_track.update_idletasks()
    w = conf_track.winfo_width() or 360
    conf_track.delete('all')
    conf_track.create_rectangle(0, 0, w, 8, fill=C_SURFACE2, outline='')
    if prob > 0:
        fill_color = C_CONFIRM if prob > 0.70 else C_WARN if prob > 0.40 else C_CRITICAL
        fill_w = max(0, int(w * prob))
        if fill_w > 0:
            conf_track.create_rectangle(0, 0, fill_w, 8, fill=fill_color, outline='')
    conf_pct_var.set(f'{prob * 100:.1f}%' if prob > 0 else '--')

# ==============================================================
# TOP-5 PROBABILITY BARS
# ==============================================================
top5_panel = tk.Frame(right_col, bg=C_SURFACE,
                       highlightbackground=C_BORDER, highlightthickness=1)
top5_panel.pack(fill='both', expand=True, pady=(0, 10))
section_header(top5_panel, 'Probability Analysis  -  Top 5')

top5_body = tk.Frame(top5_panel, bg=C_SURFACE)
top5_body.pack(fill='both', expand=True, padx=14, pady=10)

bars = []
for i in range(5):
    row_bg = C_SURFACE if i % 2 == 0 else C_SURFACE2
    row = tk.Frame(top5_body, bg=row_bg)
    row.pack(fill='x')
    tk.Label(row, text=str(i+1), font=FN_CONF_LBL, fg=C_TEXT_DIM,
             bg=row_bg, width=2, padx=4, pady=7).pack(side='left')
    tk.Frame(row, bg=C_BORDER, width=1).pack(side='left', fill='y', pady=4)
    nl = tk.Label(row, text='', font=FN_VALUE_B if i==0 else FN_BAR_NAME,
                  fg=C_TEXT_PRI if i==0 else C_TEXT_SEC,
                  bg=row_bg, width=20, anchor='w', padx=8)
    nl.pack(side='left')
    bf = tk.Frame(row, bg=row_bg)
    bf.pack(side='left', fill='x', expand=True, padx=(0, 8))
    bc = tk.Canvas(bf, height=6, bg=C_SURFACE2, highlightthickness=0)
    bc.pack(fill='x', pady=7)
    pl = tk.Label(row, text='', font=FN_BAR_PCT,
                  fg=C_TEXT_PRI if i==0 else C_TEXT_SEC,
                  bg=row_bg, width=7, anchor='e', padx=6)
    pl.pack(side='right')
    bars.append((nl, bc, pl))

# ==============================================================
# EVENT LOG
# ==============================================================
log_panel = tk.Frame(right_col, bg=C_SURFACE,
                      highlightbackground=C_BORDER, highlightthickness=1)
log_panel.pack(fill='x')
section_header(log_panel, 'Event Log')

log_text = tk.Text(log_panel, height=5, font=FN_LOG,
                    bg=C_BG, fg=C_TEXT_SEC, insertbackground=C_ACCENT,
                    relief='flat', bd=0, state='disabled',
                    padx=12, pady=8, selectbackground=C_BORDER)
log_text.pack(fill='x')

# ==============================================================
# FOOTER
# ==============================================================
hdivider(root, color=C_BORDER)
footer = tk.Frame(root, bg=C_SURFACE2, height=28)
footer.pack(fill='x', side='bottom')
footer.pack_propagate(False)

tk.Label(footer,
         text=f'MVAC v3.0  -  Offline  -  {len(CLASS_NAMES)} Classes  -  {acc_display} Val Accuracy',
         font=FN_FOOTER, fg=C_TEXT_DIM, bg=C_SURFACE2).pack(side='left', padx=14)
tk.Label(footer,
         text='Model Active' if MODEL_LOADED else 'Demo Mode',
         font=F(SAN, 9, 'bold'),
         fg=C_CONFIRM if MODEL_LOADED else C_WARN,
         bg=C_SURFACE2).pack(side='right', padx=14)

# ==============================================================
# STATE & LOGIC
# ==============================================================
log_entries = []

def log(msg, level='info'):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    color_map = {'info': C_TEXT_SEC, 'ok': C_CONFIRM, 'warn': C_WARN}
    fg = color_map.get(level, C_TEXT_SEC)
    log_text.config(state='normal')
    tag = f't{len(log_entries)}'
    log_text.insert('end', f'{ts}  {msg}\n', tag)
    log_text.tag_config(tag, foreground=fg)
    log_text.see('end')
    log_text.config(state='disabled')
    log_entries.append(msg)

def clear_results():
    result_name_var.set('Awaiting Input')
    role_var.set('')
    status_var.set('Standby')
    status_badge.config(fg=C_TEXT_DIM)
    update_confidence(0)
    for key in spec_vars:
        spec_vars[key].set('--')
    for (nl, bc, pl) in bars:
        nl.config(text='')
        pl.config(text='')
        bc.delete('all')

def run_prediction(path):
    log(f'Loading: {os.path.basename(path)}')
    status_var.set('Processing...')
    status_badge.config(fg=C_WARN)
    root.update_idletasks()

    results            = predict(path)
    top_name, top_conf = results[0]
    info               = VEHICLE_INFO.get(top_name, {})
    display_name       = top_name.replace('_', ' ').title()

    result_name_var.set(display_name)
    role_var.set(f"{info.get('role', '--')}  -  {info.get('country', '--')}")
    status_var.set('Identified')
    status_badge.config(fg=C_CONFIRM)
    update_confidence(top_conf)

    spec_vars['designation'].set(display_name)
    spec_vars['country'].set(info.get('country',  '--'))
    spec_vars['role'].set(info.get('role',         '--'))
    spec_vars['speed'].set(info.get('speed',       '--'))
    spec_vars['weapons'].set(info.get('weapons',   '--'))
    spec_vars['confidence'].set(f'{top_conf * 100:.2f}%')

    for i, (name, prob) in enumerate(results):
        nl, bc, pl = bars[i]
        nl.config(text=name.replace('_', ' ').title())
        pl.config(text=f'{prob * 100:.1f}%')
        bc.update_idletasks()
        w = bc.winfo_width() or 160
        bc.delete('all')
        bc.create_rectangle(0, 0, w, 6, fill=C_SURFACE2, outline='')
        fill_w = int(w * prob)
        if fill_w > 0:
            bar_col = (C_CONFIRM if i == 0
                       else C_ACCENT if prob > 0.10
                       else C_BORDER)
            bc.create_rectangle(0, 0, fill_w, 6, fill=bar_col, outline='')

    log(f'Identified: {display_name}  ({top_conf*100:.1f}%)', 'ok')
    log(f"Role: {info.get('role','--')}   Country: {info.get('country','--')}")

# ==============================================================
# DRAG AND DROP
# ==============================================================
def handle_drop(raw):
    path = raw.strip().strip('{}').strip('"')
    if os.path.isfile(path):
        _last_path[0] = path
        display_image(path)
        run_prediction(path)

_dnd = False
try:
    import tkinterdnd2 as dnd
    img_canvas.drop_target_register(dnd.DND_FILES)
    img_canvas.dnd_bind('<<Drop>>',      lambda e: handle_drop(e.data))
    img_canvas.dnd_bind('<<DragEnter>>', lambda e: img_canvas.config(bg=C_SURFACE2))
    img_canvas.dnd_bind('<<DragLeave>>', lambda e: img_canvas.config(bg=C_BG))
    _dnd = True
    log('Drag and Drop ready  (tkinterdnd2)', 'ok')
except Exception:
    log('Drag and Drop unavailable  -  install tkinterdnd2', 'warn')

def try_clipboard(event=None):
    try:
        clip = root.clipboard_get().strip()
        if os.path.isfile(clip):
            load_image(clip)
    except Exception:
        pass

img_canvas.bind('<Button-1>', lambda e: load_image())
root.bind('<Control-o>', lambda e: load_image())
root.bind('<Control-v>', try_clipboard)

def on_resize(event):
    if _last_path[0]:
        display_image(_last_path[0])
    else:
        render_placeholder()

img_canvas.bind('<Configure>', on_resize)

# ==============================================================
# STARTUP LOG
# ==============================================================
log('System initialised', 'ok')
log('ONNX Runtime: ' + ('active' if MODEL_LOADED else 'unavailable — demo mode'),
    'ok' if MODEL_LOADED else 'warn')
log(f'Classes loaded: {len(CLASS_NAMES)}')
log(f'Val accuracy: {acc_display}')
log('Ready — load or drop an image to begin')

root.mainloop()
