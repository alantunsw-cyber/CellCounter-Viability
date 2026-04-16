"""
CellCounter-Viability

Author: Dr. Yuan Tian
UNSW Sydney alumnus
"""

import re
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


# =========================================================
# Config
# =========================================================
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Color thresholds
LOWER_BLUE = np.array([95, 50, 40])
UPPER_BLUE = np.array([140, 255, 255])

LOWER_GREEN = np.array([45, 70, 40])
UPPER_GREEN = np.array([95, 255, 255])

# Default thresholds (can be updated by reference inputs)
LIVE_MIN_AREA = 180
LIVE_MAX_AREA = 50000

DEAD_MIN_AREA = 220
DEAD_MAX_AREA = 6000
DEAD_MIN_WIDTH = 16
DEAD_MIN_HEIGHT = 16

DIST_PEAK_RATIO = 0.28

KERNEL_SMALL = np.ones((3, 3), np.uint8)
KERNEL_MED = np.ones((5, 5), np.uint8)

CANVAS_MAX_W = 1100
CANVAS_MAX_H = 780

ZOOM_STEP = 1.15
MIN_SCALE = 0.1
MAX_SCALE = 8.0

LEFT_PANEL_W = 360


# =========================================================
# File / metadata parsing
# =========================================================
def normalize_h2o2_name(text: str) -> str:
    t = text.lower()
    if "h2o2" in t or "h₂o₂" in t:
        return "H2O2"
    return text


def extract_stressor(rel_parts):
    joined = " | ".join(rel_parts).lower()
    if "thapsigargin" in joined:
        return "Thapsigargin"
    if "h2o2" in joined or "h₂o₂" in joined:
        return "H2O2"
    if "nmda" in joined:
        return "NMDA"
    return ""


def extract_date(rel_parts):
    date_patterns = [
        r"\b\d{4}[-_]\d{2}[-_]\d{2}\b",
        r"\b\d{2}[-_]\d{2}[-_]\d{4}\b",
        r"\b\d{8}\b",
    ]
    for part in rel_parts:
        for pat in date_patterns:
            m = re.search(pat, part)
            if m:
                return m.group(0)
    return ""


def extract_concentration(rel_parts, filename=""):
    patterns = [
        r"\b\d+(\.\d+)?\s?(uM|µM|um|mM|nM|%)\b",
        r"\b\d+(\.\d+)?(uM|µM|um|mM|nM|%)\b",
    ]
    search_space = list(rel_parts)
    if filename:
        search_space.append(filename)

    for part in search_space:
        for pat in patterns:
            m = re.search(pat, part, flags=re.IGNORECASE)
            if m:
                return m.group(0).replace("µ", "u")
    return ""


def extract_image_number(filename):
    stem = Path(filename).stem
    patterns = [
        r"_(\d+)_T\d+$",
        r"_(\d+)$",
        r"(\d+)"
    ]
    for pat in patterns:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return stem


def extract_folder_details(img_path: Path, root_dir: Path):
    rel_path = img_path.relative_to(root_dir)
    rel_parts = rel_path.parts[:-1]
    stressor = extract_stressor(rel_parts)
    date = extract_date(rel_parts)
    concentration = extract_concentration(rel_parts, img_path.name)
    folder_details = str(Path(*rel_parts)) if len(rel_parts) > 0 else ""
    return stressor, date, concentration, folder_details


def collect_image_files(root_dir):
    root = Path(root_dir)
    img_files = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "_viability_ui_output" in p.parts:
            continue
        img_files.append(p)

    return sorted(img_files)


# =========================================================
# Detection
# =========================================================
def get_live_mask(hsv):
    blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, KERNEL_SMALL)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, KERNEL_MED)
    return blue_mask


def get_dead_mask(hsv):
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, KERNEL_SMALL)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, KERNEL_SMALL)
    return green_mask


def filter_dead_boxes(mask, dead_min_area, dead_max_area, dead_min_width, dead_min_height):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dead_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < dead_min_area or area > dead_max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < dead_min_width or h < dead_min_height:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.15:
            continue

        dead_boxes.append((x, y, w, h))

    return dead_boxes


def watershed_split_live_cells(image_bgr, blue_mask, live_min_area, live_max_area):
    sure_fg_base = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, KERNEL_SMALL)
    sure_bg = cv2.dilate(blue_mask, KERNEL_SMALL, iterations=2)

    dist = cv2.distanceTransform(sure_fg_base, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    _, sure_fg = cv2.threshold(dist, DIST_PEAK_RATIO * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    ws_img = image_bgr.copy()
    markers = cv2.watershed(ws_img, markers)

    boxes = []
    for label in np.unique(markers):
        if label <= 1:
            continue

        region = np.uint8(markers == label) * 255
        area = cv2.countNonZero(region)
        if area < live_min_area or area > live_max_area:
            continue

        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 12 or h < 12:
            continue

        boxes.append((x, y, w, h))

    return boxes


# =========================================================
# App
# =========================================================
class ViabilityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CellCounter-Viability")

        self.project_dir = None
        self.output_dir = None
        self.image_files = []
        self.current_index = -1

        self.current_bgr = None
        self.current_rgb = None
        self.tk_img = None
        self.img_h = 0
        self.img_w = 0

        self.scale = 1.0
        self.base_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.live_boxes = []
        self.dead_boxes = []
        self.selected_kind = None
        self.selected_idx = None

        self.temp_start = None
        self.temp_rect_id = None
        self.temp_button = None

        self.annotations = {}

        self.shift_held = False

        # reference setup
        self.setting_reference = False
        self.reference_stage = None

        # pending reference confirmation state
        self.ref_dirty = False

        # reference entry vars
        self.live_ref_w_var = tk.StringVar(value="0")
        self.live_ref_h_var = tk.StringVar(value="0")
        self.live_ref_area_var = tk.StringVar(value="0")

        self.dead_ref_w_var = tk.StringVar(value="0")
        self.dead_ref_h_var = tk.StringVar(value="0")
        self.dead_ref_area_var = tk.StringVar(value="0")

        # current thresholds used by detection
        self.live_min_area = LIVE_MIN_AREA
        self.live_max_area = LIVE_MAX_AREA
        self.dead_min_area = DEAD_MIN_AREA
        self.dead_max_area = DEAD_MAX_AREA
        self.dead_min_width = DEAD_MIN_WIDTH
        self.dead_min_height = DEAD_MIN_HEIGHT

        self.build_ui()

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.open_btn = tk.Button(top, text="Open Folder", command=self.open_folder, width=11)
        self.open_btn.pack(side=tk.LEFT, padx=3)

        self.set_ref_btn = tk.Button(top, text="Set Ref", command=self.start_reference_setup, width=9)
        self.set_ref_btn.pack(side=tk.LEFT, padx=3)

        self.prev_btn = tk.Button(top, text="Prev", command=self.prev_image, width=7)
        self.prev_btn.pack(side=tk.LEFT, padx=3)

        self.next_btn = tk.Button(top, text="Next", command=self.next_image, width=7)
        self.next_btn.pack(side=tk.LEFT, padx=3)

        self.scan_live_btn = tk.Button(top, text="Scan Live", command=self.run_auto_detect_live_current, width=11)
        self.scan_live_btn.pack(side=tk.LEFT, padx=3)

        self.scan_dead_btn = tk.Button(top, text="Scan Dead", command=self.run_auto_detect_dead_current, width=11)
        self.scan_dead_btn.pack(side=tk.LEFT, padx=3)

        self.clean_all_btn = tk.Button(top, text="Clean All", command=self.clear_all_boxes, width=11)
        self.clean_all_btn.pack(side=tk.LEFT, padx=3)

        self.save_btn = tk.Button(top, text="Save Current", command=self.save_current_annotation, width=11)
        self.save_btn.pack(side=tk.LEFT, padx=3)

        self.export_btn = tk.Button(top, text="Export Report", command=self.export_report, width=11)
        self.export_btn.pack(side=tk.LEFT, padx=3)

        self.info_var = tk.StringVar(value="No folder loaded.")
        tk.Label(self.root, textvariable=self.info_var, anchor="w").pack(side=tk.TOP, fill=tk.X, padx=8)

        self.count_var = tk.StringVar(value="Live: 0 | Dead: 0")
        tk.Label(self.root, textvariable=self.count_var, anchor="w").pack(side=tk.TOP, fill=tk.X, padx=8)

        main_area = tk.Frame(self.root)
        main_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        left_panel = tk.Frame(main_area, width=LEFT_PANEL_W)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left_panel.pack_propagate(False)

        help_text = (
            "Mouse\n"
            "------\n"
            "Scroll Up/Down   : Zoom in/out\n"
            "Left Drag        : Draw LIVE\n"
            "Ctrl + Left Drag : Draw DEAD\n"
            "Right Click Box  : Delete box\n\n"
            "Keyboard\n"
            "--------\n"
            "A / ←            : Previous\n"
            "D / →            : Next\n"
            "Q                : Scan/Clear LIVE\n"
            "E                : Scan/Clear DEAD\n"
            "R                : Set references\n"
            "Shift (hold)     : Hide boxes\n"
            "Delete/Backspace : Delete box\n"
            "S                : Save\n"
        )

        help_label = tk.Label(
            left_panel,
            text=help_text,
            anchor="nw",
            justify="left",
            bg="white",
            relief="solid",
            bd=1,
            padx=10,
            pady=10,
            font=("Consolas", 11)
        )
        help_label.pack(fill=tk.X, expand=False)

        ref_panel = tk.Frame(left_panel, bg="#f5f5f5", relief="solid", bd=1)
        ref_panel.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=(8, 0))

        tk.Label(ref_panel, text="References", bg="#f5f5f5", font=("Consolas", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(8, 4)
        )

        self.ref_status_var = tk.StringVar(value="Reference confirmed")
        self.ref_status_label = tk.Label(
            ref_panel, textvariable=self.ref_status_var, bg="#f5f5f5",
            fg="green", font=("Consolas", 10, "bold")
        )
        self.ref_status_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))

        tk.Label(ref_panel, text="Live W", bg="#f5f5f5", font=("Consolas", 10)).grid(row=2, column=0, sticky="w", padx=10)
        self.live_w_entry = tk.Entry(ref_panel, textvariable=self.live_ref_w_var, width=8)
        self.live_w_entry.grid(row=2, column=1, sticky="w", padx=4)

        tk.Label(ref_panel, text="Live H", bg="#f5f5f5", font=("Consolas", 10)).grid(row=3, column=0, sticky="w", padx=10)
        self.live_h_entry = tk.Entry(ref_panel, textvariable=self.live_ref_h_var, width=8)
        self.live_h_entry.grid(row=3, column=1, sticky="w", padx=4)

        tk.Label(ref_panel, text="Live Area", bg="#f5f5f5", font=("Consolas", 10)).grid(row=4, column=0, sticky="w", padx=10)
        tk.Label(ref_panel, textvariable=self.live_ref_area_var, bg="#f5f5f5", font=("Consolas", 10)).grid(row=4, column=1, sticky="w", padx=4)

        tk.Label(ref_panel, text="Dead W", bg="#f5f5f5", font=("Consolas", 10)).grid(row=5, column=0, sticky="w", padx=10, pady=(8, 0))
        self.dead_w_entry = tk.Entry(ref_panel, textvariable=self.dead_ref_w_var, width=8)
        self.dead_w_entry.grid(row=5, column=1, sticky="w", padx=4, pady=(8, 0))

        tk.Label(ref_panel, text="Dead H", bg="#f5f5f5", font=("Consolas", 10)).grid(row=6, column=0, sticky="w", padx=10)
        self.dead_h_entry = tk.Entry(ref_panel, textvariable=self.dead_ref_h_var, width=8)
        self.dead_h_entry.grid(row=6, column=1, sticky="w", padx=4)

        tk.Label(ref_panel, text="Dead Area", bg="#f5f5f5", font=("Consolas", 10)).grid(row=7, column=0, sticky="w", padx=10)
        tk.Label(ref_panel, textvariable=self.dead_ref_area_var, bg="#f5f5f5", font=("Consolas", 10)).grid(row=7, column=1, sticky="w", padx=4)

        self.confirm_ref_btn = tk.Button(ref_panel, text="Confirm Ref", command=self.apply_reference_values, width=12)
        self.confirm_ref_btn.grid(row=8, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 10))

        # mark dirty when entry values change
        self.live_ref_w_var.trace_add("write", lambda *args: self.on_reference_entry_changed())
        self.live_ref_h_var.trace_add("write", lambda *args: self.on_reference_entry_changed())
        self.dead_ref_w_var.trace_add("write", lambda *args: self.on_reference_entry_changed())
        self.dead_ref_h_var.trace_add("write", lambda *args: self.on_reference_entry_changed())

        right_panel = tk.Frame(main_area)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_panel, width=CANVAS_MAX_W, height=CANVAS_MAX_H, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)

        self.canvas.bind("<Control-ButtonPress-1>", self.on_ctrl_left_down)
        self.canvas.bind("<Control-B1-Motion>", self.on_ctrl_left_drag)
        self.canvas.bind("<Control-ButtonRelease-1>", self.on_ctrl_left_up)

        self.canvas.bind("<Button-3>", self.on_right_click_delete)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel_linux_up)
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux_down)

        self.root.bind("<Delete>", lambda e: self.delete_selected())
        self.root.bind("<BackSpace>", lambda e: self.delete_selected())
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<a>", lambda e: self.prev_image())
        self.root.bind("<A>", lambda e: self.prev_image())
        self.root.bind("<d>", lambda e: self.next_image())
        self.root.bind("<D>", lambda e: self.next_image())
        self.root.bind("<q>", lambda e: self.toggle_live_scan_clear())
        self.root.bind("<Q>", lambda e: self.toggle_live_scan_clear())
        self.root.bind("<e>", lambda e: self.toggle_dead_scan_clear())
        self.root.bind("<E>", lambda e: self.toggle_dead_scan_clear())
        self.root.bind("<r>", lambda e: self.start_reference_setup())
        self.root.bind("<R>", lambda e: self.start_reference_setup())
        self.root.bind("s", lambda e: self.save_current_annotation())

        self.root.bind("<KeyPress-Shift_L>", self.on_shift_press)
        self.root.bind("<KeyPress-Shift_R>", self.on_shift_press)
        self.root.bind("<KeyRelease-Shift_L>", self.on_shift_release)
        self.root.bind("<KeyRelease-Shift_R>", self.on_shift_release)

        self.set_ref_dirty(False)

    # -------------------------
    # Reference state helpers
    # -------------------------
    def on_reference_entry_changed(self):
        self.preview_reference_areas()
        self.set_ref_dirty(True)

    def preview_reference_areas(self):
        try:
            lw = max(1, int(self.live_ref_w_var.get()))
            lh = max(1, int(self.live_ref_h_var.get()))
            self.live_ref_area_var.set(str(lw * lh))
        except ValueError:
            pass

        try:
            dw = max(1, int(self.dead_ref_w_var.get()))
            dh = max(1, int(self.dead_ref_h_var.get()))
            self.dead_ref_area_var.set(str(dw * dh))
        except ValueError:
            pass

    def set_ref_dirty(self, is_dirty):
        self.ref_dirty = is_dirty

        if self.ref_dirty:
            self.ref_status_var.set("Reference changed - confirm required")
            self.ref_status_label.config(fg="red")
            self.confirm_ref_btn.config(text="Confirm Ref *", bg="#ffd966", activebackground="#ffd966")
        else:
            self.ref_status_var.set("Reference confirmed")
            self.ref_status_label.config(fg="green")
            self.confirm_ref_btn.config(text="Confirm Ref", bg=self.root.cget("bg"), activebackground=self.root.cget("bg"))

        # disable most controls until confirmed
        state = tk.DISABLED if self.ref_dirty else tk.NORMAL
        for btn in [self.set_ref_btn, self.prev_btn, self.next_btn, self.scan_live_btn,
                    self.scan_dead_btn, self.clean_all_btn, self.save_btn, self.export_btn]:
            btn.config(state=state)

    def require_confirm_guard(self):
        if self.ref_dirty:
            messagebox.showwarning("Confirm Reference", "Please click 'Confirm Ref' before continuing.")
            return True
        return False

    def release_entry_focus(self):
        self.canvas.focus_set()

    # -------------------------
    # Reference setup
    # -------------------------
    def start_reference_setup(self):
        if self.current_rgb is None:
            if self.image_files:
                self.load_current_image()
            else:
                messagebox.showinfo("Reference setup", "Please open a folder first.")
                return

        if self.ref_dirty:
            messagebox.showwarning("Confirm Reference", "Please confirm the current reference values first.")
            return

        self.setting_reference = True
        self.reference_stage = "live"
        messagebox.showinfo(
            "Set references",
            "Step 1: Draw a box around the MINIMUM LIVE cell on the current image.\n\n"
            "Use LEFT DRAG."
        )

    def finish_reference_from_box(self, box, stage):
        _, _, w, h = box

        if stage == "live":
            self.live_ref_w_var.set(str(w))
            self.live_ref_h_var.set(str(h))
            self.preview_reference_areas()

            self.reference_stage = "dead"
            messagebox.showinfo(
                "Set references",
                "Step 2: Draw a box around the MINIMUM DEAD cell on the current image.\n\n"
                "Use LEFT DRAG."
            )
            return

        if stage == "dead":
            self.dead_ref_w_var.set(str(w))
            self.dead_ref_h_var.set(str(h))
            self.preview_reference_areas()

            self.setting_reference = False
            self.reference_stage = None
            self.set_ref_dirty(True)

            messagebox.showinfo(
                "Set references",
                "Reference values are ready.\n\nPlease click 'Confirm Ref' to apply them."
            )

    def update_live_reference_from_entries(self):
        w = max(1, int(self.live_ref_w_var.get()))
        h = max(1, int(self.live_ref_h_var.get()))
        area = w * h
        self.live_ref_area_var.set(str(area))
        self.live_min_area = max(20, int(area * 0.45))

    def update_dead_reference_from_entries(self):
        w = max(1, int(self.dead_ref_w_var.get()))
        h = max(1, int(self.dead_ref_h_var.get()))
        area = w * h
        self.dead_ref_area_var.set(str(area))
        self.dead_min_area = max(20, int(area * 0.45))
        self.dead_min_width = max(4, int(w * 0.6))
        self.dead_min_height = max(4, int(h * 0.6))

    def apply_reference_values(self):
        try:
            self.update_live_reference_from_entries()
            self.update_dead_reference_from_entries()
        except ValueError:
            messagebox.showerror("Invalid Reference", "Reference width and height must be valid positive integers.")
            return

        self.update_info()
        self.release_entry_focus()
        self.set_ref_dirty(False)

    # -------------------------
    # Shift hide
    # -------------------------
    def on_shift_press(self, event=None):
        if not self.shift_held:
            self.shift_held = True
            self.refresh_canvas()

    def on_shift_release(self, event=None):
        if self.shift_held:
            self.shift_held = False
            self.refresh_canvas()

    # -------------------------
    # File open / save
    # -------------------------
    def open_folder(self):
        if self.ref_dirty:
            messagebox.showwarning("Confirm Reference", "Please confirm the current reference values first.")
            return

        folder = filedialog.askdirectory(title="Select project folder")
        if not folder:
            return

        self.project_dir = Path(folder)
        self.output_dir = self.project_dir / "_viability_ui_output"
        self.output_dir.mkdir(exist_ok=True)

        self.image_files = collect_image_files(folder)
        if not self.image_files:
            messagebox.showwarning("No images", "No supported image files found.")
            return

        self.load_annotations_json()
        self.current_index = 0
        self.load_current_image()
        self.start_reference_setup()

    def load_annotations_json(self):
        json_path = self.output_dir / "annotations.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.annotations = json.load(f)
            except Exception:
                self.annotations = {}
        else:
            self.annotations = {}

    def save_annotations_json(self):
        json_path = self.output_dir / "annotations.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=2)

    def auto_save_current_full(self):
        if not self.image_files or self.current_index < 0:
            return

        img_path = self.image_files[self.current_index]
        key = str(img_path)

        self.annotations[key] = {
            "live_boxes": [list(b) for b in self.live_boxes],
            "dead_boxes": [list(b) for b in self.dead_boxes],
        }
        self.save_annotations_json()

        preview_dir = self.output_dir / "annotated_preview"
        preview_dir.mkdir(exist_ok=True)

        annotated = self.make_annotated_image()
        rel_path = img_path.relative_to(self.project_dir)
        safe_name = "__".join(rel_path.with_suffix("").parts) + "_reviewed.png"

        out_path = preview_dir / safe_name
        cv2.imwrite(str(out_path), annotated)

    # -------------------------
    # Image loading
    # -------------------------
    def load_current_image(self):
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return

        img_path = self.image_files[self.current_index]
        img = cv2.imread(str(img_path))
        if img is None:
            messagebox.showerror("Read error", f"Cannot read image:\n{img_path}")
            return

        self.current_bgr = img
        self.current_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w = self.current_rgb.shape[:2]

        self.base_scale = min(CANVAS_MAX_W / self.img_w, CANVAS_MAX_H / self.img_h, 1.0)
        self.scale = self.base_scale
        self.recenter_image()

        key = str(img_path)
        if key in self.annotations:
            self.live_boxes = [tuple(b) for b in self.annotations[key].get("live_boxes", [])]
            self.dead_boxes = [tuple(b) for b in self.annotations[key].get("dead_boxes", [])]
        else:
            self.live_boxes = self.auto_detect_live_cells_current()
            self.dead_boxes = self.auto_detect_dead_cells_current()

        self.selected_kind = None
        self.selected_idx = None
        self.refresh_canvas()
        self.update_info()

    def recenter_image(self):
        new_w = int(self.img_w * self.scale)
        new_h = int(self.img_h * self.scale)
        self.offset_x = (CANVAS_MAX_W - new_w) // 2
        self.offset_y = (CANVAS_MAX_H - new_h) // 2

    def update_info(self):
        if not self.image_files:
            self.info_var.set("No folder loaded.")
            self.count_var.set("Live: 0 | Dead: 0")
            return

        img_path = self.image_files[self.current_index]
        rel = img_path.relative_to(self.project_dir)
        zoom_pct = int(self.scale / self.base_scale * 100) if self.base_scale > 0 else 100

        ref_stage_text = ""
        if self.setting_reference:
            ref_stage_text = f" | Setting ref: {self.reference_stage}"
        elif self.ref_dirty:
            ref_stage_text = " | Ref pending confirmation"

        self.info_var.set(
            f"[{self.current_index + 1}/{len(self.image_files)}] {rel} | Zoom: {zoom_pct}%{ref_stage_text}"
        )
        self.count_var.set(f"Live: {len(self.live_boxes)} | Dead: {len(self.dead_boxes)}")

    # -------------------------
    # Detection wrappers with current thresholds
    # -------------------------
    def auto_detect_live_cells_current(self):
        hsv = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2HSV)
        live_mask = get_live_mask(hsv)
        return watershed_split_live_cells(
            self.current_bgr, live_mask,
            self.live_min_area, self.live_max_area
        )

    def auto_detect_dead_cells_current(self):
        hsv = cv2.cvtColor(self.current_bgr, cv2.COLOR_BGR2HSV)
        dead_mask = get_dead_mask(hsv)
        return filter_dead_boxes(
            dead_mask,
            self.dead_min_area,
            self.dead_max_area,
            self.dead_min_width,
            self.dead_min_height
        )

    # -------------------------
    # Zoom
    # -------------------------
    def zoom_at_canvas_point(self, factor, canvas_x, canvas_y):
        if self.current_rgb is None:
            return

        img_x_before, img_y_before = self.canvas_to_img(canvas_x, canvas_y)

        new_scale = self.scale * factor
        new_scale = max(MIN_SCALE, min(MAX_SCALE, new_scale))
        if abs(new_scale - self.scale) < 1e-9:
            return

        self.scale = new_scale
        self.offset_x = int(canvas_x - img_x_before * self.scale)
        self.offset_y = int(canvas_y - img_y_before * self.scale)

        self.refresh_canvas()
        self.update_info()

    def on_mousewheel(self, event):
        if self.current_rgb is None:
            return
        if event.delta > 0:
            self.zoom_at_canvas_point(ZOOM_STEP, event.x, event.y)
        elif event.delta < 0:
            self.zoom_at_canvas_point(1.0 / ZOOM_STEP, event.x, event.y)

    def on_mousewheel_linux_up(self, event):
        self.zoom_at_canvas_point(ZOOM_STEP, event.x, event.y)

    def on_mousewheel_linux_down(self, event):
        self.zoom_at_canvas_point(1.0 / ZOOM_STEP, event.x, event.y)

    # -------------------------
    # Coordinate transform
    # -------------------------
    def img_to_canvas(self, x, y):
        cx = int(x * self.scale) + self.offset_x
        cy = int(y * self.scale) + self.offset_y
        return cx, cy

    def canvas_to_img(self, cx, cy):
        x = (cx - self.offset_x) / self.scale
        y = (cy - self.offset_y) / self.scale

        x = int(round(x))
        y = int(round(y))

        x = max(0, min(self.img_w - 1, x))
        y = max(0, min(self.img_h - 1, y))
        return x, y

    # -------------------------
    # Drawing
    # -------------------------
    def refresh_canvas(self):
        self.canvas.delete("all")
        if self.current_rgb is None:
            return

        new_w = max(1, int(self.img_w * self.scale))
        new_h = max(1, int(self.img_h * self.scale))
        resized = cv2.resize(self.current_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(resized)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_img)

        if not self.shift_held:
            for i, box in enumerate(self.live_boxes):
                self.draw_box(box, "blue", "Live", selected=(self.selected_kind == "live" and self.selected_idx == i))

            for i, box in enumerate(self.dead_boxes):
                self.draw_box(box, "lime", "Dead", selected=(self.selected_kind == "dead" and self.selected_idx == i))

    def draw_box(self, box, color, label, selected=False):
        x, y, w, h = box
        x1, y1 = self.img_to_canvas(x, y)
        x2, y2 = self.img_to_canvas(x + w, y + h)

        line_w = 3 if selected else 2
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=line_w)
        self.canvas.create_text(x1 + 3, y1 - 8, text=label, fill=color, anchor="sw", font=("Arial", 11, "bold"))

    # -------------------------
    # Box hit test
    # -------------------------
    def find_box_at(self, img_x, img_y):
        for i in range(len(self.live_boxes) - 1, -1, -1):
            x, y, w, h = self.live_boxes[i]
            if x <= img_x <= x + w and y <= img_y <= y + h:
                return "live", i

        for i in range(len(self.dead_boxes) - 1, -1, -1):
            x, y, w, h = self.dead_boxes[i]
            if x <= img_x <= x + w and y <= img_y <= y + h:
                return "dead", i

        return None, None

    # -------------------------
    # Mouse events
    # -------------------------
    def on_left_down(self, event):
        if self.current_rgb is None:
            return
        if self.ref_dirty and not self.setting_reference:
            return

        self.temp_button = "left"
        self.temp_start = (event.x, event.y)
        self.temp_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="yellow", width=2, dash=(4, 2)
        )

    def on_left_drag(self, event):
        if self.temp_rect_id is not None and self.temp_start is not None and self.temp_button == "left":
            x0, y0 = self.temp_start
            self.canvas.coords(self.temp_rect_id, x0, y0, event.x, event.y)

    def on_left_up(self, event):
        if self.temp_rect_id is None or self.temp_start is None or self.temp_button != "left":
            self.temp_start = None
            self.temp_button = None
            return

        x0, y0 = self.temp_start
        x1, y1 = event.x, event.y

        self.canvas.delete(self.temp_rect_id)
        self.temp_rect_id = None
        self.temp_start = None
        self.temp_button = None

        ix0, iy0 = self.canvas_to_img(min(x0, x1), min(y0, y1))
        ix1, iy1 = self.canvas_to_img(max(x0, x1), max(y0, y1))

        w = ix1 - ix0
        h = iy1 - iy0
        if w < 6 or h < 6:
            return

        box = (ix0, iy0, w, h)

        if self.setting_reference:
            self.finish_reference_from_box(box, self.reference_stage)
            self.refresh_canvas()
            self.update_info()
            return

        if self.ref_dirty:
            return

        self.live_boxes.append(box)
        self.selected_kind = "live"
        self.selected_idx = len(self.live_boxes) - 1
        self.refresh_canvas()
        self.update_info()

    def on_ctrl_left_down(self, event):
        if self.current_rgb is None or self.setting_reference or self.ref_dirty:
            return

        self.temp_button = "ctrl_left"
        self.temp_start = (event.x, event.y)
        self.temp_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="yellow", width=2, dash=(4, 2)
        )

    def on_ctrl_left_drag(self, event):
        if self.temp_rect_id is not None and self.temp_start is not None and self.temp_button == "ctrl_left":
            x0, y0 = self.temp_start
            self.canvas.coords(self.temp_rect_id, x0, y0, event.x, event.y)

    def on_ctrl_left_up(self, event):
        if self.temp_rect_id is None or self.temp_start is None or self.temp_button != "ctrl_left":
            self.temp_start = None
            self.temp_button = None
            return

        x0, y0 = self.temp_start
        x1, y1 = event.x, event.y

        self.canvas.delete(self.temp_rect_id)
        self.temp_rect_id = None
        self.temp_start = None
        self.temp_button = None

        ix0, iy0 = self.canvas_to_img(min(x0, x1), min(y0, y1))
        ix1, iy1 = self.canvas_to_img(max(x0, x1), max(y0, y1))

        w = ix1 - ix0
        h = iy1 - iy0
        if w < 6 or h < 6:
            return

        self.dead_boxes.append((ix0, iy0, w, h))
        self.selected_kind = "dead"
        self.selected_idx = len(self.dead_boxes) - 1
        self.refresh_canvas()
        self.update_info()

    def on_right_click_delete(self, event):
        if self.current_rgb is None or self.setting_reference or self.ref_dirty:
            return

        img_x, img_y = self.canvas_to_img(event.x, event.y)
        kind, idx = self.find_box_at(img_x, img_y)

        if kind == "live" and idx is not None:
            del self.live_boxes[idx]
        elif kind == "dead" and idx is not None:
            del self.dead_boxes[idx]
        else:
            return

        self.selected_kind = None
        self.selected_idx = None
        self.refresh_canvas()
        self.update_info()
        self.auto_save_current_full()

    # -------------------------
    # Edit operations
    # -------------------------
    def delete_selected(self):
        if self.ref_dirty:
            self.require_confirm_guard()
            return

        if self.selected_kind == "live" and self.selected_idx is not None:
            if 0 <= self.selected_idx < len(self.live_boxes):
                del self.live_boxes[self.selected_idx]
        elif self.selected_kind == "dead" and self.selected_idx is not None:
            if 0 <= self.selected_idx < len(self.dead_boxes):
                del self.dead_boxes[self.selected_idx]

        self.selected_kind = None
        self.selected_idx = None
        self.refresh_canvas()
        self.update_info()

    def run_auto_detect_live_current(self):
        if self.current_bgr is None:
            return
        if self.require_confirm_guard():
            return
        self.live_boxes = self.auto_detect_live_cells_current()
        self.selected_kind = None
        self.selected_idx = None
        self.refresh_canvas()
        self.update_info()

    def run_auto_detect_dead_current(self):
        if self.current_bgr is None:
            return
        if self.require_confirm_guard():
            return
        self.dead_boxes = self.auto_detect_dead_cells_current()
        self.selected_kind = None
        self.selected_idx = None
        self.refresh_canvas()
        self.update_info()

    def run_auto_detect_both_current(self):
        if self.current_bgr is None:
            return
        if self.require_confirm_guard():
            return
        self.live_boxes = self.auto_detect_live_cells_current()
        self.dead_boxes = self.auto_detect_dead_cells_current()
        self.selected_kind = None
        self.selected_idx = None
        self.refresh_canvas()
        self.update_info()

    def clear_all_boxes(self):
        if self.current_bgr is None:
            return
        if self.require_confirm_guard():
            return
        self.live_boxes = []
        self.dead_boxes = []
        self.selected_kind = None
        self.selected_idx = None
        self.refresh_canvas()
        self.update_info()
        self.auto_save_current_full()

    def clear_live_boxes(self):
        if self.current_bgr is None:
            return
        if self.require_confirm_guard():
            return
        self.live_boxes = []
        if self.selected_kind == "live":
            self.selected_kind = None
            self.selected_idx = None
        self.refresh_canvas()
        self.update_info()
        self.auto_save_current_full()

    def clear_dead_boxes(self):
        if self.current_bgr is None:
            return
        if self.require_confirm_guard():
            return
        self.dead_boxes = []
        if self.selected_kind == "dead":
            self.selected_kind = None
            self.selected_idx = None
        self.refresh_canvas()
        self.update_info()
        self.auto_save_current_full()

    def toggle_live_scan_clear(self):
        if self.current_bgr is None:
            return
        if self.require_confirm_guard():
            return
        if len(self.live_boxes) == 0:
            self.run_auto_detect_live_current()
        else:
            self.clear_live_boxes()

    def toggle_dead_scan_clear(self):
        if self.current_bgr is None:
            return
        if self.require_confirm_guard():
            return
        if len(self.dead_boxes) == 0:
            self.run_auto_detect_dead_current()
        else:
            self.clear_dead_boxes()

    # -------------------------
    # Save current / preview
    # -------------------------
    def save_current_annotation(self):
        if not self.image_files or self.current_index < 0:
            return
        if self.require_confirm_guard():
            return
        self.auto_save_current_full()
        img_path = self.image_files[self.current_index]
        messagebox.showinfo("Saved", f"Saved current annotation:\n{img_path.name}")

    def save_current_silent(self):
        if not self.image_files or self.current_index < 0:
            return
        if self.ref_dirty:
            return
        self.auto_save_current_full()

    def make_annotated_image(self):
        out = self.current_bgr.copy()

        for x, y, w, h in self.live_boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(out, "Live", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)

        for x, y, w, h in self.dead_boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(out, "Dead", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        return out

    # -------------------------
    # Navigation
    # -------------------------
    def prev_image(self):
        if not self.image_files:
            return
        if self.require_confirm_guard():
            return
        self.auto_save_current_full()
        self.current_index = max(0, self.current_index - 1)
        self.load_current_image()

    def next_image(self):
        if not self.image_files:
            return
        if self.require_confirm_guard():
            return
        self.auto_save_current_full()
        self.current_index = min(len(self.image_files) - 1, self.current_index + 1)
        self.load_current_image()

    # -------------------------
    # Export
    # -------------------------
    def export_report(self):
        if not self.project_dir or not self.image_files:
            messagebox.showwarning("No data", "Please open a folder first.")
            return
        if self.require_confirm_guard():
            return

        self.auto_save_current_full()

        rows = []
        for img_path in self.image_files:
            key = str(img_path)
            ann = self.annotations.get(key, {"live_boxes": [], "dead_boxes": []})

            stressor, date, concentration, folder_details = extract_folder_details(img_path, self.project_dir)
            image_number = extract_image_number(img_path.name)

            live_count = len(ann.get("live_boxes", []))
            dead_count = len(ann.get("dead_boxes", []))
            total = live_count + dead_count
            viability = 100.0 * live_count / total if total > 0 else np.nan

            rows.append({
                "folder_details": folder_details,
                "stressor": normalize_h2o2_name(stressor),
                "date": date,
                "concentration": concentration,
                "image_name": img_path.name,
                "image_number": image_number,
                "live_cells": live_count,
                "dead_cells": dead_count,
                "total_cells": total,
                "viability_percent": viability,
                "full_path": str(img_path),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            messagebox.showwarning("No results", "No results to export.")
            return

        df = df.sort_values(["stressor", "date", "concentration", "image_name"]).reset_index(drop=True)

        per_image_path = self.output_dir / "viability_counts_reviewed.xlsx"
        summary_path = self.output_dir / "viability_summary_reviewed.xlsx"

        df.to_excel(per_image_path, index=False)

        summary = (
            df.groupby(["stressor", "date", "concentration"], dropna=False)
            .agg(
                n_images=("image_name", "count"),
                live_cells_sum=("live_cells", "sum"),
                dead_cells_sum=("dead_cells", "sum"),
                total_cells_sum=("total_cells", "sum"),
                viability_mean_percent=("viability_percent", "mean"),
            )
            .reset_index()
        )
        summary.to_excel(summary_path, index=False)

        messagebox.showinfo(
            "Exported",
            f"Report exported.\n\nPer-image:\n{per_image_path}\n\nSummary:\n{summary_path}"
        )


def main():
    root = tk.Tk()
    app = ViabilityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
