import streamlit as st
import pandas as pd
import zipfile
import os
import io
import re
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import fitz  # PyMuPDF

# ==============================
# 0. 기본 설정 & 폰트 체크
# ==============================
st.set_page_config(page_title="SAT MATH", layout="centered")

FONT_REGULAR = "fonts/NanumGothic.ttf"
FONT_BOLD = "fonts/NanumGothicBold.ttf"
pdf_font_name = "NanumGothic"

font_ready = os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD)

if not font_ready:
    st.error("⚠️ 한글 출력을 위해 fonts 폴더에 NanumGothic.ttf / NanumGothicBold.ttf가 필요합니다.")

# Tab1용 PDF
if font_ready:
    class KoreanPDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_margins(25.4, 30, 25.4)
            self.set_auto_page_break(auto=True, margin=25.4)
            self.add_font(pdf_font_name, "", FONT_REGULAR, uni=True)
            self.add_font(pdf_font_name, "B", FONT_BOLD, uni=True)
            self.set_font(pdf_font_name, size=10)

# =========================================================
# [전역] Tab 1/2/3 공용 함수
# =========================================================
def _clean(x):
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).replace("\r", "").strip()

def parse_wrong_list(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return set()
    s = str(val).strip()
    if s == "" or s.upper() in ["X", "Х", "-"]:
        return set()
    s = s.replace("，", ",").replace(";", ",")
    nums = [t.strip() for t in s.split(",") if t.strip()]
    out = set()
    for n in nums:
        try:
            out.add(int(float(n)))
        except:
            pass
    return out

def score_to_slash22(s):
    s = _clean(s)
    if s == "":
        return ""
    if "/" in s:
        return s
    return f"{s} / 22"

def assert_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"⚠️ {label} 컬럼 누락: {missing}")
        st.write(f"현재 {label} 컬럼:", list(df.columns))
        st.stop()

# =========================================================
# [Tab 1] 오답노트 생성기
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def keyify(s: str) -> str:
        return (
            s.replace("\u3000", " ")
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace("[", "")
            .replace("]", "")
        )

    name_alias = {"이름", "name", "학생명", "학생이름", "studentname"}
    m1_alias = {"module1", "모듈1", "m1", "module01", "m1틀린문제", "module1틀린문제", "m1wrong"}
    m2_alias = {"module2", "모듈2", "m2", "module02", "m2틀린문제", "module2틀린문제", "m2wrong"}

    key_map = {c: keyify(c) for c in df.columns}
    rename_map = {}
    found = {"이름": None, "Module1": None, "Module2": None}

    if df.columns.size:
        name_keys = {keyify(x) for x in name_alias}
        m1_keys = {keyify(x) for x in m1_alias}
        m2_keys = {keyify(x) for x in m2_alias}

        for c, k in key_map.items():
            if k in name_keys and found["이름"] is None:
                found["이름"] = c
            elif k in m1_keys and found["Module1"] is None:
                found["Module1"] = c
            elif k in m2_keys and found["Module2"] is None:
                found["Module2"] = c

    if found["이름"]:
        rename_map[found["이름"]] = "이름"
    if found["Module1"]:
        rename_map[found["Module1"]] = "Module1"
    if found["Module2"]:
        rename_map[found["Module2"]] = "Module2"

    return df.rename(columns=rename_map)

def extract_zip_to_dict(zip_file):
    m1_imgs, m2_imgs = {}, {}
    with zipfile.ZipFile(zip_file) as z:
        for file in z.namelist():
            if file.lower().endswith(("png", "jpg", "jpeg", "webp")):
                parts = file.split("/")
                if len(parts) < 2:
                    continue
                folder = parts[0].lower()
                q_num = os.path.splitext(os.path.basename(file))[0]
                with z.open(file) as f:
                    img = Image.open(f).convert("RGB")
                    if folder == "m1":
                        m1_imgs[q_num] = img
                    elif folder == "m2":
                        m2_imgs[q_num] = img
    return m1_imgs, m2_imgs

def create_student_pdf(name, m1_imgs, m2_imgs, doc_title, output_dir):
    if not font_ready:
        return None
    pdf = KoreanPDF()
    pdf.add_page()
    pdf.set_font(pdf_font_name, style="B", size=10)
    pdf.cell(0, 8, txt=f"<{name}_{doc_title}>", ln=True)

    def add_images(title, images):
        est_height = 80
        if images and (pdf.get_y() + 10 + est_height > pdf.page_break_trigger):
            pdf.add_page()

        pdf.set_font(pdf_font_name, size=10)
        pdf.cell(0, 8, txt=title, ln=True)

        if images:
            for img in images:
                temp_filename = f"temp_{datetime.now().timestamp()}_{os.urandom(4).hex()}.jpg"
                img.save(temp_filename)
                pdf.image(temp_filename, w=150)
                try:
                    os.remove(temp_filename)
                except:
                    pass
                pdf.ln(8)
        else:
            pdf.ln(8)

    add_images("<Module1>", m1_imgs)
    add_images("<Module2>", m2_imgs)

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{name}_{doc_title}.pdf")
    pdf.output(pdf_path)
    return pdf_path

# =========================================================
# [Tab 2] PDF 문제 자르기
# =========================================================
MODULE_RE = re.compile(r"<\s*MODULE\s*(\d+)\s*>", re.IGNORECASE)
HEADER_FOOTER_HINT_RE = re.compile(
    r"(YOU,\s*GENIUS|700\+\s*MOCK\s*TEST|Kakaotalk|Instagram|010-\d{3,4}-\d{4}|Module\s*\d+|SECTION)",
    re.IGNORECASE,
)
NUMDOT_RE = re.compile(r"^(\d{1,2})\.$")
NUM_RE = re.compile(r"^\d{1,2}$")
CHOICE_LABELS = ["D)", "C)", "B)", "A)"]
SIDE_PAD_PX = 10
SCAN_ZOOM = 0.6
WHITE_THRESH = 250

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def find_module_on_page(page):
    txt = page.get_text("text") or ""
    m = MODULE_RE.search(txt)
    if not m:
        return None
    mid = int(m.group(1))
    return mid if mid in (1, 2) else None

def group_words_into_lines(words):
    lines = {}
    for w in words:
        x0, y0, x1, y1, txt, block_no, line_no, word_no = w
        key = (block_no, line_no)
        lines.setdefault(key, []).append((x0, y0, x1, y1, txt))
    for k in lines:
        lines[k].sort(key=lambda t: t[0])
    return list(lines.values())

def detect_question_anchors(page, left_ratio=0.25, max_line_chars=4):
    w_page = page.rect.width
    words = page.get_text("words")
    if not words:
        return []
    lines = group_words_into_lines(words)
    anchors = []

    for tokens in lines:
        line_text = " ".join(t[4] for t in tokens).strip()
        compact = re.sub(r"\s+", "", line_text)
        if HEADER_FOOTER_HINT_RE.search(line_text):
            continue
        if len(compact) > max_line_chars:
            continue
        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio:
            continue

        qnum = None
        y_top = None
        for (x0, y0, x1, y1, txt) in tokens:
            m = NUMDOT_RE.match(txt)
            if m:
                qnum = int(m.group(1))
                y_top = y0
                break

        if qnum is None:
            for i in range(len(tokens) - 1):
                t1 = tokens[i][4]
                t2 = tokens[i + 1][4]
                if NUM_RE.match(t1) and t2 == ".":
                    qnum = int(t1)
                    y_top = tokens[i][1]
                    break

        if qnum is None:
            continue
        if not (1 <= qnum <= 22):
            continue
        anchors.append((qnum, y_top))

    anchors.sort(key=lambda t: t[1])
    return anchors

def last_choice_bottom_y_in_band(page, y_from, y_to):
    clip = fitz.Rect(0, y_from, page.rect.width, y_to)
    t = (page.get_text("text", clip=clip) or "")
    if "A)" not in t:
        return None
    for lab in CHOICE_LABELS:
        rects = page.search_for(lab)
        bottoms = [r.y1 for r in rects if (r.y1 >= y_from and r.y0 <= y_to)]
        if bottoms:
            return max(bottoms)
    return None

def find_footer_start_y(page, y_from, y_to):
    ys = []
    for b in page.get_text("blocks"):
        if len(b) < 5:
            continue
        y0 = b[1]
        text = b[4]
        if y0 < y_from or y0 > y_to:
            continue
        if text and HEADER_FOOTER_HINT_RE.search(str(text)):
            ys.append(y0)
    return min(ys) if ys else None

def content_bottom_y(page, y_from, y_to):
    bottoms = []
    for b in page.get_text("blocks"):
        if len(b) < 5:
            continue
        y0, y1, text = b[1], b[3], b[4]
        if y1 < y_from or y0 > y_to:
            continue
        if text and HEADER_FOOTER_HINT_RE.search(str(text)):
            continue
        if text and str(text).strip():
            bottoms.append(y1)
    return max(bottoms) if bottoms else None

def text_x_bounds_in_band(page, y_from, y_to, min_len=2):
    xs0, xs1 = [], []
    for b in page.get_text("blocks"):
        if len(b) < 5:
            continue
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        if y1 < y_from or y0 > y_to:
            continue
        if not text:
            continue
        t = str(text).strip()
        if len(t) < min_len:
            continue
        if HEADER_FOOTER_HINT_RE.search(t):
            continue
        xs0.append(x0)
        xs1.append(x1)
    if not xs0:
        return None
    return min(xs0), max(xs1)

def ink_bbox_by_raster(page, clip, scan_zoom=SCAN_ZOOM, white_thresh=WHITE_THRESH):
    mat = fitz.Matrix(scan_zoom, scan_zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    w, h = img.size
    px = img.load()

    minx, miny = w, h
    maxx, maxy = -1, -1

    step = 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = px[x, y]
            if r < white_thresh or g < white_thresh or b < white_thresh:
                minx = min(minx, x)
                miny = min(miny, y)
                maxx = max(maxx, x)
                maxy = max(maxy, y)

    if maxx < 0:
        return None
    return (minx, miny, maxx, maxy, w, h)

def px_bbox_to_page_rect(clip, px_bbox, pad_px=10):
    minx, miny, maxx, maxy, w, h = px_bbox
    minx = max(0, minx - pad_px)
    miny = max(0, miny - pad_px)
    maxx = min(w - 1, maxx + pad_px)
    maxy = min(h - 1, maxy + pad_px)

    x0 = clip.x0 + (minx / (w - 1)) * (clip.x1 - clip.x0)
    x1 = clip.x0 + (maxx / (w - 1)) * (clip.x1 - clip.x0)
    y0 = clip.y0 + (miny / (h - 1)) * (clip.x1 - clip.x0)  # FIX: (typo 방지용) 아래에서 다시 계산
    y0 = clip.y0 + (miny / (h - 1)) * (clip.y1 - clip.y0)
    y1 = clip.y0 + (maxy / (h - 1)) * (clip.y1 - clip.y0)
    return fitz.Rect(x0, y0, x1, y1)

def render_png(page, clip, zoom):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    return pix.tobytes("png")

def expand_rect_to_width_right_only(rect, target_width, page_width):
    cur = rect.width
    if cur >= target_width:
        return rect
    new_x0 = rect.x0
    new_x1 = rect.x0 + target_width
    new_x1 = clamp(new_x1, new_x0 + 80, page_width)
    return fitz.Rect(new_x0, rect.y0, new_x1, rect.y1)

def compute_rects_for_pdf(pdf_bytes, zoom=3.0, pad_top=10, pad_bottom=12, frq_extra_space_px=250):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    rects = []
    current_module = None
    side_pad_pt = SIDE_PAD_PX / zoom
    frq_extra_pt = frq_extra_space_px / zoom

    for pno in range(len(doc)):
        page = doc[pno]
        w, h = page.rect.width, page.rect.height
        page_blocks = page.get_text("blocks")

        mid = find_module_on_page(page)
        if mid is not None:
            current_module = mid
        if current_module not in (1, 2):
            continue

        anchors = detect_question_anchors(page)
        if not anchors:
            continue

        for i, (qnum, y0) in enumerate(anchors):
            y_start_candidate = clamp(y0 - pad_top, 0, h)
            safe_y = y_start_candidate
            for b in page_blocks:
                b_y1 = b[3]
                b_text = b[4]
                if HEADER_FOOTER_HINT_RE.search(b_text):
                    if b_y1 < y0 and b_y1 > safe_y:
                        safe_y = b_y1 + 2
                else:
                    if b_y1 > safe_y and b_y1 < y0 - 2:
                        safe_y = b_y1 + 2
            y_start = clamp(safe_y, 0, h)

            if i + 1 < len(anchors):
                next_y = anchors[i + 1][1]
                y_cap = clamp(next_y - 1, 0, h)
                y_end = clamp(next_y - pad_bottom, y_start + 80, y_cap)
            else:
                y_cap = h
                y_end = clamp(h - 8, y_start + 80, h)

            footer_y = find_footer_start_y(page, y_start, y_cap)
            if footer_y is not None and footer_y > y_start + 120:
                y_cap = min(y_cap, footer_y - 4)
                y_end = min(y_end, y_cap)

            mcq_last = last_choice_bottom_y_in_band(page, y_start, y_cap)
            is_frq = (mcq_last is None)

            if mcq_last is not None:
                y_end = clamp(max(y_end, mcq_last + 18), y_start + 80, y_cap)

            bottom = content_bottom_y(page, y_start, y_end)
            if bottom is not None and bottom > y_start + 140:
                if mcq_last is not None:
                    bottom = max(bottom, mcq_last + 10)
                y_end = min(y_end, bottom + 14)

            xb = text_x_bounds_in_band(page, y_start, y_end)
            if xb is None:
                x0, x1 = 0, w
            else:
                x0 = clamp(xb[0] - side_pad_pt, 0, w)
                x1 = clamp(xb[1] + side_pad_pt, x0 + 80, w)

            scan_clip = fitz.Rect(0, y_start, w, y_end)
            px_bbox = ink_bbox_by_raster(page, scan_clip)
            if px_bbox is not None:
                tight = px_bbox_to_page_rect(scan_clip, px_bbox)
                x0 = clamp(tight.x0, 0, w)
                x1 = clamp(tight.x1, x0 + 80, w)
                new_y_end = clamp(tight.y1, y_start + 80, y_end)
                if mcq_last is not None:
                    new_y_end = max(new_y_end, mcq_last + 12)
                y_end = clamp(new_y_end, y_start + 80, y_end)

            if is_frq:
                y_end = min(y_cap, y_end + frq_extra_pt)

            rects.append({"mod": current_module, "qnum": qnum, "page": pno, "rect": fitz.Rect(x0, y_start, x1, y_end), "page_width": w})

    return doc, rects

def make_zip_from_rects(doc, rects, zoom, zip_base_name, unify_width_right=True):
    maxw = {1: 0.0, 2: 0.0}
    for r in rects:
        maxw[r["mod"]] = max(maxw[r["mod"]], r["rect"].width)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for r in rects:
            page = doc[r["page"]]
            rect = r["rect"]
            if unify_width_right and maxw.get(r["mod"], 0) > 0:
                rect = expand_rect_to_width_right_only(rect, maxw[r["mod"]], r["page_width"])
            png = render_png(page, rect, zoom)
            z.writestr(f"M{r['mod']}/{r['qnum']}.png", png)
    buf.seek(0)
    return buf, zip_base_name + ".zip"

# =========================================================
# [Tab 3] 개인 성적표 (ReportLab PDF + PDF->PNG + ZIP)
# =========================================================
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def ensure_fonts_registered():
    try:
        pdfmetrics.registerFont(TTFont("NanumGothic", FONT_REGULAR))
    except:
        pass
    try:
        pdfmetrics.registerFont(TTFont("NanumGothic-Bold", FONT_BOLD))
    except:
        pass

def str_w(text, font_name, font_size):
    return pdfmetrics.stringWidth(text, font_name, font_size)

def fit_font_size(text, font_name, max_size, min_size, max_width):
    s = max_size
    while s >= min_size:
        if str_w(text, font_name, s) <= max_width:
            return s
        s -= 0.5
    return min_size

def fit_font_size_two_lines(lines, font_name, max_size, min_size, max_width):
    need = max_size
    for ln in lines:
        ln = ln.strip()
        if ln == "":
            continue
        need = min(need, fit_font_size(ln, font_name, max_size, min_size, max_width))
    return need

def draw_round_rect(c, x, y, w, h, r, fill, stroke, stroke_width=1):
    c.setLineWidth(stroke_width)
    c.setStrokeColor(stroke)
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, r, fill=1, stroke=1)

def draw_text_center(c, x_center, y_baseline, text, font_name, font_size, color=colors.black):
    c.setFont(font_name, font_size)
    c.setFillColor(color)
    tw = str_w(text, font_name, font_size)
    c.drawString(x_center - tw / 2, y_baseline, text)

def wr_to_text(v):
    if v is None:
        return "-"
    try:
        v = float(v)
        return f"{int(round(v * 100))}%"
    except:
        return "-"

def build_wrong_rate_dict_fixed_ranges(eta_xl, sheet_name="Error Analysis"):
    df = pd.read_excel(eta_xl, sheet_name=sheet_name, header=None)
    colC = df.iloc[:, 2].tolist()  # C열

    m1_vals = colC[2:24]     # C3:C24
    m2_vals = colC[25:47]    # C26:C47

    def to_dict(vals):
        out = {}
        for i, v in enumerate(vals, start=1):
            try:
                out[i] = float(v)
            except:
                out[i] = None
        return out

    return to_dict(m1_vals), to_dict(m2_vals)

def read_mock_answers(mock_bytes):
    df = pd.read_excel(mock_bytes)
    cols = set(df.columns.astype(str))

    if {"모듈", "문항번호", "정답"}.issubset(cols):
        m1 = df[df["모듈"].astype(str).str.upper().eq("M1")].set_index("문항번호")["정답"].astype(str).to_dict()
        m2 = df[df["모듈"].astype(str).str.upper().eq("M2")].set_index("문항번호")["정답"].astype(str).to_dict()
        m1 = {int(k): _clean(v) for k, v in m1.items() if str(k).strip().isdigit()}
        m2 = {int(k): _clean(v) for k, v in m2.items() if str(k).strip().isdigit()}
        return m1, m2

    # fallback
    c0, c1 = df.columns[0], df.columns[1]
    m2_idxs = df.index[df[c0].astype(str).str.contains("Module2", case=False, na=False)].tolist()
    if not m2_idxs:
        out = {}
        for _, r in df.iterrows():
            try:
                q = int(str(r[c0]).strip())
            except:
                continue
            out[q] = _clean(r[c1])
        return out, {}

    m2i = m2_idxs[0]
    m1_rows = df.iloc[:m2i]
    m2_rows = df.iloc[m2i + 1:]

    def rows_to_ans(rows):
        dct = {}
        for _, r in rows.iterrows():
            try:
                q = int(str(r[c0]).strip())
            except:
                continue
            dct[q] = _clean(r[c1])
        return dct

    return rows_to_ans(m1_rows), rows_to_ans(m2_rows)

def create_report_pdf_reportlab(
    output_path: str,
    title: str,
    subtitle: str,
    gen_date_str: str,
    student_name: str,
    m1_meta: dict,
    m2_meta: dict,
    ans_m1: dict,
    ans_m2: dict,
    wr_m1: dict,
    wr_m2: dict,
    wrong_m1: set,
    wrong_m2: set,
    result_blank: bool = False,          # ✅ TEMPLATE용 Result 공란
    footer_left_text: str = "",          # ✅ 하단왼쪽 텍스트
):
    ensure_fonts_registered()
    c = canvas.Canvas(output_path, pagesize=A4)
    W, H = A4

    # colors
    stroke = colors.Color(203 / 255, 213 / 255, 225 / 255)
    title_col = colors.Color(15 / 255, 23 / 255, 42 / 255)
    muted = colors.Color(100 / 255, 116 / 255, 139 / 255)
    pill_fill = colors.Color(241 / 255, 245 / 255, 249 / 255)
    row_stripe = colors.Color(248 / 255, 250 / 255, 252 / 255)
    green = colors.Color(22 / 255, 101 / 255, 52 / 255)
    red = colors.Color(220 / 255, 38 / 255, 38 / 255)

    # layout
    L = 15 * mm
    R = 15 * mm
    TOP = H - 18 * mm
    usable_w = W - L - R

    # Generated (조금 위로)
    c.setFont("NanumGothic", 10)
    c.setFillColor(muted)
    c.drawRightString(W - R, TOP + 8 * mm, f"Generated: {gen_date_str}")

    # Title / subtitle
    c.setFillColor(title_col)
    c.setFont("NanumGothic-Bold", 30)
    c.drawString(L, TOP, title)

    c.setFillColor(muted)
    c.setFont("NanumGothic", 14)
    c.drawString(L, TOP - 11 * mm, subtitle)

    # Name pill (조금 아래로)
    pill_w = 78 * mm
    pill_h = 20 * mm
    pill_x = L + usable_w - pill_w
    pill_y = TOP - 9 * mm  # ✅ 조금 내려줌
    draw_round_rect(c, pill_x, pill_y, pill_w, pill_h, 10 * mm, pill_fill, stroke, 1)

    c.setFillColor(muted)
    c.setFont("NanumGothic-Bold", 10)
    c.drawString(pill_x + 7 * mm, pill_y + 12.2 * mm, "Name")

    c.setFillColor(title_col)
    max_name_w = pill_w - 26 * mm
    name_fs = fit_font_size(student_name, "NanumGothic-Bold", 16, 10, max_name_w)
    c.setFont("NanumGothic-Bold", name_fs)
    c.drawRightString(pill_x + pill_w - 7 * mm, pill_y + 6.0 * mm, student_name)

    # divider
    line_y = TOP - 22 * mm
    c.setLineWidth(2)
    c.setStrokeColor(title_col)
    c.line(L, line_y, W - R, line_y)

    # KPI cards (높이 줄이고, 점수/날짜 겹침 해결)
    kpi_h = 32 * mm  # ✅ 더 줄임 (40 -> 32)
    gap = 10 * mm
    kpi_w = (usable_w - gap) / 2

    kpi_gap_from_line = 7 * mm
    kpi_y = line_y - kpi_gap_from_line - kpi_h

    def draw_kpi_card(x, y, w, h, label, score, dt, t):
        draw_round_rect(c, x, y, w, h, 8 * mm, colors.white, stroke, 1)

        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 16)
        c.drawString(x + 8 * mm, y + h - 11 * mm, label)

        # score 조금 아래로
        c.setFont("NanumGothic-Bold", 32)
        c.drawRightString(x + w - 8 * mm, y + h - 13.5 * mm, str(score))

        # mid line
        mid_y = y + 11 * mm
        c.setLineWidth(0.6)
        c.setStrokeColor(colors.Color(241 / 255, 245 / 255, 249 / 255))
        c.line(x + 6 * mm, mid_y, x + w - 6 * mm, mid_y)

        # Date/Time 글자 줄이고, 겹침 방지 배치
        c.setFillColor(muted)
        c.setFont("NanumGothic", 9)  # ✅ 더 작게
        c.drawString(x + 8 * mm, y + 5.2 * mm, f"{dt}")

        c.setFont("NanumGothic", 9)
        c.drawRightString(x + w - 8 * mm, y + 5.2 * mm, f"{t}")

    draw_kpi_card(L, kpi_y, kpi_w, kpi_h, "Module 1", m1_meta["score"], m1_meta["dt"], m1_meta["time"])
    draw_kpi_card(L + kpi_w + gap, kpi_y, kpi_w, kpi_h, "Module 2", m2_meta["score"], m2_meta["dt"], m2_meta["time"])

    # Analysis tables start
    cards_top = kpi_y - 8 * mm
    card_gap_bottom = 14 * mm  # 하단 푸터/여백용
    card_h = cards_top - (20 * mm)  # 대략값(아래에서 끝 맞춤)
    # ✅ 22번 끝줄에 표도 끝나게: row_h로 높이를 계산해서 card_h 결정
    header_h = 8.5 * mm
    table_header_gap = 7.5 * mm
    row_h = 6.3 * mm
    top_padding = 6.5 * mm
    bottom_padding = 6.0 * mm
    card_h = top_padding + header_h + table_header_gap + (22 * row_h) + bottom_padding

    card_y = cards_top - card_h

    card_w = kpi_w
    left_x = L
    right_x = L + card_w + gap

    def draw_table(x, y, w, h, module_name, ans_dict, wr_dict, wrong_set):
        draw_round_rect(c, x, y, w, h, 10 * mm, colors.white, stroke, 1)

        # title (검은 캡 제거, 그냥 텍스트)
        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 14)
        c.drawString(x + 9 * mm, y + h - 12 * mm, module_name)

        # column header
        strip_y = y + h - 22.5 * mm
        strip_h = 8.5 * mm
        draw_round_rect(c, x + 6 * mm, strip_y, w - 12 * mm, strip_h, 6 * mm, pill_fill, stroke, 1)

        inner_x = x + 8 * mm
        inner_w = w - 16 * mm

        # ✅ Answer 칸을 줄이고 정답률/Result를 넓힘
        col_q = 10 * mm
        col_ans = 18 * mm
        col_wr = 22 * mm
        col_res = inner_w - (col_q + col_ans + col_wr)

        q_center = inner_x + col_q / 2
        ans_center = inner_x + col_q + col_ans / 2
        wr_center = inner_x + col_q + col_ans + col_wr / 2
        res_center = inner_x + col_q + col_ans + col_wr + col_res / 2

        header_text_y = strip_y + 2.5 * mm
        draw_text_center(c, q_center, header_text_y, "No.", "NanumGothic-Bold", 9.5, muted)
        draw_text_center(c, ans_center, header_text_y, "Answer", "NanumGothic-Bold", 9.5, muted)
        draw_text_center(c, wr_center, header_text_y, "정답률", "NanumGothic-Bold", 9.5, muted)
        draw_text_center(c, res_center, header_text_y, "Result", "NanumGothic-Bold", 9.5, muted)

        start_y = strip_y - 2.1 * mm - row_h
        base = 1.6 * mm

        for i, q in enumerate(range(1, 23)):
            ry = start_y - i * row_h

            # stripe
            if q % 2 == 0:
                c.setFillColor(row_stripe)
                c.setStrokeColor(row_stripe)
                c.roundRect(x + 6 * mm, ry, w - 12 * mm, row_h, 6 * mm, fill=1, stroke=0)

            ans_raw = _clean(ans_dict.get(q, ""))
            lines = ans_raw.split("\n") if "\n" in ans_raw else [ans_raw]
            lines = [ln.strip() for ln in lines if ln.strip() != ""]
            if not lines:
                lines = [""]

            if len(lines) > 2:
                lines = [lines[0], " ".join(lines[1:])]

            rate_val = wr_dict.get(q, None)
            wr_txt = wr_to_text(rate_val)

            # ✅ Result 공란 옵션
            if result_blank:
                res_txt = ""
            else:
                res_txt = "X" if q in wrong_set else "O"

            # No.
            draw_text_center(c, q_center, ry + base, str(q), "NanumGothic", 10.2, title_col)

            # Answer (작게)
            ans_max_w = col_ans - 3 * mm
            fs = fit_font_size_two_lines(lines, "NanumGothic-Bold", 10.0, 6.6, ans_max_w)
            if len(lines) == 1:
                draw_text_center(c, ans_center, ry + base, lines[0], "NanumGothic-Bold", fs, title_col)
            else:
                draw_text_center(c, ans_center, ry + (base + 0.75 * mm), lines[0], "NanumGothic-Bold", fs, title_col)
                draw_text_center(c, ans_center, ry + (base - 0.75 * mm), lines[1], "NanumGothic-Bold", fs, title_col)

            # 정답률 (50% 미만 굵게)
            is_low = False
            try:
                if rate_val is not None and float(rate_val) < 0.5:
                    is_low = True
            except:
                pass

            if is_low:
                draw_text_center(c, wr_center, ry + base, wr_txt, "NanumGothic-Bold", 10.6, title_col)
            else:
                draw_text_center(c, wr_center, ry + base, wr_txt, "NanumGothic", 10.2, title_col)

            # Result
            if res_txt == "":
                # 템플릿 공란
                draw_text_center(c, res_center, ry + base, "", "NanumGothic-Bold", 11.6, title_col)
            else:
                ox_color = red if res_txt == "X" else green
                draw_text_center(c, res_center, ry + base, res_txt, "NanumGothic-Bold", 11.6, ox_color)

    draw_table(left_x, card_y, card_w, card_h, "Module 1", ans_m1, wr_m1, wrong_m1)
    draw_table(right_x, card_y, card_w, card_h, "Module 2", ans_m2, wr_m2, wrong_m2)

    # Footer left (글자 8)
    if footer_left_text:
        c.setFillColor(title_col)
        c.setFont("NanumGothic", 8)
        c.drawString(L, 12 * mm, footer_left_text)

    c.showPage()
    c.save()
    return output_path

def render_pdf_to_png(pdf_path: str, png_path: str, zoom: float = 2.0):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    with open(png_path, "wb") as f:
        f.write(pix.tobytes("png"))
    doc.close()

# =========================================================
# 메인 UI
# =========================================================
tab1, tab2, tab3 = st.tabs(["📝 오답노트 생성기", "✂️ 문제캡처 ZIP생성기", "📊 개인 성적표"])

# ---------------------------------------------------------
# Tab 1
# ---------------------------------------------------------
with tab1:
    st.header("📝 SAT 오답노트 생성기")

    if "generated_files" not in st.session_state:
        st.session_state.generated_files = []
    if "zip_buffer" not in st.session_state:
        st.session_state.zip_buffer = None
    if "skipped_details" not in st.session_state:
        st.session_state.skipped_details = {}

    doc_title = st.text_input("문서 제목", value="Mock Test1", key="t1_title")
    img_zip = st.file_uploader("m1, m2 폴더가 들어있는 ZIP 파일", type="zip", key="t1_zip")
    excel_file = st.file_uploader("학생 결과 엑셀(.xlsx)", type="xlsx", key="t1_excel")

    if st.button("🚀 오답노트 생성 시작", type="primary", key="t1_btn"):
        if not img_zip or not excel_file:
            st.warning("⚠️ 파일들을 모두 업로드해주세요.")
        else:
            try:
                m1_imgs, m2_imgs = extract_zip_to_dict(img_zip)
                raw = pd.read_excel(excel_file)
                df = normalize_columns(raw)

                missing = {"이름", "Module1", "Module2"} - set(df.columns)
                if missing:
                    st.error(f"필수 컬럼 누락: {missing}")
                    st.stop()

                output_dir = "generated_pdfs"
                os.makedirs(output_dir, exist_ok=True)

                temp_files = []
                skipped_details = {"만점": [], "M1/M2 하나 미제출": [], "미제출": []}
                progress_bar = st.progress(0)

                for idx, row in df.iterrows():
                    name = row["이름"]

                    def parse_module_data(x):
                        if pd.isna(x):
                            return None
                        s = str(x).strip()
                        if s == "":
                            return None
                        if s.upper() in ["X", "Х", "-"]:
                            return []
                        s = s.replace("，", ",").replace(";", ",")
                        nums = [t.strip() for t in s.split(",") if t.strip()]
                        return nums if nums else []

                    m1_data = parse_module_data(row["Module1"])
                    m2_data = parse_module_data(row["Module2"])

                    skip_reason = None
                    if m1_data is None and m2_data is None:
                        skip_reason = "미제출"
                    elif m1_data is None or m2_data is None:
                        skip_reason = "M1/M2 하나 미제출"
                    elif len(m1_data) == 0 and len(m2_data) == 0:
                        skip_reason = "만점"

                    if skip_reason:
                        skipped_details[skip_reason].append(name)
                        progress_bar.progress((idx + 1) / len(df))
                        continue

                    m1_list = [m1_imgs[n] for n in m1_data] if m1_data else []
                    m2_list = [m2_imgs[n] for n in m2_data] if m2_data else []

                    pdf_path = create_student_pdf(name, m1_list, m2_list, doc_title, output_dir)
                    if pdf_path:
                        temp_files.append((name, pdf_path))
                    progress_bar.progress((idx + 1) / len(df))

                st.session_state.generated_files = temp_files
                st.session_state.skipped_details = skipped_details

                if temp_files:
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zipf:
                        for name, path in temp_files:
                            zipf.write(path, os.path.basename(path))
                    zip_buf.seek(0)
                    st.session_state.zip_buffer = zip_buf
                    st.success(f"✅ 총 {len(temp_files)}명의 오답노트 생성 완료!")
                else:
                    st.warning("생성된 파일이 없습니다.")
            except Exception as e:
                st.error(f"오류 발생: {e}")

    if st.session_state.zip_buffer:
        st.download_button(
            "📦 전체 오답노트 ZIP 다운로드",
            st.session_state.zip_buffer,
            file_name=f"오답노트_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            key="t1_down_all",
        )

# ---------------------------------------------------------
# Tab 2
# ---------------------------------------------------------
with tab2:
    st.header("✂️ 문제캡처 ZIP생성기")
    st.info("SAT Mock PDF를 업로드하면 문제 번호를 인식하여 자릅니다.")
    pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"], key="t2_pdf")

    c1, c2, c3, c4 = st.columns(4)
    zoom_val = c1.slider("해상도(zoom)", 2.0, 4.0, 3.0, 0.1, key="t2_zoom")
    pt_val = c2.slider("위 여백", 0, 140, 10, 1, key="t2_pt")
    pb_val = c3.slider("아래 여백", 0, 200, 12, 1, key="t2_pb")
    frq_val = c4.slider("FRQ 여백", 0, 600, 250, 25, key="t2_frq")
    unify_width = st.checkbox("가로폭 통일", value=True, key="t2_chk")

    if pdf_file and st.button("✂️ 자르기 & ZIP 생성", type="primary", key="t2_btn"):
        with st.spinner("이미지 생성 중..."):
            try:
                pdf_bytes = pdf_file.read()
                pdf_name = pdf_file.name
                zip_base = pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name
                doc_obj, rects_data = compute_rects_for_pdf(
                    pdf_bytes,
                    zoom=zoom_val,
                    pad_top=pt_val,
                    pad_bottom=pb_val,
                    frq_extra_space_px=frq_val,
                )
                zbuf_data, zname = make_zip_from_rects(
                    doc_obj,
                    rects_data,
                    zoom_val,
                    zip_base,
                    unify_width_right=unify_width,
                )
                st.success(f"✅ 완료! (총 {len(rects_data)}문제)")
                st.download_button("📦 ZIP 다운로드", data=zbuf_data, file_name=zname, mime="application/zip", key="t2_down")
            except Exception as e:
                st.error(f"오류 발생: {e}")

# ---------------------------------------------------------
# Tab 3
# ---------------------------------------------------------
with tab3:
    st.header("📊 개인 성적표")
    st.info("ETA.xlsx (Student Analysis + QuizResults + Error/Accuracy Analysis) + Mock데이터.xlsx")

    eta_file = st.file_uploader("ETA 결과 파일 (ETA.xlsx)", type=["xlsx"], key="t3_eta")
    mock_file = st.file_uploader("Mock 정답 파일 (Mock데이터.xlsx)", type=["xlsx"], key="t3_mock")

    c1, c2 = st.columns([1, 1])
    with c1:
        report_title = st.text_input("리포트 제목", value="SAT Math Report", key="t3_title")
    with c2:
        generated_date = st.date_input("Generated 날짜", value=datetime.now().date(), key="t3_gen_date")

    # sheets/cols
    STUDENT_SHEET = "Student Analysis"
    QUIZ_SHEET = "QuizResults"
    ERROR_SHEET = "Error Analysis"

    SA_HEADER_ROW_IDX = 1
    QZ_HEADER_ROW_IDX = 0

    SA_NAME_COL = "학생 이름"
    QZ_KEYWORD_COL = "검색 키워드"
    QZ_MODULE_COL = "모듈"
    QZ_NAME_COL = "학생 이름"
    QZ_DT_COL = "응답 날짜"
    QZ_TIME_COL = "소요 시간"
    QZ_SCORE_COL = "점수"
    QZ_WRONG_COL = "틀린 문제 번호"

    footer_left_text = "Kakaotalk : yujinj524\nPhone : 010-6395-8733"

    if st.button("🚀 개인 성적표 생성 (PDF + PNG)", type="primary", key="t3_btn"):
        if not eta_file or not mock_file:
            st.warning("⚠️ 파일 2개를 모두 업로드해주세요.")
            st.stop()
        if not font_ready:
            st.error("⚠️ fonts 폴더에 폰트 파일이 필요합니다.")
            st.stop()

        try:
            eta_xl = pd.ExcelFile(eta_file)

            # 1) Student Analysis
            if STUDENT_SHEET not in eta_xl.sheet_names:
                st.error(f"'{STUDENT_SHEET}' 시트가 없습니다.")
                st.stop()

            raw_sa = pd.read_excel(eta_xl, sheet_name=STUDENT_SHEET, header=None)
            if raw_sa.shape[0] <= SA_HEADER_ROW_IDX:
                st.error("Student Analysis 데이터 부족(헤더행 없음)")
                st.stop()

            sa_header = raw_sa.iloc[SA_HEADER_ROW_IDX].astype(str).tolist()
            student_df = raw_sa.iloc[SA_HEADER_ROW_IDX + 1:].copy()
            student_df.columns = sa_header
            student_df = student_df.dropna(axis=1, how="all").dropna(axis=0, how="all")
            assert_columns(student_df, [SA_NAME_COL], STUDENT_SHEET)

            students = [_clean(x) for x in student_df[SA_NAME_COL].dropna().tolist()]
            students = [s for s in students if s != ""]
            if not students:
                st.error("학생 목록이 비어있습니다.")
                st.stop()

            # 2) QuizResults
            if QUIZ_SHEET not in eta_xl.sheet_names:
                st.error(f"'{QUIZ_SHEET}' 시트가 없습니다.")
                st.stop()

            quiz_df = pd.read_excel(eta_xl, sheet_name=QUIZ_SHEET, header=QZ_HEADER_ROW_IDX)
            quiz_df.columns = [str(c).strip() for c in quiz_df.columns]
            quiz_df = quiz_df.dropna(axis=1, how="all").dropna(axis=0, how="all")

            assert_columns(
                quiz_df,
                [QZ_KEYWORD_COL, QZ_MODULE_COL, QZ_NAME_COL, QZ_DT_COL, QZ_TIME_COL, QZ_SCORE_COL, QZ_WRONG_COL],
                QUIZ_SHEET,
            )

            quiz_map = {}
            for _, r in quiz_df.iterrows():
                nm = _clean(r.get(QZ_NAME_COL, ""))
                md = _clean(r.get(QZ_MODULE_COL, "")).upper()
                if nm == "":
                    continue

                if md in ["M1", "MODULE1", "1"]:
                    mod = 1
                elif md in ["M2", "MODULE2", "2"]:
                    mod = 2
                else:
                    continue

                quiz_map.setdefault(nm, {})[mod] = {
                    "dt": _clean(r.get(QZ_DT_COL, "")) or "-",
                    "time": _clean(r.get(QZ_TIME_COL, "")) or "-",
                    "score": score_to_slash22(r.get(QZ_SCORE_COL, "")),
                    "wrong_set": parse_wrong_list(r.get(QZ_WRONG_COL, "")),
                    "keyword": _clean(r.get(QZ_KEYWORD_COL, "")) or "",
                }

            # 3) Accuracy/정답률 시트 찾기
            target_sheet_name = None
            if "Accuracy Analysis" in eta_xl.sheet_names:
                target_sheet_name = "Accuracy Analysis"
            elif "Error Analysis" in eta_xl.sheet_names:
                target_sheet_name = "Error Analysis"

            if target_sheet_name:
                wr1, wr2 = build_wrong_rate_dict_fixed_ranges(eta_xl, target_sheet_name)
            else:
                wr1, wr2 = {}, {}

            # 4) Mock Answers
            ans1, ans2 = read_mock_answers(mock_file)

            # output dirs
            out_pdf = "generated_reports_pdf"
            out_png = "generated_reports_png"
            os.makedirs(out_pdf, exist_ok=True)
            os.makedirs(out_png, exist_ok=True)

            made_files = []  # (stu, pdf_path, png_path)
            skipped = []
            prog = st.progress(0)

            # ===== TEMPLATE 항상 생성 (Name='-', Result 공란) =====
            template_name = "-"
            template_m1_meta = {"score": " / 22", "dt": "-", "time": "-"}
            template_m2_meta = {"score": " / 22", "dt": "-", "time": "-"}

            template_pdf_path = os.path.join(out_pdf, f"TEMPLATE_{generated_date.strftime('%Y%m%d')}.pdf")
            template_png_path = os.path.join(out_png, f"TEMPLATE_{generated_date.strftime('%Y%m%d')}.png")

            create_report_pdf_reportlab(
                output_path=template_pdf_path,
                title=report_title,
                subtitle="-",
                gen_date_str=generated_date.strftime("%Y-%m-%d"),
                student_name=template_name,
                m1_meta=template_m1_meta,
                m2_meta=template_m2_meta,
                ans_m1=ans1,
                ans_m2=ans2,
                wr_m1=wr1,
                wr_m2=wr2,
                wrong_m1=set(),
                wrong_m2=set(),
                result_blank=True,  # ✅ Result 공란
                footer_left_text=footer_left_text,
            )
            render_pdf_to_png(template_pdf_path, template_png_path, zoom=2.0)

            # ===== 학생별 생성 =====
            for i, stu in enumerate(students):
                q = quiz_map.get(stu, {})
                m1 = q.get(1, {})
                m2 = q.get(2, {})

                m1_score_txt = _clean(m1.get("score", ""))
                m2_score_txt = _clean(m2.get("score", ""))

                if m1_score_txt == "" or m2_score_txt == "":
                    skipped.append(stu)
                    prog.progress((i + 1) / len(students))
                    continue

                subtitle_kw = _clean(m1.get("keyword", "")) or _clean(m2.get("keyword", "")) or "-"

                m1_meta = {"score": m1_score_txt, "dt": m1.get("dt", "-"), "time": m1.get("time", "-")}
                m2_meta = {"score": m2_score_txt, "dt": m2.get("dt", "-"), "time": m2.get("time", "-")}

                wrong1 = set(m1.get("wrong_set", set()))
                wrong2 = set(m2.get("wrong_set", set()))

                pdf_path = os.path.join(out_pdf, f"{stu}_{generated_date.strftime('%Y%m%d')}.pdf")
                png_path = os.path.join(out_png, f"{stu}_{generated_date.strftime('%Y%m%d')}.png")

                create_report_pdf_reportlab(
                    output_path=pdf_path,
                    title=report_title,
                    subtitle=subtitle_kw,
                    gen_date_str=generated_date.strftime("%Y-%m-%d"),
                    student_name=stu,
                    m1_meta=m1_meta,
                    m2_meta=m2_meta,
                    ans_m1=ans1,
                    ans_m2=ans2,
                    wr_m1=wr1,
                    wr_m2=wr2,
                    wrong_m1=wrong1,
                    wrong_m2=wrong2,
                    result_blank=False,
                    footer_left_text=footer_left_text,
                )
                render_pdf_to_png(pdf_path, png_path, zoom=2.0)

                made_files.append((stu, pdf_path, png_path))
                prog.progress((i + 1) / len(students))

            # ===== ZIP (PDF + PNG + TEMPLATE) =====
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                # template
                if os.path.exists(template_pdf_path):
                    z.write(template_pdf_path, arcname=f"PDF/{os.path.basename(template_pdf_path)}")
                if os.path.exists(template_png_path):
                    z.write(template_png_path, arcname=f"PNG/{os.path.basename(template_png_path)}")

                # students
                for stu, pdf_path, png_path in made_files:
                    if os.path.exists(pdf_path):
                        z.write(pdf_path, arcname=f"PDF/{os.path.basename(pdf_path)}")
                    if os.path.exists(png_path):
                        z.write(png_path, arcname=f"PNG/{os.path.basename(png_path)}")

            zip_buf.seek(0)

            st.success(f"✅ 생성 완료: {len(made_files)}명 + TEMPLATE 1개 (제외: {len(skipped)}명)")
            if skipped:
                with st.expander(f"제외된 학생 ({len(skipped)}명) - 점수 blank"):
                    for s in skipped:
                        st.write(f"- {s}")

            st.download_button(
                "📦 (PDF+PNG+TEMPLATE) 통합 ZIP 다운로드",
                data=zip_buf,
                file_name=f"개인성적표_PDF+PNG_{generated_date.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="t3_download_zip_all",
            )

        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.exception(e)
