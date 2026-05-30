import streamlit as st
import pandas as pd
import zipfile
import os
import io
import re
import logging
import warnings
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import fitz  # PyMuPDF

# ==============================
# [FIX] fontTools subset 로그 폭주 차단 + DeprecationWarning 숨김
# ==============================
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==============================
# [FIX] fpdf2 신/구버전 호환: ln=True 대체용 함수
# ==============================
try:
    from fpdf.enums import XPos, YPos

    def pdf_cell_ln(pdf: FPDF, w, h, text: str, **kwargs):
        pdf.cell(w, h, text=text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, **kwargs)
except Exception:
    def pdf_cell_ln(pdf: FPDF, w, h, text: str, **kwargs):
        pdf.cell(w, h, txt=text, ln=True, **kwargs)

# ==============================
# 0. 기본 설정
# ==============================
st.set_page_config(page_title="SAT MATH", layout="centered")

# 폰트 설정 (오답노트용)
FONT_REGULAR = "fonts/NanumGothic.ttf"
FONT_BOLD = "fonts/NanumGothicBold.ttf"
pdf_font_name = "NanumGothic"

# 폰트 존재 여부 확인
font_ready = os.path.exists(FONT_REGULAR) and os.path.exists(FONT_BOLD)

if font_ready:
    class KoreanPDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_margins(25.4, 30, 25.4)
            self.set_auto_page_break(auto=True, margin=25.4)
            # [FIX] uni=True 제거 (fpdf2 deprecation)
            self.add_font(pdf_font_name, style="", fname=FONT_REGULAR)
            self.add_font(pdf_font_name, style="B", fname=FONT_BOLD)
            self.set_font(pdf_font_name, size=10)
else:
    st.error("⚠️ 한글 PDF 생성을 위해 fonts 폴더에 NanumGothic.ttf 와 NanumGothicBold.ttf 모두 필요합니다.")

# =========================================================
# [Tab 1] 오답노트 생성기 관련 함수
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    def keyify(s: str) -> str:
        return (
            str(s)
            .replace("\u3000", " ")
            .replace("\u00A0", " ")
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
        )

    name_alias = {
        "학생 이름",
        "학생이름",
        "이름",
        "학생명",
        "name",
        "studentname"
    }

    m1_alias = {
        "[M1] 틀린 문제",
        "[M1] 틀린문제",
        "M1 틀린 문제",
        "M1 틀린문제",
        "m1틀린문제",
        "module1틀린문제",
        "m1wrong",
        "module1wrong"
    }

    m2_alias = {
        "[M2] 틀린 문제",
        "[M2] 틀린문제",
        "M2 틀린 문제",
        "M2 틀린문제",
        "m2틀린문제",
        "module2틀린문제",
        "m2wrong",
        "module2wrong"
    }

    name_keys = {keyify(x) for x in name_alias}
    m1_keys = {keyify(x) for x in m1_alias}
    m2_keys = {keyify(x) for x in m2_alias}

    rename_map = {}

    for col in df.columns:
        key = keyify(col)

        if key in name_keys:
            rename_map[col] = "이름"

        elif key in m1_keys:
            rename_map[col] = "Module1"

        elif key in m2_keys:
            rename_map[col] = "Module2"

    df = df.rename(columns=rename_map)

    return df


def example_input_df():
    return pd.DataFrame({
        '학생 이름': ['홍길동', '김철수', '이영희', '박지성', '손흥민'],
        '[M1] 틀린 문제': ['1,3,5', 'X', 'X', '1', None],
        '[M2] 틀린 문제': ['X', '1,3', 'X', None, None]
    })


def get_example_excel():
    output = io.BytesIO()
    df = example_input_df()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="예시")
    output.seek(0)
    return output


def extract_zip_to_dict(zip_file):
    m1_imgs, m2_imgs = {}, {}

    with zipfile.ZipFile(zip_file) as z:
        for file in z.namelist():
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'webp')):
                parts = file.split('/')
                if len(parts) < 2:
                    continue

                folder = parts[0].lower().strip()
                q_num = os.path.splitext(os.path.basename(file))[0].strip()

                with z.open(file) as f:
                    img = Image.open(f).convert("RGB")

                    if folder == "m1":
                        m1_imgs[q_num] = img
                    elif folder == "m2":
                        m2_imgs[q_num] = img

    return m1_imgs, m2_imgs


def create_student_pdf(name, m1_items, m2_items, doc_title, output_dir,
                       m1_ans=None, m2_ans=None):
    # m1_items / m2_items : [(문항번호, 이미지), ...]
    # m1_ans / m2_ans     : {문항번호: "학생 답"}  (없으면 캡션 생략)
    if not font_ready:
        return None

    m1_ans = m1_ans or {}
    m2_ans = m2_ans or {}

    pdf = KoreanPDF()
    pdf.add_page()
    pdf.set_font(pdf_font_name, style='B', size=10)

    pdf_cell_ln(pdf, 0, 8, f"<{name}_{doc_title}>")

    def add_images(title, items, ans_map):
        est_height = 80

        if items and (pdf.get_y() + 10 + est_height > pdf.page_break_trigger):
            pdf.add_page()

        pdf.set_font(pdf_font_name, size=10)
        pdf_cell_ln(pdf, 0, 8, title)

        if items:
            for qnum, img in items:
                temp_filename = f"temp_{datetime.now().timestamp()}_{os.urandom(4).hex()}.jpg"
                img.save(temp_filename)

                pdf.image(temp_filename, w=150)

                try:
                    os.remove(temp_filename)
                except:
                    pass

                # 문제 이미지 아래에 "내가 쓴 답" 캡션 (정답은 표시하지 않음)
                ans = str(ans_map.get(qnum, "")).strip()
                if ans:
                    pdf.set_font(pdf_font_name, style='B', size=9)
                    pdf_cell_ln(pdf, 0, 6, f"   \u21b3 \ub0b4\uac00 \uc4f4 \ub2f5: {ans}")
                    pdf.set_font(pdf_font_name, size=10)

                pdf.ln(8)
        else:
            pdf.ln(8)

    add_images("<Module1>", m1_items, m1_ans)
    add_images("<Module2>", m2_items, m2_ans)

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{name}_{doc_title}.pdf")
    pdf.output(pdf_path)

    return pdf_path


def _norm_qnum(t):
    """문항번호 정규화: '22.0' -> '22', ' 8 ' -> '8' (이미지/답 매칭용)"""
    t = str(t).strip()
    try:
        return str(int(float(t)))
    except Exception:
        return t


def load_wrong_answers(excel_bytes):
    """
    ETA 엑셀의 'Wrong Answers' 시트를 읽어 학생 답 lookup을 만든다.
    (ETA 앱스크립트의 '2-1. 오답(학생 답) 가져오기'가 생성하는 시트)
    반환: {(이름, 'M1'/'M2', 문항번호str): '학생 답'}  — 시트가 없으면 빈 dict
    """
    try:
        wa = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Wrong Answers")
    except Exception:
        return {}

    wa.columns = [str(c).strip() for c in wa.columns]

    def find_col(cands):
        for c in wa.columns:
            k = str(c).replace(" ", "")
            if any(k == x.replace(" ", "") for x in cands):
                return c
        return None

    name_c = find_col(["학생 이름", "이름", "name"])
    mod_c  = find_col(["모듈", "module"])
    q_c    = find_col(["문제 번호", "문항번호", "문제번호", "question"])
    ans_c  = find_col(["학생 답", "학생답", "useranswer", "답"])
    if not all([name_c, mod_c, q_c, ans_c]):
        return {}

    lookup = {}
    for _, r in wa.iterrows():
        name = str(r[name_c]).strip()
        mod  = str(r[mod_c]).strip().upper()
        qn   = _norm_qnum(r[q_c])
        ans  = str(r[ans_c]).strip()
        if name and mod and qn and ans and ans.lower() != "nan":
            lookup[(name, mod, qn)] = ans
    return lookup


MODULE_RE = re.compile(r"<\s*MODULE\s*(\d+)\s*>", re.IGNORECASE)
HEADER_FOOTER_HINT_RE = re.compile(
    r"(YOU,\s*GENIUS|700\+\s*MOCK\s*TEST|Kakaotalk|Instagram|010-\d{3,4}-\d{4}|Module\s*\d+|SECTION)",
    re.IGNORECASE,
)
PAGE_NUM_ONLY_RE = re.compile(r"^\s*\d{1,3}\s*$")
NUMDOT_RE = re.compile(r"^(\d{1,2})\.$")
NUM_RE = re.compile(r"^\d{1,2}$")
CHOICE_LABELS = ["D)", "C)", "B)", "A)"]
SIDE_PAD_PX = 10
INK_PAD_PX = 10
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
    if not words: return []
    lines = group_words_into_lines(words)
    anchors = []

    for tokens in lines:
        line_text = " ".join(t[4] for t in tokens).strip()
        compact = re.sub(r"\s+", "", line_text)
        if HEADER_FOOTER_HINT_RE.search(line_text): continue
        if len(compact) > max_line_chars: continue
        x_left = min(t[0] for t in tokens)
        if x_left > w_page * left_ratio: continue

        qnum = None
        y_top = None

        # case 1: "21."
        for (x0, y0, x1, y1, txt) in tokens:
            m = NUMDOT_RE.match(txt)
            if m:
                qnum = int(m.group(1))
                y_top = y0
                break

        # case 2: "21" "."  (words 분리)
        if qnum is None:
            for i in range(len(tokens) - 1):
                t1 = tokens[i][4]
                t2 = tokens[i + 1][4]
                if NUM_RE.match(t1) and t2 == ".":
                    qnum = int(t1)
                    y_top = tokens[i][1]
                    break

        if qnum is None: continue
        if not (1 <= qnum <= 22): continue
        anchors.append((qnum, y_top))

    anchors.sort(key=lambda t: t[1])
    return anchors

def band_text(page, clip):
    return (page.get_text("text", clip=clip) or "")

def last_choice_bottom_y_in_band(page, y_from, y_to):
    clip = fitz.Rect(0, y_from, page.rect.width, y_to)
    t = band_text(page, clip)
    if "A)" not in t: return None
    for lab in CHOICE_LABELS:
        rects = page.search_for(lab)
        bottoms = [r.y1 for r in rects if (r.y1 >= y_from and r.y0 <= y_to)]
        if bottoms: return max(bottoms)
    return None

def find_footer_start_y(page, y_from, y_to):
    ys = []
    page_h = page.rect.height

    # ✅ 페이지 하단 18% 영역만 footer 후보로 봄 (너무 공격적이면 0.85~0.88로 올려도 됨)
    bottom_zone_y = page_h * 0.82

    for b in page.get_text("blocks"):
        if len(b) < 5:
            continue

        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]

        if y0 < y_from or y0 > y_to:
            continue
        if not text:
            continue

        t = str(text).strip()

        # (1) 기존 footer 힌트(카톡/전화번호/브랜딩 등)
        if HEADER_FOOTER_HINT_RE.search(t):
            ys.append(y0)
            continue

        # (2) ✅ 추가: 페이지 하단 + 오른쪽에 있는 "숫자만" 블록(페이지번호)도 footer로 간주
        #     - 하단 영역(bottom_zone) AND
        #     - 오른쪽 영역(페이지 폭의 60% 이후) AND
        #     - 텍스트가 숫자만
        if (y0 >= bottom_zone_y) and (x0 >= page.rect.width * 0.60) and PAGE_NUM_ONLY_RE.match(t):
            ys.append(y0)

    return min(ys) if ys else None


def content_bottom_y(page, y_from, y_to):
    bottoms = []
    for b in page.get_text("blocks"):
        if len(b) < 5: continue
        y0, y1, text = b[1], b[3], b[4]
        if y1 < y_from or y0 > y_to: continue
        if text and HEADER_FOOTER_HINT_RE.search(str(text)): continue
        if text and str(text).strip():
            bottoms.append(y1)
    return max(bottoms) if bottoms else None

def text_x_bounds_in_band(page, y_from, y_to, min_len=2):
    xs0, xs1 = [], []
    for b in page.get_text("blocks"):
        if len(b) < 5: continue
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        if y1 < y_from or y0 > y_to: continue
        if not text: continue
        t = str(text).strip()
        if len(t) < min_len: continue
        if HEADER_FOOTER_HINT_RE.search(t): continue
        xs0.append(x0)
        xs1.append(x1)
    if not xs0: return None
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
                if x < minx: minx = x
                if y < miny: miny = y
                if x > maxx: maxx = x
                if y > maxy: maxy = y

    if maxx < 0: return None
    return (minx, miny, maxx, maxy, w, h)

def px_bbox_to_page_rect(clip, px_bbox, pad_px=INK_PAD_PX):
    minx, miny, maxx, maxy, w, h = px_bbox
    minx = max(0, minx - pad_px)
    miny = max(0, miny - pad_px)
    maxx = min(w - 1, maxx + pad_px)
    maxy = min(h - 1, maxy + pad_px)

    x0 = clip.x0 + (minx / (w - 1)) * (clip.x1 - clip.x0)
    x1 = clip.x0 + (maxx / (w - 1)) * (clip.x1 - clip.x0)
    y0 = clip.y0 + (miny / (h - 1)) * (clip.y1 - clip.y0)
    y1 = clip.y0 + (maxy / (h - 1)) * (clip.y1 - clip.y0)
    return fitz.Rect(x0, y0, x1, y1)

def render_png(page, clip, zoom):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    return pix.tobytes("png")

def expand_rect_to_width_right_only(rect, target_width, page_width):
    cur = rect.width
    if cur >= target_width: return rect
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
        if mid is not None: current_module = mid
        if current_module not in (1, 2): continue

        anchors = detect_question_anchors(page)
        if not anchors: continue

        for i, (qnum, y0) in enumerate(anchors):
            y_start_candidate = clamp(y0 - pad_top, 0, h)
            safe_y = y_start_candidate
            for b in page_blocks:
                b_y1 = b[3]
                b_text = b[4]
                if HEADER_FOOTER_HINT_RE.search(str(b_text) if b_text else ""):
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

            rects.append({
                "mod": current_module,
                "qnum": qnum,
                "page": pno,
                "rect": fitz.Rect(x0, y_start, x1, y_end),
                "page_width": w,
            })
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
# 메인 UI 구조
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["📝 오답노트 생성기", "✂️ 문제캡처 ZIP생성기", "📊 개인 성적표", "📈 개인 성적표(단원/난이도)"])


# ---------------------------------------------------------
# [Tab 1] 오답노트 생성기
# ---------------------------------------------------------
with tab1:
    st.header("📝 SAT 오답노트 생성기")

    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = []
    if 'zip_buffer' not in st.session_state:
        st.session_state.zip_buffer = None
    if 'skipped_details' not in st.session_state:
        st.session_state.skipped_details = {}

    st.markdown("---")
    st.subheader("📊 예시 엑셀 양식")

    with st.expander("예시 엑셀파일 미리보기 (클릭하여 열기)"):
        st.dataframe(example_input_df(), width="stretch")

    example = get_example_excel()
    st.download_button(
        "📥 예시 엑셀파일 다운로드 (.xlsx)",
        example,
        file_name="Mock결과_양식.xlsx"
    )

    st.markdown("---")
    st.header("📄 문서 제목 입력")
    doc_title = st.text_input(
        "문서 제목 (예: 25 S2 SAT MATH 만점반 Mock Test1, 26 6월대비 문풀반 MATH Mock1)",
        value="25 S2 SAT MATH 만점반 Mock Test1",
        key="t1_title"
    )

    st.header("📦 파일 업로드")

    st.markdown("#### 문제 이미지 ZIP 파일")
    img_zip = st.file_uploader("m1, m2 폴더가 들어있는 ZIP 파일", type="zip", key="t1_zip")

    st.markdown("#### 오답 현황 엑셀 파일")
    excel_file = st.file_uploader("학생들의 결과 데이터가 담긴 엑셀 파일", type="xlsx", key="t1_excel")

    if st.button("🚀 오답노트 생성 시작", type="primary", key="t1_btn"):
        if not img_zip or not excel_file:
            st.warning("⚠️ 이미지 ZIP 파일과 엑셀 파일을 모두 업로드해주세요.")
        else:
            try:
                m1_imgs, m2_imgs = extract_zip_to_dict(img_zip)

                excel_bytes = excel_file.getvalue() if hasattr(excel_file, "getvalue") else excel_file.read()
                raw = pd.read_excel(io.BytesIO(excel_bytes), header=1)
                df = normalize_columns(raw)

                # 'Wrong Answers' 시트가 있으면 학생 답 lookup 생성 (없으면 빈 dict → 캡션 생략)
                ans_lookup = load_wrong_answers(excel_bytes)

                required = {"이름", "Module1", "Module2"}
                missing = required - set(df.columns)

                if missing:
                    st.error(
                        f"필수 컬럼 누락: {missing}\n\n"
                        f"현재 인식된 컬럼: {list(df.columns)}\n\n"
                        f"엑셀 헤더는 반드시 다음 중 하나로 입력해주세요:\n"
                        f"- 학생 이름\n"
                        f"- [M1] 틀린 문제\n"
                        f"- [M2] 틀린 문제"
                    )
                    st.stop()

                output_dir = "generated_pdfs"
                os.makedirs(output_dir, exist_ok=True)

                temp_files = []
                skipped_details = {"만점": [], "M1/M2 하나 미제출": [], "미제출": []}
                progress_bar = st.progress(0)

                def parse_module_data(x):
                    if pd.isna(x):
                        return None

                    s = str(x).strip()

                    if s == "":
                        return None

                    if s.upper() in ["X", "Х", "-", "없음", "NONE", "NAN"]:
                        return []

                    s = (
                        s.replace("，", ",")
                         .replace(";", ",")
                         .replace(" ", "")
                    )

                    nums = [_norm_qnum(t) for t in s.split(",") if t.strip()]
                    return nums if nums else []

                for idx, row in df.iterrows():
                    name = str(row["이름"]).strip()

                    if not name or name.lower() == "nan":
                        progress_bar.progress((idx + 1) / len(df))
                        continue

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

                    m1_list = []
                    m2_list = []

                    if m1_data:
                        for n in m1_data:
                            if n in m1_imgs:
                                m1_list.append((n, m1_imgs[n]))

                    if m2_data:
                        for n in m2_data:
                            if n in m2_imgs:
                                m2_list.append((n, m2_imgs[n]))

                    # 이 학생의 문항별 '학생 답' 맵 (Wrong Answers 시트 기반)
                    m1_ans = {n: ans_lookup.get((name, "M1", n), "") for n in (m1_data or [])}
                    m2_ans = {n: ans_lookup.get((name, "M2", n), "") for n in (m2_data or [])}

                    pdf_path = create_student_pdf(
                        name, m1_list, m2_list, doc_title, output_dir,
                        m1_ans=m1_ans, m2_ans=m2_ans
                    )

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

    if st.session_state.generated_files or st.session_state.skipped_details:
        if st.session_state.skipped_details:
            total_skipped = sum(len(v) for v in st.session_state.skipped_details.values())

            if total_skipped > 0:
                with st.expander(f"📋 생성 제외 명단 (총 {total_skipped}명) - 클릭하여 보기", expanded=True):
                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown("**🏆 만점 (Perfect)**")
                        if st.session_state.skipped_details["만점"]:
                            for n in st.session_state.skipped_details["만점"]:
                                st.text(f"- {n}")
                        else:
                            st.caption("없음")

                    with c2:
                        st.markdown("**⚠️ 하나 미제출**")
                        if st.session_state.skipped_details["M1/M2 하나 미제출"]:
                            for n in st.session_state.skipped_details["M1/M2 하나 미제출"]:
                                st.text(f"- {n}")
                        else:
                            st.caption("없음")

                    with c3:
                        st.markdown("**❌ 미제출**")
                        if st.session_state.skipped_details["미제출"]:
                            for n in st.session_state.skipped_details["미제출"]:
                                st.text(f"- {n}")
                        else:
                            st.caption("없음")

        st.markdown("---")
        st.header("💾 다운로드")

        if st.session_state.zip_buffer:
            st.download_button(
                "📦 전체 오답노트 ZIP 다운로드",
                st.session_state.zip_buffer,
                file_name=f"오답노트_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
                key="t1_down_all"
            )

        st.subheader("👁️ 개별 PDF 다운로드")

        student_names = [name for name, _ in st.session_state.generated_files]

        if student_names:
            selected_student = st.selectbox("학생을 선택하세요", student_names, key="t1_select")

            if selected_student:
                file_map = {name: path for name, path in st.session_state.generated_files}
                target_path = file_map[selected_student]

                if os.path.exists(target_path):
                    with open(target_path, "rb") as f:
                        st.download_button(
                            f"📄 '{selected_student}' PDF 다운로드",
                            f,
                            file_name=f"{selected_student}_{doc_title}.pdf",
                            key="t1_down_indiv"
                        )

# ---------------------------------------------------------
# [Tab 2] PDF 문제 자르기 (✅ 원본 그대로)
# ---------------------------------------------------------
with tab2:
    st.header("✂️ 문제캡처 ZIP생성기")
    st.info("SAT Mock PDF를 업로드하면 문제 번호를 인식하여 개별 이미지(PNG)로 자르고 오답노트 생성기에 연동가능한 양식의 ZIP파일로 정리해줍니다")

    pdf_file = st.file_uploader("PDF 파일 업로드", type=["pdf"], key="t2_pdf")

    c1, c2, c3, c4 = st.columns(4)
    zoom_val = c1.slider("해상도(zoom)", 2.0, 4.0, 3.0, 0.1, key="t2_zoom")
    pt_val = c2.slider("위 여백(번호 포함)", 0, 140, 10, 1, key="t2_pt")
    pb_val = c3.slider("아래 여백(다음 문제 전)", 0, 200, 12, 1, key="t2_pb")
    frq_val = c4.slider("단답형 아래 여백(px)", 0, 600, 250, 25, key="t2_frq")



    if pdf_file:
        if st.button("✂️ 자르기 & ZIP 생성", type="primary", key="t2_btn"):
            with st.spinner("PDF 분석 및 이미지 생성 중... (시간이 조금 걸릴 수 있습니다)"):
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
                    count_m1 = sum(1 for r in rects_data if r['mod'] == 1)
                    count_m2 = sum(1 for r in rects_data if r['mod'] == 2)
                    
                    zbuf_data, zname = make_zip_from_rects(
                        doc_obj,
                        rects_data,
                        zoom_val,
                        zip_base,
                        unify_width_right=True,  # [수정] UI 없이 항상 True로 고정
                    )

                    st.success(f"✅ 처리 완료! (총 {len(rects_data)}문제: M1 {count_m1}개 / M2 {count_m2}개)")
                    st.download_button(
                        "📦 ZIP 다운로드",
                        data=zbuf_data,
                        file_name=zname,
                        mime="application/zip",
                        key="t2_down"
                    )
                except Exception as e:
                    st.error(f"오류 발생: {e}")

# ---------------------------------------------------------
# [Tab 3] 개인 성적표 (✅ 여기만 개선/수정)
# ---------------------------------------------------------
with tab3:
    st.header("📊 개인 성적표")
    st.info("Student Analysis(학생목록) + QuizResults + (Accuracy Analysis/ Error Analysis 정답률) + Mock데이터(정답)")

    eta_file = st.file_uploader("ETA 결과 파일 업로드 (ETA.xlsx)", type=["xlsx"], key="t3_eta")
    mock_file = st.file_uploader("Mock 정답 파일 업로드 (Mock데이터.xlsx)", type=["xlsx"], key="t3_mock")

    c1, c2 = st.columns([1, 1])
    with c1:
        report_title = st.text_input("리포트 제목", value="SAT Math Report", key="t3_title")
    with c2:
        generated_date = st.date_input("Generated 날짜", value=datetime.now().date(), key="t3_gen_date")

    st.caption("부제목은 QuizResults의 '검색 키워드'가 학생별로 자동으로 들어갑니다.")

    STUDENT_SHEET = "Student Analysis"
    QUIZ_SHEET = "QuizResults"

    SA_HEADER_ROW_IDX = 1
    QZ_HEADER_ROW_IDX = 0

    SA_NAME_COL = "학생 이름"
    SA_M1_SCORE_COL = "[M1] 점수"
    SA_M2_SCORE_COL = "[M2] 점수"

    QZ_KEYWORD_COL = "검색 키워드"
    QZ_MODULE_COL  = "모듈"
    QZ_NAME_COL    = "학생 이름"
    QZ_DT_COL      = "응답 날짜"
    QZ_TIME_COL    = "소요 시간"
    QZ_SCORE_COL   = "점수"
    QZ_WRONG_COL   = "틀린 문제 번호"

    FOOTER_LEFT_TEXT = "Kakaotalk: yujinj524\nPhone: 010-6395-8733"

    def _clean(x):
        if x is None: return ""
        if isinstance(x, float) and pd.isna(x): return ""
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

    def build_wrong_rate_dict_fixed_ranges(eta_xl, sheet_name):
        df = pd.read_excel(eta_xl, sheet_name=sheet_name, header=None)
        colC = df.iloc[:, 2].tolist()

        m1_vals = colC[2:24]
        m2_vals = colC[25:47]

        def to_dict(vals):
            out = {}
            for i, v in enumerate(vals, start=1):
                try:
                    out[i] = float(v)
                except:
                    out[i] = None
            return out

        return to_dict(m1_vals), to_dict(m2_vals)

    def read_mock_answers(mock_bytes) -> tuple[dict, dict]:
        df = pd.read_excel(mock_bytes)
        cols = set(df.columns.astype(str))

        if {"모듈", "문항번호", "정답"}.issubset(cols):
            m1 = df[df["모듈"].astype(str).str.upper().eq("M1")].set_index("문항번호")["정답"].astype(str).to_dict()
            m2 = df[df["모듈"].astype(str).str.upper().eq("M2")].set_index("문항번호")["정답"].astype(str).to_dict()
            m1 = {int(k): _clean(v) for k, v in m1.items() if str(k).strip().isdigit()}
            m2 = {int(k): _clean(v) for k, v in m2.items() if str(k).strip().isdigit()}
            return m1, m2

        c0, c1 = df.columns[0], df.columns[1]
        m2_idxs = df.index[df[c0].astype(str).str.contains("Module2", case=False, na=False)].tolist()
        if not m2_idxs:
            out = {}
            for _, r in df.iterrows():
                try: q = int(str(r[c0]).strip())
                except: continue
                out[q] = _clean(r[c1])
            return out, {}

        m2i = m2_idxs[0]
        m1_rows = df.iloc[:m2i]
        m2_rows = df.iloc[m2i+1:]

        def rows_to_ans(rows):
            dct = {}
            for _, r in rows.iterrows():
                try: q = int(str(r[c0]).strip())
                except: continue
                dct[q] = _clean(r[c1])
            return dct

        return rows_to_ans(m1_rows), rows_to_ans(m2_rows)

    # ===== ReportLab PDF + PNG 렌더링 =====
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
            ln = (ln or "").strip()
            if ln == "":
                continue
            need = min(need, fit_font_size(ln, font_name, max_size, min_size, max_width))
        return need

    def draw_round_rect(c, x, y, w, h, r, fill, stroke, stroke_width=1):
        c.setLineWidth(stroke_width)
        c.setStrokeColor(stroke)
        c.setFillColor(fill)
        c.roundRect(x, y, w, h, r, fill=1, stroke=1)

    def wr_to_text(v):
        if v is None:
            return "-"
        try:
            v = float(v)
            return f"{int(round(v * 100))}%"
        except:
            return "-"

    # -------------------------------------------------------------
    # [수정된 함수] 제목 제거, 테이블 위로 이동, 헤더 축소, KPI 줄 제거,
    #              헤더와 1행 사이 간격 축소 유지
    # -------------------------------------------------------------
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
        result_blank: bool = False,
        footer_left_text: str = "",
    ):
        ensure_fonts_registered()
        c = canvas.Canvas(output_path, pagesize=A4)
        W, H = A4

        # colors
        stroke = colors.Color(203/255, 213/255, 225/255)
        title_col = colors.Color(15/255, 23/255, 42/255)
        muted = colors.Color(100/255, 116/255, 139/255)
        pill_fill = colors.Color(241/255, 245/255, 249/255)
        row_stripe = colors.Color(248/255, 250/255, 252/255)
        green = colors.Color(22/255, 101/255, 52/255)
        red = colors.Color(220/255, 38/255, 38/255)

        # layout
        L = 15 * mm
        R = 15 * mm
        TOP = H - 28 * mm
        usable_w = W - L - R

        # Generated
        c.setFont("NanumGothic", 10)
        c.setFillColor(muted)
        c.drawRightString(W - R, TOP + 16*mm, f"Generated: {gen_date_str}")

        # Title / subtitle
        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 30)
        c.drawString(L, TOP, title)

        c.setFillColor(muted)
        c.setFont("NanumGothic", 14)
        c.drawString(L, TOP - 11*mm, subtitle)

        # Name pill
        pill_w = 78 * mm
        pill_h = 20 * mm
        pill_x = L + usable_w - pill_w
        pill_y = TOP - 12 * mm
        draw_round_rect(c, pill_x, pill_y, pill_w, pill_h, 10*mm, pill_fill, stroke, 1)

        c.setFillColor(muted)
        c.setFont("NanumGothic-Bold", 10)
        c.drawString(pill_x + 7*mm, pill_y + 12.2*mm, "Name")

        c.setFillColor(title_col)
        max_name_w = pill_w - 26 * mm
        name_fs = fit_font_size(student_name, "NanumGothic-Bold", 16, 10, max_name_w)
        c.setFont("NanumGothic-Bold", name_fs)
        c.drawRightString(pill_x + pill_w - 7*mm, pill_y + 6.0*mm, student_name)

        # divider
        line_y = TOP - 22 * mm
        c.setLineWidth(2)
        c.setStrokeColor(title_col)
        c.line(L, line_y, W - R, line_y)

        # KPI
        kpi_h = 30 * mm
        gap = 10 * mm
        kpi_w = (usable_w - gap) / 2
        kpi_gap_from_line = 7 * mm
        kpi_y = line_y - kpi_gap_from_line - kpi_h

        def draw_kpi_card(x, y, w, h, label, score, dt, t):
            draw_round_rect(c, x, y, w, h, 8*mm, colors.white, stroke, 1)

            c.setFillColor(title_col)
            c.setFont("NanumGothic-Bold", 16)
            c.drawString(x + 8*mm, y + h - 11*mm, label)

            c.setFont("NanumGothic-Bold", 28)
            c.drawRightString(x + w - 8*mm, y + h - 16.5*mm, str(score))

            c.setFillColor(muted)
            c.setFont("NanumGothic", 8)
            c.drawString(x + 8*mm, y + 4.8*mm, f"{dt}")
            c.drawRightString(x + w - 8*mm, y + 4.8*mm, f"{t}")

        draw_kpi_card(L, kpi_y, kpi_w, kpi_h, "Module 1", m1_meta["score"], m1_meta["dt"], m1_meta["time"])
        draw_kpi_card(L + kpi_w + gap, kpi_y, kpi_w, kpi_h, "Module 2", m2_meta["score"], m2_meta["dt"], m2_meta["time"])

        # [수정] 테이블 사이즈 및 레이아웃 설정
        header_h = 6.0 * mm    # 헤더 높이 축소
        row_h = 5.6 * mm
        top_padding = 5.0 * mm # 제목 제거로 상단 여백 축소
        bottom_padding = 6.0 * mm
        
        # 전체 카드 높이 계산
        card_h = top_padding + header_h + (22 * row_h) + bottom_padding
        
        # 카드 위치 (KPI 아래로 바짝 붙임)
        card_y = kpi_y - 4 * mm - card_h 

        card_w = kpi_w
        left_x = L
        right_x = L + card_w + gap

        def draw_table(x, y, w, h, module_name, ans_dict, wr_dict, wrong_set):
            draw_round_rect(c, x, y, w, h, 10*mm, colors.white, stroke, 1)

            # [수정] 헤더 위치 조정
            strip_y = y + h - top_padding - header_h
            strip_h = header_h
            
            c.setLineWidth(1)
            c.setStrokeColor(stroke)
            c.setFillColor(pill_fill)
            c.rect(x + 6*mm, strip_y, w - 12*mm, strip_h, stroke=1, fill=1)

            inner_x = x + 8 * mm
            inner_w = w - 16 * mm

            col_q = 10 * mm
            col_ans = 26 * mm
            col_wr = 20 * mm
            col_res = inner_w - (col_q + col_ans + col_wr)

            q_center = inner_x + col_q / 2
            ans_center = inner_x + col_q + col_ans / 2
            wr_center = inner_x + col_q + col_ans + col_wr / 2
            res_center = inner_x + col_q + col_ans + col_wr + col_res / 2

            # [수정] 헤더 텍스트 위치 미세 조정
            header_text_y = strip_y + 1.8 * mm
            
            c.setFillColor(muted)
            c.setFont("NanumGothic-Bold", 9.5)
            c.drawCentredString(q_center, header_text_y, "No.")
            c.drawCentredString(ans_center, header_text_y, "Answer")
            c.drawCentredString(wr_center, header_text_y, "정답률")
            c.drawCentredString(res_center, header_text_y, "Result")

            # [수정] 헤더와 1행 사이 간격 축소 (2.0mm -> 0.5mm)
            start_y = strip_y - 0.5*mm - row_h
            base = 1.35 * mm

            for i, q in enumerate(range(1, 23)):
                ry = start_y - i * row_h

                if q % 2 == 0:
                    c.setFillColor(row_stripe)
                    c.setStrokeColor(row_stripe)
                    c.rect(x + 6*mm, ry, w - 12*mm, row_h, stroke=0, fill=1)

                ans_raw = _clean(ans_dict.get(q, ""))
                lines = ans_raw.split("\n") if "\n" in ans_raw else [ans_raw]
                lines = [ln.strip() for ln in lines if ln.strip()]
                if not lines:
                    lines = [""]

                if len(lines) > 2:
                    lines = [lines[0], " ".join(lines[1:])]

                rate_val = wr_dict.get(q, None)
                wr_txt = wr_to_text(rate_val)

                if result_blank:
                    res_txt = ""
                else:
                    res_txt = "X" if q in wrong_set else "O"

                # No.
                c.setFillColor(title_col)
                c.setFont("NanumGothic", 10.0)
                c.drawCentredString(q_center, ry + base, str(q))

                # Answer
                ans_max_w = col_ans - 3*mm
                fs = fit_font_size_two_lines(lines, "NanumGothic-Bold", 10.0, 7.0, ans_max_w)
                c.setFont("NanumGothic-Bold", fs)
                if len(lines) == 1:
                    c.drawCentredString(ans_center, ry + base, lines[0])
                else:
                    c.drawCentredString(ans_center, ry + base + 0.7*mm, lines[0])
                    c.drawCentredString(ans_center, ry + base - 0.7*mm, lines[1])

                # 정답률
                is_low = False
                try:
                    if rate_val is not None and float(rate_val) < 0.5:
                        is_low = True
                except:
                    pass

                c.setFillColor(title_col)
                if is_low:
                    c.setFont("NanumGothic-Bold", 10.3)
                else:
                    c.setFont("NanumGothic", 10.0)
                c.drawCentredString(wr_center, ry + base, wr_txt)

                # Result
                if res_txt:
                    ox_color = red if res_txt == "X" else green
                    c.setFillColor(ox_color)
                    c.setFont("NanumGothic-Bold", 11.0)
                    c.drawCentredString(res_center, ry + base, res_txt)

        draw_table(left_x, card_y, card_w, card_h, "Module 1", ans_m1, wr_m1, wrong_m1)
        draw_table(right_x, card_y, card_w, card_h, "Module 2", ans_m2, wr_m2, wrong_m2)

        # footer
        if footer_left_text:
            c.setFillColor(title_col)
            c.setFont("NanumGothic", 8)
            lines = str(footer_left_text).splitlines()
            y0 = 12 * mm
            line_gap = 4.2 * mm
            for idx, ln in enumerate(lines):
                c.drawString(L, y0 + (len(lines)-1-idx)*line_gap, ln)

        c.showPage()
        c.save()
        return output_path

    def render_pdf_first_page_to_png_bytes(pdf_path: str, zoom: float = 2.0) -> bytes:
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")

    if st.button("🚀 개인 성적표 생성", type="primary", key="t3_btn"):
        if not eta_file or not mock_file:
            st.warning("⚠️ ETA.xlsx와 Mock데이터.xlsx를 모두 업로드해주세요.")
            st.stop()

        if not font_ready:
            st.error("⚠️ 한글 PDF 생성을 위해 fonts 폴더에 NanumGothic.ttf / NanumGothicBold.ttf가 필요합니다.")
            st.stop()

        try:
            eta_xl = pd.ExcelFile(eta_file)

            # ---- Student Analysis ----
            if STUDENT_SHEET not in eta_xl.sheet_names:
                st.error(f"⚠️ ETA.xlsx에 '{STUDENT_SHEET}' 시트가 없습니다.")
                st.stop()

            raw_sa = pd.read_excel(eta_xl, sheet_name=STUDENT_SHEET, header=None)
            if raw_sa.shape[0] <= SA_HEADER_ROW_IDX:
                st.error("⚠️ Student Analysis에서 2행(헤더)을 찾을 수 없습니다.")
                st.stop()

            sa_header = raw_sa.iloc[SA_HEADER_ROW_IDX].astype(str).tolist()
            student_df = raw_sa.iloc[SA_HEADER_ROW_IDX + 1:].copy()
            student_df.columns = sa_header
            student_df = student_df.dropna(axis=1, how="all").dropna(axis=0, how="all")

            assert_columns(student_df, [SA_NAME_COL, SA_M1_SCORE_COL, SA_M2_SCORE_COL], STUDENT_SHEET)

            students = [_clean(x) for x in student_df[SA_NAME_COL].dropna().tolist()]
            students = [s for s in students if s != ""]
            if not students:
                st.error("학생 목록이 비어있습니다.")
                st.stop()

            # ---- QuizResults ----
            if QUIZ_SHEET not in eta_xl.sheet_names:
                st.error(f"⚠️ ETA.xlsx에 '{QUIZ_SHEET}' 시트가 없습니다.")
                st.stop()

            quiz_df = pd.read_excel(eta_xl, sheet_name=QUIZ_SHEET, header=QZ_HEADER_ROW_IDX)
            quiz_df.columns = [str(c).strip() for c in quiz_df.columns]
            quiz_df = quiz_df.dropna(axis=1, how="all").dropna(axis=0, how="all")

            assert_columns(
                quiz_df,
                [QZ_KEYWORD_COL, QZ_MODULE_COL, QZ_NAME_COL, QZ_DT_COL, QZ_TIME_COL, QZ_SCORE_COL, QZ_WRONG_COL],
                QUIZ_SHEET
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

            # ---- Accuracy / Error Analysis (정답률) ----
            target_sheet = None
            if "Accuracy Analysis" in eta_xl.sheet_names:
                target_sheet = "Accuracy Analysis"
            elif "Error Analysis" in eta_xl.sheet_names:
                target_sheet = "Error Analysis"

            if target_sheet:
                wr1, wr2 = build_wrong_rate_dict_fixed_ranges(eta_xl, target_sheet)
            else:
                wr1, wr2 = {}, {}

            # ---- Mock Answers ----
            ans1, ans2 = read_mock_answers(mock_file)

            # ---- PDF 생성 ----
            output_dir = "generated_reports"
            os.makedirs(output_dir, exist_ok=True)

            made_files = []
            made_images = []
            skipped = []
            prog = st.progress(0)

            # [추가] 템플릿용 공통 부제목 저장 변수
            common_subtitle = "-"

            for i, stu in enumerate(students):
                q = quiz_map.get(stu, {})
                m1 = q.get(1, {})
                m2 = q.get(2, {})

                m1_score_txt = _clean(m1.get("score", ""))
                m2_score_txt = _clean(m2.get("score", ""))

                if m1_score_txt == "" or m2_score_txt == "":
                    skipped.append(stu)
                    prog.progress((i+1)/len(students))
                    continue

                subtitle_kw = _clean(m1.get("keyword", "")) or _clean(m2.get("keyword", "")) or "-"
                
                # [추가] 유효한 키워드가 있으면 템플릿용으로 저장
                if subtitle_kw != "-" and common_subtitle == "-":
                    common_subtitle = subtitle_kw

                m1_meta = {"score": m1_score_txt, "dt": m1.get("dt", "-"), "time": m1.get("time", "-")}
                m2_meta = {"score": m2_score_txt, "dt": m2.get("dt", "-"), "time": m2.get("time", "-")}

                wrong1 = set(m1.get("wrong_set", set()))
                wrong2 = set(m2.get("wrong_set", set()))

                pdf_path = os.path.join(output_dir, f"{stu}_{generated_date.strftime('%Y%m%d')}.pdf")

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
                    footer_left_text=FOOTER_LEFT_TEXT,
                )

                made_files.append((stu, pdf_path))

                # PNG (첫 페이지만)
                try:
                    png_bytes = render_pdf_first_page_to_png_bytes(pdf_path, zoom=2.0)
                    png_path = os.path.join(output_dir, f"{stu}_{generated_date.strftime('%Y%m%d')}.png")
                    with open(png_path, "wb") as f:
                        f.write(png_bytes)
                    made_images.append((stu, png_path))
                except:
                    pass

                prog.progress((i+1)/len(students))

            # ---- 템플릿 1개 추가 (Name='-', Result 빈칸) ----
            # [수정] 파일명 __TEMPLATE__ -> Report_ 로 변경
            template_pdf = os.path.join(output_dir, f"Report_{generated_date.strftime('%Y%m%d')}.pdf")
            create_report_pdf_reportlab(
                output_path=template_pdf,
                title=report_title,
                subtitle=common_subtitle, 
                gen_date_str=generated_date.strftime("%Y-%m-%d"),
                student_name="-",
                m1_meta={"score": "-", "dt": "-", "time": "-"},
                m2_meta={"score": "-", "dt": "-", "time": "-"},
                ans_m1=ans1,
                ans_m2=ans2,
                wr_m1=wr1,
                wr_m2=wr2,
                wrong_m1=set(),
                wrong_m2=set(),
                result_blank=True,
                footer_left_text=FOOTER_LEFT_TEXT,
            )
            made_files.append(("Report", template_pdf))

            try:
                png_bytes = render_pdf_first_page_to_png_bytes(template_pdf, zoom=2.0)
                template_png = os.path.join(output_dir, f"Report_{generated_date.strftime('%Y%m%d')}.png")
                with open(template_png, "wb") as f:
                    f.write(png_bytes)
                made_images.append(("Report", template_png))
            except:
                pass

            if not made_files:
                st.warning("생성된 PDF가 없습니다. (QuizResults 점수 blank로 모두 제외되었을 수 있어요)")
                if skipped:
                    with st.expander(f"제외된 학생 ({len(skipped)}명) - 점수 blank"):
                        for s in skipped:
                            st.write(f"- {s}")
                st.stop()

            # ---- PDF ZIP ----
            pdf_zip_buf = io.BytesIO()
            with zipfile.ZipFile(pdf_zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for stu, path in made_files:
                    if os.path.exists(path):
                        z.write(path, arcname=os.path.basename(path))
            pdf_zip_buf.seek(0)

            # ---- PNG ZIP ----
            img_zip_buf = io.BytesIO()
            with zipfile.ZipFile(img_zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for stu, path in made_images:
                    if os.path.exists(path):
                        z.write(path, arcname=os.path.basename(path))
            img_zip_buf.seek(0)

            st.success(f"✅ 생성 완료: PDF {len(made_files)}개 / 이미지 {len(made_images)}개 (제외: {len(skipped)}명)")
            if skipped:
                with st.expander(f"제외된 학생 ({len(skipped)}명) - 점수 blank"):
                    for s in skipped:
                        st.write(f"- {s}")

            st.download_button(
                "📦 개인 성적표 PDF ZIP 다운로드",
                data=pdf_zip_buf,
                file_name=f"개인성적표_PDF_{generated_date.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="t3_download_pdf_zip"
            )

            st.download_button(
                "🖼️ 개인 성적표 이미지(PNG) ZIP 다운로드",
                data=img_zip_buf,
                file_name=f"개인성적표_PNG_{generated_date.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="t3_download_png_zip"
            )

        except ModuleNotFoundError as e:
            st.error("❌ reportlab이 설치되어 있지 않습니다. (requirements.txt에 reportlab 추가 필요)")
            st.exception(e)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.exception(e)


# ---------------------------------------------------------
# [Tab 4] 개인 성적표(단원/난이도)  ✅ Tab3는 건드리지 않고 Tab4만 추가
# ---------------------------------------------------------
with tab4:
    st.header("📈 개인 성적표(단원/난이도)")
    st.info("Tab3 성적표 + (Mock데이터.xlsx의 '단원', '난이도' 컬럼) → 문항표에 난이도/단원 추가 + 하단 Topic 막대그래프 + Difficulty Accuracy")

    # ---- 업로드/UI (Tab3와 키 충돌 방지: t4_ prefix) ----
    eta_file4  = st.file_uploader("ETA 결과 파일 업로드 (ETA.xlsx)", type=["xlsx"], key="t4_eta")
    mock_file4 = st.file_uploader("Mock 정답/메타 파일 업로드 (Mock데이터.xlsx: 정답+단원+난이도)", type=["xlsx"], key="t4_mock")

    c1, c2 = st.columns([1, 1])
    with c1:
        report_title4 = st.text_input("리포트 제목", value="SAT Math Report", key="t4_title")
    with c2:
        generated_date4 = st.date_input("Generated 날짜", value=datetime.now().date(), key="t4_gen_date")

    st.caption("부제목은 QuizResults의 '검색 키워드'가 학생별로 자동으로 들어갑니다. (Tab3과 동일)")

    # ====== 상수 (Tab3과 동일) ======
    STUDENT_SHEET = "Student Analysis"
    QUIZ_SHEET = "QuizResults"

    SA_HEADER_ROW_IDX = 1
    QZ_HEADER_ROW_IDX = 0

    SA_NAME_COL = "학생 이름"
    SA_M1_SCORE_COL = "[M1] 점수"
    SA_M2_SCORE_COL = "[M2] 점수"

    QZ_KEYWORD_COL = "검색 키워드"
    QZ_MODULE_COL  = "모듈"
    QZ_NAME_COL    = "학생 이름"
    QZ_DT_COL      = "응답 날짜"
    QZ_TIME_COL    = "소요 시간"
    QZ_SCORE_COL   = "점수"
    QZ_WRONG_COL   = "틀린 문제 번호"

    FOOTER_LEFT_TEXT = "Kakaotalk: yujinj524\nPhone: 010-6395-8733"

    # ====== 유틸 (Tab4 내부 재정의: Tab3 scope 의존 제거) ======
    def _clean4(x):
        if x is None: return ""
        if isinstance(x, float) and pd.isna(x): return ""
        return str(x).replace("\r", "").strip()

    def parse_wrong_list4(val):
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

    def score_to_slash224(s):
        s = _clean4(s)
        if s == "":
            return ""
        if "/" in s:
            return s
        return f"{s} / 22"

    def assert_columns4(df, cols, label):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            st.error(f"⚠️ {label} 컬럼 누락: {missing}")
            st.write(f"현재 {label} 컬럼:", list(df.columns))
            st.stop()

    def build_wrong_rate_dict_fixed_ranges4(eta_xl, sheet_name):
        df = pd.read_excel(eta_xl, sheet_name=sheet_name, header=None)
        colC = df.iloc[:, 2].tolist()

        m1_vals = colC[2:24]
        m2_vals = colC[25:47]

        def to_dict(vals):
            out = {}
            for i, v in enumerate(vals, start=1):
                try:
                    out[i] = float(v)
                except:
                    out[i] = None
            return out

        return to_dict(m1_vals), to_dict(m2_vals)

    # ---- Mock데이터: 정답 + 단원 + 난이도 읽기 ----
    # 기대 컬럼: 모듈, 문항번호, 정답, 단원, 난이도
    def read_mock_answers_with_meta(mock_bytes):
        df = pd.read_excel(mock_bytes)
        df.columns = [str(c).strip() for c in df.columns]

        need = {"모듈", "문항번호", "정답", "단원", "난이도"}
        if not need.issubset(set(df.columns)):
            st.error(f"⚠️ Mock데이터.xlsx에 다음 컬럼이 필요합니다: {sorted(list(need))}")
            st.write("현재 컬럼:", list(df.columns))
            st.stop()

        def norm_mod(v):
            s = str(v).strip().upper()
            if s in ["M1", "MODULE1", "1"]: return 1
            if s in ["M2", "MODULE2", "2"]: return 2
            return None

        ans = {1: {}, 2: {}}
        topic = {1: {}, 2: {}}
        diff = {1: {}, 2: {}}

        for _, r in df.iterrows():
            md = norm_mod(r.get("모듈"))
            if md not in (1, 2):
                continue
            q_raw = r.get("문항번호")
            try:
                q = int(float(str(q_raw).strip()))
            except:
                continue
            if not (1 <= q <= 22):
                continue

            ans[md][q] = _clean4(r.get("정답"))
            topic[md][q] = _clean4(r.get("단원"))      # 예: "5.3"
            diff[md][q] = _clean4(r.get("난이도")).upper()  # E/M/H

        return ans[1], ans[2], topic[1], topic[2], diff[1], diff[2]

    # ---- 정답률 텍스트 ----
    def wr_to_text4(v):
        if v is None:
            return "-"
        try:
            v = float(v)
            return f"{int(round(v * 100))}%"
        except:
            return "-"

    # ---- 단원 major(1~7) ----
    TOPIC_NAMES = {
        1: "1. Linear",
        2: "2. Percent & Unit Conversion",
        3: "3. Quadratic",
        4: "4. Exponential",
        5: "5. Polynomials, radical\nand rational functions",
        6: "6. Geometry",
        7: "7. Statistics",
    }

    def major_topic_id(topic_str):
        s = str(topic_str).strip()
        if s == "" or s.lower() == "nan":
            return None
        m = re.match(r"^\s*(\d+)", s)
        if not m:
            return None
        v = int(m.group(1))
        return v if 1 <= v <= 7 else None

    # ---- Topic/난이도 집계 ----
    def build_topic_rows(items):
        stats = {k: {"c": 0, "t": 0} for k in range(1, 8)}
        for it in items:
            tid = major_topic_id(it.get("topic"))
            if tid is None:
                continue
            stats[tid]["t"] += 1
            if it.get("is_correct") is True:
                stats[tid]["c"] += 1

        rows = []
        for tid in range(1, 8):
            c = stats[tid]["c"]
            t = stats[tid]["t"]
            acc = (c / t) if t else 0.0
            rows.append((tid, TOPIC_NAMES[tid], c, t, acc))
        return rows

    def build_difficulty_summary(items):
        # returns dict { "E":(c,t,acc), "M":..., "H":... }
        stats = {k: {"c": 0, "t": 0} for k in ["E", "M", "H"]}
        for it in items:
            d = str(it.get("diff") or "").strip().upper()
            if d not in stats:
                continue
            stats[d]["t"] += 1
            if it.get("is_correct") is True:
                stats[d]["c"] += 1
        out = {}
        for d in ["E", "M", "H"]:
            c = stats[d]["c"]
            t = stats[d]["t"]
            acc = (c / t) if t else 0.0
            out[d] = (c, t, acc)
        return out

    # ===== ReportLab (Tab4에서 다시 import) =====
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    def ensure_fonts_registered4():
        try:
            pdfmetrics.registerFont(TTFont("NanumGothic", FONT_REGULAR))
        except:
            pass
        try:
            pdfmetrics.registerFont(TTFont("NanumGothic-Bold", FONT_BOLD))
        except:
            pass

    def str_w4(text, font_name, font_size):
        return pdfmetrics.stringWidth(text, font_name, font_size)

    def fit_font_size4(text, font_name, max_size, min_size, max_width):
        s = max_size
        while s >= min_size:
            if str_w4(text, font_name, s) <= max_width:
                return s
            s -= 0.5
        return min_size

    def fit_font_size_two_lines4(lines, font_name, max_size, min_size, max_width):
        need = max_size
        for ln in lines:
            ln = (ln or "").strip()
            if ln == "":
                continue
            need = min(need, fit_font_size4(ln, font_name, max_size, min_size, max_width))
        return need

    def draw_round_rect4(c, x, y, w, h, r, fill, stroke, stroke_width=1):
        c.setLineWidth(stroke_width)
        c.setStrokeColor(stroke)
        c.setFillColor(fill)
        c.roundRect(x, y, w, h, r, fill=1, stroke=1)

    # ---- Tab4: Topic Panel (캡처 스타일: 회색 remainder + 파란 fill + % 막대 안) ----
    def draw_topic_panel_domain_style4(c, x, y, w, h, topic_rows, diff_summary, title="Topic"):
        stroke    = colors.Color(203/255, 213/255, 225/255)
        title_col = colors.Color(15/255, 23/255, 42/255)
        muted     = colors.Color(100/255, 116/255, 139/255)

        bg_bar    = colors.Color(226/255, 232/255, 240/255)  # remainder
        fill_bar  = colors.Color(191/255, 219/255, 254/255)  # fill
        diff_box_bg = colors.Color(250/255, 250/255, 250/255)

        # card
        draw_round_rect4(c, x, y, w, h, 10*mm, colors.white, stroke, 1)

        pad = 8*mm
        inner_x = x + pad
        inner_w = w - 2*pad

        # header row
        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 14)
        c.drawString(inner_x, y + h - pad - 4*mm, title)

        # difficulty box (우측)
        box_w = 50*mm
        box_h = 22*mm
        box_x = x + w - pad - box_w
        box_y = y + h - pad - box_h - 1*mm

        c.setLineWidth(1)
        c.setStrokeColor(stroke)
        c.setFillColor(diff_box_bg)
        c.roundRect(box_x, box_y, box_w, box_h, 3*mm, stroke=1, fill=1)

        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 9.5)
        c.drawString(box_x + 4*mm, box_y + box_h - 7*mm, "Difficulty")

        c.setFont("NanumGothic", 9)
        c.setFillColor(muted)

        def _diff_line(d, yline):
            c_cnt, t_cnt, acc = diff_summary.get(d, (0, 0, 0.0))
            pct = int(round(acc*100)) if t_cnt else 0
            c.setFillColor(colors.Color(90/255, 127/255, 170/255))
            c.setFont("NanumGothic-Bold", 9)
            c.drawString(box_x + 4*mm, yline, d)

            c.setFillColor(title_col)
            c.setFont("NanumGothic", 9)
            c.drawString(box_x + 10*mm, yline, f"{pct}%")

            c.setFillColor(muted)
            c.setFont("NanumGothic", 8.5)
            c.drawRightString(box_x + box_w - 4*mm, yline, f"({c_cnt}/{t_cnt})")

        _diff_line("E", box_y + 9.5*mm)
        _diff_line("M", box_y + 5.5*mm)
        _diff_line("H", box_y + 1.5*mm)

        # rows layout
        top_gap = 12*mm
        row_top = y + h - pad - top_gap
        row_bottom = y + pad
        n = max(1, len(topic_rows))
        row_h = (row_top - row_bottom) / n
        row_h = min(row_h, 10.5*mm)

        # columns
        col_label = 78*mm
        col_score = 18*mm
        col_bar = inner_w - col_label - col_score
        if col_bar < 40*mm:
            col_label = max(55*mm, col_label - (40*mm - col_bar))
            col_bar = inner_w - col_label - col_score

        label_x = inner_x
        bar_x0  = inner_x + col_label
        score_x = inner_x + col_label + col_bar

        bar_h = 6.5*mm
        r = 1.2*mm
        bar_w = col_bar - 3*mm

        def _clamp01(v):
            try:
                return max(0.0, min(1.0, float(v)))
            except:
                return 0.0

        for i, (_, label, correct, total, acc) in enumerate(topic_rows):
            ry = row_top - (i+1)*row_h + (row_h - bar_h)/2

            # label
            c.setFillColor(title_col)
            c.setFont("NanumGothic", 11)
            lab = str(label)
            if "\n" in lab:
                a, b = lab.split("\n", 1)
                c.drawString(label_x, ry + bar_h + 1.2*mm, a)
                c.drawString(label_x, ry + bar_h - 3.3*mm, b)
            else:
                c.drawString(label_x, ry + 1.2*mm, lab)

            # background (remainder)
            c.setFillColor(bg_bar)
            c.setStrokeColor(bg_bar)
            c.roundRect(bar_x0, ry, bar_w, bar_h, r, stroke=0, fill=1)

            # fill
            acc = _clamp01(acc)
            fill_w = bar_w * acc
            if fill_w > 0:
                c.setFillColor(fill_bar)
                c.setStrokeColor(fill_bar)
                c.roundRect(bar_x0, ry, fill_w, bar_h, r, stroke=0, fill=1)

            # % inside bar
            pct_txt = f"{int(round(acc*100))}%"
            c.setFont("NanumGothic-Bold", 10)

            if fill_w >= 18*mm:
                c.setFillColor(colors.white)
                c.drawRightString(bar_x0 + min(fill_w - 2*mm, bar_w - 2*mm), ry + 1.7*mm, pct_txt)
            else:
                c.setFillColor(title_col)
                c.drawString(bar_x0 + 2*mm, ry + 1.7*mm, pct_txt)

            # score right
            c.setFillColor(muted)
            c.setFont("NanumGothic", 10.5)
            score_txt = f"{int(correct)}/{int(total)}" if total else "-"
            c.drawRightString(score_x + col_score - 1*mm, ry + 1.7*mm, score_txt)

    # ---- Tab4 PDF 생성 (문항표에 난이도/단원 추가 + 정답률<50% 굵게+남색) ----
    def create_report_pdf_reportlab_tab4(
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
        topic_m1: dict,
        topic_m2: dict,
        diff_m1: dict,
        diff_m2: dict,
        footer_left_text: str = "",
    ):
        ensure_fonts_registered4()
        c = canvas.Canvas(output_path, pagesize=A4)
        W, H = A4

        # colors
        stroke    = colors.Color(203/255, 213/255, 225/255)
        title_col = colors.Color(15/255, 23/255, 42/255)
        muted     = colors.Color(100/255, 116/255, 139/255)
        pill_fill = colors.Color(241/255, 245/255, 249/255)
        row_stripe = colors.Color(248/255, 250/255, 252/255)
        green = colors.Color(22/255, 101/255, 52/255)
        red   = colors.Color(220/255, 38/255, 38/255)
        navy  = colors.Color(30/255, 64/255, 175/255)   # 정답률<50% 남색

        # layout
        L = 15*mm
        R = 15*mm
        usable_w = W - L - R
        TOP = H - 26*mm

        # generated
        c.setFont("NanumGothic", 9.5)
        c.setFillColor(muted)
        c.drawRightString(W - R, TOP + 15*mm, f"Generated: {gen_date_str}")

        # title/subtitle (한 페이지 맞추기 위해 살짝 축소)
        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 26)
        c.drawString(L, TOP, title)

        c.setFillColor(muted)
        c.setFont("NanumGothic", 12.5)
        c.drawString(L, TOP - 9.5*mm, subtitle)

        # name pill
        pill_w = 76*mm
        pill_h = 18*mm
        pill_x = L + usable_w - pill_w
        pill_y = TOP - 11*mm
        draw_round_rect4(c, pill_x, pill_y, pill_w, pill_h, 9*mm, pill_fill, stroke, 1)

        c.setFillColor(muted)
        c.setFont("NanumGothic-Bold", 9.5)
        c.drawString(pill_x + 6.5*mm, pill_y + 11.2*mm, "Name")

        c.setFillColor(title_col)
        max_name_w = pill_w - 24*mm
        name_fs = fit_font_size4(student_name, "NanumGothic-Bold", 15, 9.5, max_name_w)
        c.setFont("NanumGothic-Bold", name_fs)
        c.drawRightString(pill_x + pill_w - 6.5*mm, pill_y + 5.2*mm, student_name)

        # divider
        line_y = TOP - 20*mm
        c.setLineWidth(2)
        c.setStrokeColor(title_col)
        c.line(L, line_y, W - R, line_y)

        # KPI cards (높이 축소)
        kpi_h = 24*mm
        gap = 10*mm
        kpi_w = (usable_w - gap) / 2
        kpi_y = line_y - 6*mm - kpi_h

        def draw_kpi_card(x, y, w, h, label, score, dt, t):
            draw_round_rect4(c, x, y, w, h, 8*mm, colors.white, stroke, 1)
            c.setFillColor(title_col)
            c.setFont("NanumGothic-Bold", 13)
            c.drawString(x + 7*mm, y + h - 9.2*mm, label)

            c.setFont("NanumGothic-Bold", 24)
            c.drawRightString(x + w - 7*mm, y + h - 13.2*mm, str(score))

            c.setFillColor(muted)
            c.setFont("NanumGothic", 8)
            c.drawString(x + 7*mm, y + 4.0*mm, f"{dt}")
            c.drawRightString(x + w - 7*mm, y + 4.0*mm, f"{t}")

        draw_kpi_card(L, kpi_y, kpi_w, kpi_h, "Module 1", m1_meta["score"], m1_meta["dt"], m1_meta["time"])
        draw_kpi_card(L + kpi_w + gap, kpi_y, kpi_w, kpi_h, "Module 2", m2_meta["score"], m2_meta["dt"], m2_meta["time"])

        # ---- 문항표 (2개 카드) ----
        header_h = 6.0*mm
        row_h = 5.15*mm
        top_padding = 5.0*mm
        bottom_padding = 5.0*mm
        card_h = top_padding + header_h + (22 * row_h) + bottom_padding

        card_y = kpi_y - 3.2*mm - card_h

        card_w = kpi_w
        left_x = L
        right_x = L + card_w + gap

        def draw_table(x, y, w, h, ans_dict, wr_dict, wrong_set, topic_dict, diff_dict):
            draw_round_rect4(c, x, y, w, h, 10*mm, colors.white, stroke, 1)

            strip_y = y + h - top_padding - header_h
            strip_h = header_h

            c.setLineWidth(1)
            c.setStrokeColor(stroke)
            c.setFillColor(pill_fill)
            c.rect(x + 6*mm, strip_y, w - 12*mm, strip_h, stroke=1, fill=1)

            inner_x = x + 8*mm
            inner_w = w - 16*mm

            # 순서: No./ Answer/ 정답률/ Result/ 난이도/ 단원
            col_no   = 8*mm
            col_ans  = 18*mm
            col_wr   = 15*mm
            col_res  = 11*mm
            col_diff = 12*mm
            col_top  = inner_w - (col_no + col_ans + col_wr + col_res + col_diff)

            cx_no   = inner_x + col_no/2
            cx_ans  = inner_x + col_no + col_ans/2
            cx_wr   = inner_x + col_no + col_ans + col_wr/2
            cx_res  = inner_x + col_no + col_ans + col_wr + col_res/2
            cx_diff = inner_x + col_no + col_ans + col_wr + col_res + col_diff/2
            cx_top  = inner_x + col_no + col_ans + col_wr + col_res + col_diff + col_top/2

            header_text_y = strip_y + 1.8*mm
            c.setFillColor(muted)
            c.setFont("NanumGothic-Bold", 8.8)
            c.drawCentredString(cx_no, header_text_y, "No.")
            c.drawCentredString(cx_ans, header_text_y, "Answer")
            c.drawCentredString(cx_wr, header_text_y, "정답률")
            c.drawCentredString(cx_res, header_text_y, "Result")
            c.drawCentredString(cx_diff, header_text_y, "난이도")
            c.drawCentredString(cx_top, header_text_y, "단원")

            start_y = strip_y - 0.5*mm - row_h
            base = 1.15*mm

            for i, q in enumerate(range(1, 23)):
                ry = start_y - i * row_h

                if q % 2 == 0:
                    c.setFillColor(row_stripe)
                    c.setStrokeColor(row_stripe)
                    c.rect(x + 6*mm, ry, w - 12*mm, row_h, stroke=0, fill=1)

                ans_raw = _clean4(ans_dict.get(q, ""))
                lines = ans_raw.split("\n") if "\n" in ans_raw else [ans_raw]
                lines = [ln.strip() for ln in lines if ln.strip()]
                if not lines:
                    lines = [""]

                if len(lines) > 2:
                    lines = [lines[0], " ".join(lines[1:])]

                rate_val = wr_dict.get(q, None)
                wr_txt = wr_to_text4(rate_val)

                res_txt = "X" if q in wrong_set else "O"
                diff_txt = _clean4(diff_dict.get(q, "")) or "-"
                topic_txt = _clean4(topic_dict.get(q, "")) or "-"

                # No.
                c.setFillColor(title_col)
                c.setFont("NanumGothic", 9.5)
                c.drawCentredString(cx_no, ry + base, str(q))

                # Answer
                ans_max_w = col_ans - 2*mm
                fs = fit_font_size_two_lines4(lines, "NanumGothic-Bold", 9.0, 6.8, ans_max_w)
                c.setFont("NanumGothic-Bold", fs)
                c.setFillColor(title_col)
                if len(lines) == 1:
                    c.drawCentredString(cx_ans, ry + base, lines[0])
                else:
                    c.drawCentredString(cx_ans, ry + base + 0.6*mm, lines[0])
                    c.drawCentredString(cx_ans, ry + base - 0.6*mm, lines[1])

                # 정답률: 50% 미만이면 굵게 + 남색
                is_low = False
                try:
                    if rate_val is not None and float(rate_val) < 0.5:
                        is_low = True
                except:
                    pass

                if is_low:
                    c.setFillColor(navy)
                    c.setFont("NanumGothic-Bold", 9.6)
                else:
                    c.setFillColor(title_col)
                    c.setFont("NanumGothic", 9.4)

                c.drawCentredString(cx_wr, ry + base, wr_txt)

                # Result
                ox_color = red if res_txt == "X" else green
                c.setFillColor(ox_color)
                c.setFont("NanumGothic-Bold", 10.0)
                c.drawCentredString(cx_res, ry + base, res_txt)

                # 난이도
                c.setFillColor(title_col)
                c.setFont("NanumGothic-Bold", 9.2)
                c.drawCentredString(cx_diff, ry + base, diff_txt)

                # Topic (단원)
                c.setFillColor(title_col)
                c.setFont("NanumGothic", 9.0)
                c.drawCentredString(cx_top, ry + base, topic_txt)

        draw_table(left_x, card_y, card_w, card_h, ans_m1, wr_m1, wrong_m1, topic_m1, diff_m1)
        draw_table(right_x, card_y, card_w, card_h, ans_m2, wr_m2, wrong_m2, topic_m2, diff_m2)

        # ---- 하단 Domain Breakdown 카드 (TOPIC_NAMES 사용) ----
        # meta dict (q -> "1.1" 같은 문자열 / q -> "E/M/H")
        meta_topic = {1: topic_m1, 2: topic_m2}
        meta_diff  = {1: diff_m1,  2: diff_m2}

        # 통계 계산
        dom_stats = {k: {"ok": 0, "tot": 0} for k in range(1, 8)}
        dif_stats = {k: {"ok": 0, "tot": 0} for k in ["E", "M", "H"]}

        def _major_from_topic(topic_str: str):
            s = str(topic_str or "").strip()
            m = re.match(r"^\s*(\d+)", s)
            if not m:
                return None
            v = int(m.group(1))
            return v if 1 <= v <= 7 else None

        for mod in (1, 2):
            wrong_set = wrong_m1 if mod == 1 else wrong_m2
            for q in range(1, 23):
                is_ok = (q not in wrong_set)

                # Domain(Topic)
                t = (meta_topic.get(mod, {}) or {}).get(q, "")
                major = _major_from_topic(t)
                if major is not None:
                    dom_stats[major]["tot"] += 1
                    if is_ok:
                        dom_stats[major]["ok"] += 1

                # Difficulty
                d = (meta_diff.get(mod, {}) or {}).get(q, "")
                d = str(d or "").strip().upper()
                if d in dif_stats:
                    dif_stats[d]["tot"] += 1
                    if is_ok:
                        dif_stats[d]["ok"] += 1

        # 카드 위치/크기
        domain_h = 63 * mm
        domain_gap = 4 * mm
        domain_y = card_y - domain_gap - domain_h
        domain_x = L
        domain_w = usable_w

        # footer 침범 방지
        min_bottom = 20 * mm
        if domain_y < min_bottom:
            shrink = (min_bottom - domain_y)
            domain_h = max(40 * mm, domain_h - shrink)
            domain_y = min_bottom

        # 막대 색상 (오른쪽 회색 = track, 왼쪽 파랑 = fill)
        bar_track = colors.Color(226/255, 232/255, 240/255)  # 연회색
        bar_fill  = colors.Color(191/255, 219/255, 254/255)  # 연파랑

        draw_round_rect4(c, domain_x, domain_y, domain_w, domain_h, 10*mm, colors.white, stroke, 1)

        # 제목만 (설명줄 삭제 + 살짝 위로)
        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 12.5)
        c.drawString(domain_x + 8*mm, domain_y + domain_h - 9*mm, "Topic")

        # 내부 영역 시작 y도 조금 올림(설명줄 없어진 만큼)
        inner_x = domain_x + 8*mm
        inner_y_top = domain_y + domain_h - 13*mm   # (기존 -20mm → -15mm)
        inner_w = domain_w - 16*mm

        # 우측 difficulty 박스 (박스 크기는 그대로 사용)
        diff_box_w = 48 * mm
        diff_box_h = 32 * mm
        diff_box_x = inner_x + inner_w - diff_box_w
        diff_box_y = inner_y_top - diff_box_h + 2*mm
        draw_round_rect4(c, diff_box_x, diff_box_y, diff_box_w, diff_box_h, 7*mm, colors.white, stroke, 1)

        # ✅ 제목 삭제 (원래 "Difficulty" 그리던 부분 제거)

        def pct_only(ok, tot):
            if tot <= 0:
                return "-"
            return f"{int(round((ok/tot)*100))}%"

        def frac_only(ok, tot):
            if tot <= 0:
                return ""
            return f"({ok}/{tot})"

        # ✅ E만 Bold + 나머지(asy/edium/ard)는 Regular
        rows = [
            ("E", "asy",   dif_stats["E"]),
            ("M", "edium", dif_stats["M"]),
            ("H", "ard",   dif_stats["H"]),
        ]

        # ✅ 좌/우 여백 줄이기
        pad = 4 * mm
        x_left  = diff_box_x + pad
        x_right = diff_box_x + diff_box_w - pad

        # ✅ 오른쪽 정렬 기준선 2개 (퍼센트 / 분수)
        #    (ok/tot)는 맨 오른쪽, %는 그 왼쪽에 "고정 폭"으로 정렬
        x_frac_r = x_right
        x_pct_r  = x_right - 12 * mm   # 간격: 16~22mm 사이로 취향 조절

        # ✅ 세로 중앙 정렬 (3줄이 박스 중앙에 오도록)
        row_step = 8.0 * mm            # 줄 겹치면 8.4~8.8로
        y_mid = diff_box_y + diff_box_h / 2
        y0 = y_mid + row_step - 1.2*mm          # 첫 줄 baseline (Easy)

        for i, (b, rest, stt) in enumerate(rows):
            y = y0 - i * row_step

            # 왼쪽: Easy/Medium/Hard (E만 Bold로 보이게)
            c.setFillColor(title_col)

            c.setFont("NanumGothic-Bold", 10)
            c.drawString(x_left, y, b)

            b_w = pdfmetrics.stringWidth(b, "NanumGothic-Bold", 10)
            c.setFont("NanumGothic", 9.5)
            c.drawString(x_left + b_w, y, rest)

            # 오른쪽: % / (ok/tot) 정렬 (둘 다 right align)
            c.setFillColor(muted)
            c.setFont("NanumGothic", 9.5)
            c.drawRightString(x_pct_r,  y, pct_only(stt["ok"], stt["tot"]))
            c.setFont("NanumGothic", 9.5)
            c.drawRightString(x_frac_r, y, frac_only(stt["ok"], stt["tot"]))



        # 7개 막대그래프(좌측)
        chart_w = inner_w - diff_box_w - 6*mm
        chart_x = inner_x
        chart_y_top = inner_y_top - 3*mm

        # ✅ 여기만 조절하면 됨 (Topic 글자 영역 줄이기)
        label_w = 50 * mm      # (기존 62mm → 50mm로 축소)
        value_w = 18 * mm      # 오른쪽 (ok/tot)용

        row_gap = 6.4 * mm
        bar_h = 4.6 * mm       # %를 막대 안에 넣을 거라 살짝 키움
        bar_max_w = max(10*mm, chart_w - label_w - value_w - 4*mm)

        # ✅ 낮은 Topic 기준(원하면 0.6, 0.7 등으로 바꿔도 됨)
        LOW_TOPIC_PCT = 0.70

        # 색상
        bar_track = colors.Color(226/255, 232/255, 240/255)     # 연회색(track)
        bar_fill  = colors.Color(191/255, 219/255, 254/255)     # 연파랑(fill)
        bar_fill_low = colors.Color(248/255, 200/255, 214/255)    #  파스텔 핑크(low fill)

        for idx, major in enumerate(range(1, 8)):
            y = chart_y_top - idx * row_gap

            label = TOPIC_NAMES.get(major, str(major))
            stt = dom_stats[major]
            ok, tot = stt["ok"], stt["tot"]
            pct = (ok / tot) if tot > 0 else None

            # --- 라벨(왼쪽) ---
            c.setFillColor(title_col)
            if major == 5:
                c.setFont("NanumGothic", 8.2)
                c.drawString(chart_x, y + 1.1*mm, "5. Polynomials, radical")
                c.drawString(chart_x, y - 2.3*mm, "   and rational functions")
            else:
                c.setFont("NanumGothic", 8.6)
                c.drawString(chart_x, y, label.replace("\n", " "))

            # --- 막대 track ---
            track_x = chart_x + label_w
            track_y = y - 2.0*mm
            c.setFillColor(bar_track)
            c.setStrokeColor(bar_track)
            c.rect(track_x, track_y, bar_max_w, bar_h, stroke=0, fill=1)

            # --- 막대 fill ---
            fill_w = (bar_max_w * pct) if pct is not None else 0

            is_low = (pct is not None and pct < LOW_TOPIC_PCT)
            fill_color = bar_fill_low if is_low else bar_fill

            c.setFillColor(fill_color)
            c.setStrokeColor(fill_color)
            c.rect(track_x, track_y, fill_w, bar_h, stroke=0, fill=1)

            # ✅ %를 막대 안에 "검정색"으로 표시
            if pct is not None and tot > 0:
                pct_txt = f"{int(round(pct*100))}%"
                c.setFillColor(title_col)          # ✅ 검정/진남색 계열 글자
                c.setFont("NanumGothic-Bold", 8.2)

                # 텍스트를 fill 안쪽 오른쪽에 최대한 붙이되,
                # fill이 너무 짧으면 track 시작쪽에서 보이게 처리
                txt_w = pdfmetrics.stringWidth(pct_txt, "NanumGothic-Bold", 8.2)
                pad = 1.4 * mm

                # 기본: fill 끝 - pad (오른쪽 정렬 느낌)
                x_right = track_x + fill_w - pad
                x_left_min = track_x + pad
                x_right_max = track_x + bar_max_w - pad

                # fill이 너무 짧아서 텍스트가 못 들어가면, 그냥 track 안 왼쪽에 표시
                x_draw = max(x_left_min, min(x_right - txt_w, x_right_max - txt_w))
                c.drawString(x_draw, track_y + 1.15*mm, pct_txt)

            # --- 오른쪽 (ok/tot)만 유지 (기존 %는 막대 안으로 이동했으니 제거) ---
            c.setFillColor(muted)
            c.setFont("NanumGothic", 8.2)
            c.drawRightString(
                chart_x + label_w + bar_max_w + value_w,
                y - 1.0*mm,
                f"{ok}/{tot}" if tot > 0 else "-"
            )


        # footer
        if footer_left_text:
            c.setFillColor(title_col)
            c.setFont("NanumGothic", 8)
            lines = str(footer_left_text).splitlines()
            y0 = 12*mm
            line_gap = 4.2*mm
            for idx, ln in enumerate(lines):
                c.drawString(L, y0 + (len(lines)-1-idx)*line_gap, ln)

        c.showPage()
        c.save()
        return output_path

    def render_pdf_first_page_to_png_bytes4(pdf_path: str, zoom: float = 2.0) -> bytes:
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")

    # ====== 실행 버튼 ======
    if st.button("🚀 개인 성적표(단원/난이도) 생성", type="primary", key="t4_btn"):
        if not eta_file4 or not mock_file4:
            st.warning("⚠️ ETA.xlsx와 Mock데이터.xlsx를 모두 업로드해주세요.")
            st.stop()

        if not font_ready:
            st.error("⚠️ 한글 PDF 생성을 위해 fonts 폴더에 NanumGothic.ttf / NanumGothicBold.ttf가 필요합니다.")
            st.stop()

        try:
            eta_xl = pd.ExcelFile(eta_file4)

            # ---- Student Analysis ----
            if STUDENT_SHEET not in eta_xl.sheet_names:
                st.error(f"⚠️ ETA.xlsx에 '{STUDENT_SHEET}' 시트가 없습니다.")
                st.stop()

            raw_sa = pd.read_excel(eta_xl, sheet_name=STUDENT_SHEET, header=None)
            if raw_sa.shape[0] <= SA_HEADER_ROW_IDX:
                st.error("⚠️ Student Analysis에서 2행(헤더)을 찾을 수 없습니다.")
                st.stop()

            sa_header = raw_sa.iloc[SA_HEADER_ROW_IDX].astype(str).tolist()
            student_df = raw_sa.iloc[SA_HEADER_ROW_IDX + 1:].copy()
            student_df.columns = sa_header
            student_df = student_df.dropna(axis=1, how="all").dropna(axis=0, how="all")

            assert_columns4(student_df, [SA_NAME_COL, SA_M1_SCORE_COL, SA_M2_SCORE_COL], STUDENT_SHEET)

            students = [_clean4(x) for x in student_df[SA_NAME_COL].dropna().tolist()]
            students = [s for s in students if s != ""]
            if not students:
                st.error("학생 목록이 비어있습니다.")
                st.stop()

            # ---- QuizResults ----
            if QUIZ_SHEET not in eta_xl.sheet_names:
                st.error(f"⚠️ ETA.xlsx에 '{QUIZ_SHEET}' 시트가 없습니다.")
                st.stop()

            quiz_df = pd.read_excel(eta_xl, sheet_name=QUIZ_SHEET, header=QZ_HEADER_ROW_IDX)
            quiz_df.columns = [str(c).strip() for c in quiz_df.columns]
            quiz_df = quiz_df.dropna(axis=1, how="all").dropna(axis=0, how="all")

            assert_columns4(
                quiz_df,
                [QZ_KEYWORD_COL, QZ_MODULE_COL, QZ_NAME_COL, QZ_DT_COL, QZ_TIME_COL, QZ_SCORE_COL, QZ_WRONG_COL],
                QUIZ_SHEET
            )

            quiz_map = {}
            for _, r in quiz_df.iterrows():
                nm = _clean4(r.get(QZ_NAME_COL, ""))
                md = _clean4(r.get(QZ_MODULE_COL, "")).upper()
                if nm == "":
                    continue

                if md in ["M1", "MODULE1", "1"]:
                    mod = 1
                elif md in ["M2", "MODULE2", "2"]:
                    mod = 2
                else:
                    continue

                quiz_map.setdefault(nm, {})[mod] = {
                    "dt": _clean4(r.get(QZ_DT_COL, "")) or "-",
                    "time": _clean4(r.get(QZ_TIME_COL, "")) or "-",
                    "score": score_to_slash224(r.get(QZ_SCORE_COL, "")),
                    "wrong_set": parse_wrong_list4(r.get(QZ_WRONG_COL, "")),
                    "keyword": _clean4(r.get(QZ_KEYWORD_COL, "")) or "",
                }

            # ---- 정답률(Accuracy/Error Analysis) ----
            target_sheet = None
            if "Accuracy Analysis" in eta_xl.sheet_names:
                target_sheet = "Accuracy Analysis"
            elif "Error Analysis" in eta_xl.sheet_names:
                target_sheet = "Error Analysis"

            if target_sheet:
                wr1, wr2 = build_wrong_rate_dict_fixed_ranges4(eta_xl, target_sheet)
            else:
                wr1, wr2 = {}, {}

            # ---- Mock Answers + Topic/Difficulty ----
            ans1, ans2, topic1, topic2, diff1, diff2 = read_mock_answers_with_meta(mock_file4)

            # ---- 생성 ----
            output_dir = "generated_reports_tab4"
            os.makedirs(output_dir, exist_ok=True)

            made_files = []
            made_images = []
            skipped = []
            prog = st.progress(0)

            common_subtitle = "-"

            for i, stu in enumerate(students):
                q = quiz_map.get(stu, {})
                m1 = q.get(1, {})
                m2 = q.get(2, {})

                m1_score_txt = _clean4(m1.get("score", ""))
                m2_score_txt = _clean4(m2.get("score", ""))

                if m1_score_txt == "" or m2_score_txt == "":
                    skipped.append(stu)
                    prog.progress((i+1)/len(students))
                    continue

                subtitle_kw = _clean4(m1.get("keyword", "")) or _clean4(m2.get("keyword", "")) or "-"
                if subtitle_kw != "-" and common_subtitle == "-":
                    common_subtitle = subtitle_kw

                m1_meta = {"score": m1_score_txt, "dt": m1.get("dt", "-"), "time": m1.get("time", "-")}
                m2_meta = {"score": m2_score_txt, "dt": m2.get("dt", "-"), "time": m2.get("time", "-")}

                wrong1 = set(m1.get("wrong_set", set()))
                wrong2 = set(m2.get("wrong_set", set()))

                pdf_path = os.path.join(output_dir, f"{stu}_{generated_date4.strftime('%Y%m%d')}.pdf")

                create_report_pdf_reportlab_tab4(
                    output_path=pdf_path,
                    title=report_title4,
                    subtitle=subtitle_kw,
                    gen_date_str=generated_date4.strftime("%Y-%m-%d"),
                    student_name=stu,
                    m1_meta=m1_meta,
                    m2_meta=m2_meta,
                    ans_m1=ans1,
                    ans_m2=ans2,
                    wr_m1=wr1,
                    wr_m2=wr2,
                    wrong_m1=wrong1,
                    wrong_m2=wrong2,
                    topic_m1=topic1,
                    topic_m2=topic2,
                    diff_m1=diff1,
                    diff_m2=diff2,
                    footer_left_text=FOOTER_LEFT_TEXT,
                )

                made_files.append((stu, pdf_path))

                # PNG
                try:
                    png_bytes = render_pdf_first_page_to_png_bytes4(pdf_path, zoom=2.0)
                    png_path = os.path.join(output_dir, f"{stu}_{generated_date4.strftime('%Y%m%d')}.png")
                    with open(png_path, "wb") as f:
                        f.write(png_bytes)
                    made_images.append((stu, png_path))
                except:
                    pass

                prog.progress((i+1)/len(students))

            # 템플릿 1개 (Report)
            template_pdf = os.path.join(output_dir, f"Report_{generated_date4.strftime('%Y%m%d')}.pdf")
            create_report_pdf_reportlab_tab4(
                output_path=template_pdf,
                title=report_title4,
                subtitle=common_subtitle,
                gen_date_str=generated_date4.strftime("%Y-%m-%d"),
                student_name="-",
                m1_meta={"score": "-", "dt": "-", "time": "-"},
                m2_meta={"score": "-", "dt": "-", "time": "-"},
                ans_m1=ans1,
                ans_m2=ans2,
                wr_m1=wr1,
                wr_m2=wr2,
                wrong_m1=set(),
                wrong_m2=set(),
                topic_m1=topic1,
                topic_m2=topic2,
                diff_m1=diff1,
                diff_m2=diff2,
                footer_left_text=FOOTER_LEFT_TEXT,
            )
            made_files.append(("Report", template_pdf))

            try:
                png_bytes = render_pdf_first_page_to_png_bytes4(template_pdf, zoom=2.0)
                template_png = os.path.join(output_dir, f"Report_{generated_date4.strftime('%Y%m%d')}.png")
                with open(template_png, "wb") as f:
                    f.write(png_bytes)
                made_images.append(("Report", template_png))
            except:
                pass

            if not made_files:
                st.warning("생성된 PDF가 없습니다. (QuizResults 점수 blank로 모두 제외되었을 수 있어요)")
                if skipped:
                    with st.expander(f"제외된 학생 ({len(skipped)}명) - 점수 blank"):
                        for s in skipped:
                            st.write(f"- {s}")
                st.stop()

            # ZIPs
            pdf_zip_buf = io.BytesIO()
            with zipfile.ZipFile(pdf_zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for _, path in made_files:
                    if os.path.exists(path):
                        z.write(path, arcname=os.path.basename(path))
            pdf_zip_buf.seek(0)

            img_zip_buf = io.BytesIO()
            with zipfile.ZipFile(img_zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for _, path in made_images:
                    if os.path.exists(path):
                        z.write(path, arcname=os.path.basename(path))
            img_zip_buf.seek(0)

            st.success(f"✅ 생성 완료(Tab4): PDF {len(made_files)}개 / 이미지 {len(made_images)}개 (제외: {len(skipped)}명)")
            if skipped:
                with st.expander(f"제외된 학생 ({len(skipped)}명) - 점수 blank"):
                    for s in skipped:
                        st.write(f"- {s}")

            st.download_button(
                "📦 개인 성적표(Tab4) PDF ZIP 다운로드",
                data=pdf_zip_buf,
                file_name=f"개인성적표_TAB4_PDF_{generated_date4.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="t4_download_pdf_zip"
            )

            st.download_button(
                "🖼️ 개인 성적표(Tab4) 이미지(PNG) ZIP 다운로드",
                data=img_zip_buf,
                file_name=f"개인성적표_TAB4_PNG_{generated_date4.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="t4_download_png_zip"
            )

        except ModuleNotFoundError as e:
            st.error("❌ reportlab이 설치되어 있지 않습니다. (requirements.txt에 reportlab 추가 필요)")
            st.exception(e)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.exception(e)

