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

    if found["이름"]: rename_map[found["이름"]] = "이름"
    if found["Module1"]: rename_map[found["Module1"]] = "Module1"
    if found["Module2"]: rename_map[found["Module2"]] = "Module2"

    df = df.rename(columns=rename_map)
    return df

def example_input_df():
    return pd.DataFrame({
        '학생 이름': ['홍길동', '김철수', '이영희', '박지성', '손흥민'],
        '[M1] 점수': [100, 90, 100, 50, None],
        '[M1] 틀린 문제': ['1,3,5', 'X', 'X', '1', None],
        '[M2] 점수': [95, 85, 100, None, None],
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
                if len(parts) < 2: continue
                folder = parts[0].lower()
                q_num = os.path.splitext(os.path.basename(file))[0]
                with z.open(file) as f:
                    img = Image.open(f).convert("RGB")
                    if folder == "m1": m1_imgs[q_num] = img
                    elif folder == "m2": m2_imgs[q_num] = img
    return m1_imgs, m2_imgs

def create_student_pdf(name, m1_imgs, m2_imgs, doc_title, output_dir):
    if not font_ready: return None
    pdf = KoreanPDF()
    pdf.add_page()
    pdf.set_font(pdf_font_name, style='B', size=10)
    # [FIX] txt/ln deprecated 대응
    pdf_cell_ln(pdf, 0, 8, f"<{name}_{doc_title}>")

    def add_images(title, images):
        est_height = 80
        if images and (pdf.get_y() + 10 + est_height > pdf.page_break_trigger):
            pdf.add_page()

        pdf.set_font(pdf_font_name, size=10)
        # [FIX] txt/ln deprecated 대응
        pdf_cell_ln(pdf, 0, 8, title)

        if images:
            for img in images:
                temp_filename = f"temp_{datetime.now().timestamp()}_{os.urandom(4).hex()}.jpg"
                img.save(temp_filename)
                # [고정] A4 여백 고려하여 가장 예쁜 사이즈 150mm로 고정
                pdf.image(temp_filename, w=150)
                try: os.remove(temp_filename)
                except: pass
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
# [Tab 2] PDF 문제 자르기 관련 상수 및 함수
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
    for b in page.get_text("blocks"):
        if len(b) < 5: continue
        y0 = b[1]
        text = b[4]
        if y0 < y_from or y0 > y_to: continue
        if text and HEADER_FOOTER_HINT_RE.search(str(text)):
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
# [Tab 1] 오답노트 생성기 (✅ 원본 그대로)
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
    doc_title = st.text_input("문서 제목 (예: 25 S2 SAT MATH 만점반 Mock Test1)", value="25 S2 SAT MATH 만점반 Mock Test1", key="t1_title")

    st.header("📦 파일 업로드")

    st.write("")
    st.markdown("#### 문제 이미지 ZIP 파일")
    img_zip = st.file_uploader("m1, m2 폴더가 들어있는 ZIP 파일", type="zip", key="t1_zip")

    st.markdown("#### 오답 현황 엑셀 파일")
    excel_file = st.file_uploader("학생들의 결과 데이터가 담긴 엑셀 파일", type="xlsx", key="t1_excel")

    st.write("")

    if st.button("🚀 오답노트 생성 시작", type="primary", key="t1_btn"):
        if not img_zip or not excel_file:
            st.warning("⚠️ 이미지 ZIP 파일과 엑셀 파일을 모두 업로드해주세요.")
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
                    name = row['이름']

                    def parse_module_data(x):
                        if pd.isna(x): return None
                        s = str(x).strip()
                        if s == "": return None
                        if s.upper() in ["X", "Х", "-"]: return []
                        s = s.replace("，", ",").replace(";", ",")
                        nums = [t.strip() for t in s.split(",") if t.strip()]
                        return nums if nums else []

                    m1_data = parse_module_data(row['Module1'])
                    m2_data = parse_module_data(row['Module2'])

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

    # 결과 표시 로직 (원본)
    if st.session_state.generated_files or st.session_state.skipped_details:
        if st.session_state.skipped_details:
            total_skipped = sum(len(v) for v in st.session_state.skipped_details.values())
            if total_skipped > 0:
                with st.expander(f"📋 생성 제외 명단 (총 {total_skipped}명) - 클릭하여 보기", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**🏆 만점 (Perfect)**")
                        if st.session_state.skipped_details["만점"]:
                            for n in st.session_state.skipped_details["만점"]: st.text(f"- {n}")
                        else:
                            st.caption("없음")
                    with c2:
                        st.markdown("**⚠️ 하나 미제출**")
                        if st.session_state.skipped_details["M1/M2 하나 미제출"]:
                            for n in st.session_state.skipped_details["M1/M2 하나 미제출"]: st.text(f"- {n}")
                        else:
                            st.caption("없음")
                    with c3:
                        st.markdown("**❌ 미제출**")
                        if st.session_state.skipped_details["미제출"]:
                            for n in st.session_state.skipped_details["미제출"]: st.text(f"- {n}")
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
    frq_val = c4.slider("FRQ 아래 여백(px)", 0, 600, 250, 25, key="t2_frq")

    unify_width = st.checkbox("모듈 내 가로폭을 가장 넓은 문제에 맞춤(오른쪽만 확장)", value=True, key="t2_chk")

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

                    zbuf_data, zname = make_zip_from_rects(
                        doc_obj,
                        rects_data,
                        zoom_val,
                        zip_base,
                        unify_width_right=unify_width,
                    )

                    st.success(f"✅ 처리가 완료되었습니다! (총 {len(rects_data)}문제 추출)")
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
# [Tab 4] 개인 성적표 + 단원/난이도 + HTML 스타일 Topic 패널
# (✅ Tab1~Tab3 건드리지 않음 / Tab4만 추가)
# ---------------------------------------------------------
with tab4:
    st.header("📈 개인 성적표(단원/난이도)")
    st.info("Tab3와 동일 데이터 + Mock데이터.xlsx의 '단원','난이도' 컬럼을 사용하여 표에 추가 + 하단 Topic 막대그래프(HTML 스타일)")

    eta_file4 = st.file_uploader("ETA 결과 파일 업로드 (ETA.xlsx)", type=["xlsx"], key="t4_eta")
    mock_file4 = st.file_uploader("Mock 정답+단원+난이도 파일 업로드 (Mock데이터.xlsx)", type=["xlsx"], key="t4_mock")

    c1, c2 = st.columns([1, 1])
    with c1:
        report_title4 = st.text_input("리포트 제목", value="SAT Math Report", key="t4_title")
    with c2:
        generated_date4 = st.date_input("Generated 날짜", value=datetime.now().date(), key="t4_gen_date")

    st.caption("부제목은 QuizResults의 '검색 키워드'가 학생별로 자동으로 들어갑니다.")

    # ---------- Tab3와 동일 상수 ----------
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

    TOPIC_NAMES = {
        1: "1. Linear",
        2: "2. Percent & Unit Conversion",
        3: "3. Quadratic",
        4: "4. Exponential",
        5: "5. Polynomials, radical and rational functions",
        6: "6. Geometry",
        7: "7. Statistics",
    }

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

    # ---------- [Tab4 전용] Mock데이터: 정답 + 단원 + 난이도 ----------
    def read_mock_answers_with_meta(mock_bytes):
        df = pd.read_excel(mock_bytes)
        df.columns = [str(c).strip() for c in df.columns]

        need = {"모듈", "문항번호", "정답"}
        if not need.issubset(set(df.columns)):
            raise ValueError("Mock데이터.xlsx에 최소 '모듈','문항번호','정답' 컬럼이 필요합니다. (+ '단원','난이도' 권장)")

        def norm_mod(x):
            s = _clean(x).upper()
            if s in ["M1", "MODULE1", "1"]:
                return 1
            if s in ["M2", "MODULE2", "2"]:
                return 2
            return None

        ans1, ans2 = {}, {}
        meta_topic = {}  # (mod, q) -> "5.3"
        meta_diff  = {}  # (mod, q) -> "E/M/H"

        for _, r in df.iterrows():
            mod = norm_mod(r.get("모듈", ""))
            if mod not in (1, 2):
                continue

            try:
                q = int(float(str(r.get("문항번호", "")).strip()))
            except:
                continue

            ans = _clean(r.get("정답", ""))
            topic = _clean(r.get("단원", ""))          # 예: "5.10"
            diff  = _clean(r.get("난이도", "")).upper()  # E/M/H

            if mod == 1:
                ans1[q] = ans
            else:
                ans2[q] = ans

            meta_topic[(mod, q)] = topic
            meta_diff[(mod, q)] = diff

        return ans1, ans2, meta_topic, meta_diff

    def topic_group_from_topic_code(topic_code: str):
        s = _clean(topic_code)
        if s == "":
            return None
        m = re.match(r"^\s*(\d+)", s)
        if not m:
            return None
        g = int(m.group(1))
        return g if 1 <= g <= 7 else None

    def compute_topic_and_difficulty_stats(wrong1: set, wrong2: set, meta_topic: dict, meta_diff: dict):
        topic_stats = {g: {"correct": 0, "total": 0} for g in range(1, 8)}
        diff_stats  = {d: {"correct": 0, "total": 0} for d in ["E", "M", "H"]}

        for mod in (1, 2):
            wrong = wrong1 if mod == 1 else wrong2
            for q in range(1, 23):
                g = topic_group_from_topic_code(meta_topic.get((mod, q), ""))
                if g is not None:
                    topic_stats[g]["total"] += 1
                    if q not in wrong:
                        topic_stats[g]["correct"] += 1

                d = _clean(meta_diff.get((mod, q), "")).upper()
                if d in diff_stats:
                    diff_stats[d]["total"] += 1
                    if q not in wrong:
                        diff_stats[d]["correct"] += 1

        return topic_stats, diff_stats

    # ---------- ReportLab ----------
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

    # ---------- [HTML 스타일] 하단 Topic 패널 ----------
    def draw_topic_panel_html_style(c, x, y, w, h, topic_stats, diff_stats, stroke, title_col, muted):
        # palette
        panel_bg = colors.white
        bar_bg = colors.Color(240/255, 240/255, 240/255)
        blue1 = colors.Color(123/255, 163/255, 201/255)
        blue2 = colors.Color(90/255, 139/255, 184/255)
        amber1 = colors.Color(201/255, 160/255, 123/255)
        amber2 = colors.Color(184/255, 136/255, 90/255)

        diff_box_bg = colors.Color(250/255, 250/255, 250/255)
        diff_box_border = colors.Color(224/255, 224/255, 224/255)

        pad = 7*mm
        draw_round_rect(c, x, y, w, h, 6*mm, panel_bg, stroke, 1)

        # header left: "Topic" + sub
        header_y = y + h - pad - 2*mm
        c.setFillColor(title_col); c.setFont("NanumGothic-Bold", 12)
        c.drawString(x + pad, header_y, "Topic")
        c.setFillColor(muted); c.setFont("NanumGothic", 9)
        c.drawString(x + pad, header_y - 4.2*mm, "(단원별 정답률)")

        # difficulty box right
        box_w = 54*mm
        box_h = 25*mm
        box_x = x + w - pad - box_w
        box_y = header_y - box_h + 3*mm
        draw_round_rect(c, box_x, box_y, box_w, box_h, 3*mm, diff_box_bg, diff_box_border, 1)

        c.setFillColor(title_col); c.setFont("NanumGothic-Bold", 9)
        c.drawString(box_x + 4*mm, box_y + box_h - 6.2*mm, "Difficulty Accuracy")

        def diff_line(d, yy):
            tot = diff_stats[d]["total"]
            cor = diff_stats[d]["correct"]
            p = int(round((cor/tot)*100)) if tot else 0

            c.setFont("NanumGothic-Bold", 9.2)
            c.setFillColor(colors.Color(90/255, 127/255, 170/255))
            c.drawString(box_x + 4*mm, yy, d)

            c.setFont("NanumGothic", 9.0)
            c.setFillColor(title_col)
            c.drawString(box_x + 10*mm, yy, f"{p}%")

            c.setFont("NanumGothic", 8.6)
            c.setFillColor(muted)
            c.drawRightString(box_x + box_w - 4*mm, yy, f"({cor}/{tot})")

        diff_line("E", box_y + box_h - 12.2*mm)
        diff_line("M", box_y + box_h - 17.8*mm)
        diff_line("H", box_y + box_h - 23.4*mm)

        # topic rows
        list_top = box_y - 5*mm
        row_h = 8.2*mm
        name_w = 62*mm
        score_w = 16*mm

        bar_x0 = x + pad + name_w
        bar_x1 = x + w - pad - score_w
        bar_w = max(10*mm, (bar_x1 - bar_x0))

        low_thresh = 0.70  # 70% 미만이면 low 톤

        for idx, g in enumerate(range(1, 8)):
            ry = list_top - idx*row_h

            # topic name
            c.setFillColor(title_col); c.setFont("NanumGothic", 9.5)
            nm = TOPIC_NAMES[g]
            if g == 5:
                nm = "5. Polynomials, radical\n    and rational functions"
            c.drawString(x + pad, ry, nm)

            cor = topic_stats[g]["correct"]
            tot = topic_stats[g]["total"]
            pct = (cor/tot) if tot else 0.0
            pct_txt = f"{int(round(pct*100))}%"

            # bar container
            bar_y = ry - 2.0*mm
            bar_h = 5.4*mm
            c.setFillColor(bar_bg); c.setStrokeColor(bar_bg)
            c.rect(bar_x0, bar_y, bar_w, bar_h, stroke=0, fill=1)

            # bar fill (base + right overlay)
            fill_w = max(14*mm, bar_w * pct) if tot else 14*mm
            fill_w = min(fill_w, bar_w)
            is_low = pct < low_thresh

            base_col = amber1 if is_low else blue1
            edge_col = amber2 if is_low else blue2

            c.setFillColor(base_col); c.setStrokeColor(base_col)
            c.rect(bar_x0, bar_y, fill_w, bar_h, stroke=0, fill=1)

            overlay_w = min(10*mm, fill_w)
            c.setFillColor(edge_col); c.setStrokeColor(edge_col)
            c.rect(bar_x0 + fill_w - overlay_w, bar_y, overlay_w, bar_h, stroke=0, fill=1)

            # percent text in bar
            c.setFillColor(colors.white); c.setFont("NanumGothic-Bold", 9.2)
            c.drawRightString(bar_x0 + fill_w - 2.2*mm, bar_y + 1.25*mm, pct_txt)

            # score
            c.setFillColor(muted); c.setFont("NanumGothic", 9.2)
            c.drawRightString(x + w - pad, ry, f"{cor}/{tot}")

    # ---------- Tab4 PDF ----------
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
        meta_topic: dict,
        meta_diff: dict,
        topic_stats: dict,
        diff_stats: dict,
        result_blank: bool = False,
        footer_left_text: str = "",
    ):
        ensure_fonts_registered()
        c = canvas.Canvas(output_path, pagesize=A4)
        W, H = A4

        stroke = colors.Color(203/255, 213/255, 225/255)
        title_col = colors.Color(15/255, 23/255, 42/255)
        muted = colors.Color(100/255, 116/255, 139/255)
        pill_fill = colors.Color(241/255, 245/255, 249/255)
        row_stripe = colors.Color(248/255, 250/255, 252/255)
        green = colors.Color(22/255, 101/255, 52/255)
        red = colors.Color(220/255, 38/255, 38/255)

        # [요청] 정답률 50% 미만 남색
        navy = colors.Color(30/255, 58/255, 138/255)  # #1e3a8a

        L = 15 * mm
        R = 15 * mm
        TOP = H - 26 * mm
        usable_w = W - L - R

        # Generated
        c.setFont("NanumGothic", 9.5)
        c.setFillColor(muted)
        c.drawRightString(W - R, TOP + 15*mm, f"Generated: {gen_date_str}")

        # Title / subtitle (조금 축소)
        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 26)
        c.drawString(L, TOP, title)

        c.setFillColor(muted)
        c.setFont("NanumGothic", 12)
        c.drawString(L, TOP - 9.2*mm, subtitle)

        # Name pill
        pill_w = 78 * mm
        pill_h = 18 * mm
        pill_x = L + usable_w - pill_w
        pill_y = TOP - 11.5 * mm
        draw_round_rect(c, pill_x, pill_y, pill_w, pill_h, 9*mm, pill_fill, stroke, 1)

        c.setFillColor(muted)
        c.setFont("NanumGothic-Bold", 9.3)
        c.drawString(pill_x + 7*mm, pill_y + 11.5*mm, "Name")

        c.setFillColor(title_col)
        max_name_w = pill_w - 26 * mm
        name_fs = fit_font_size(student_name, "NanumGothic-Bold", 15, 9.5, max_name_w)
        c.setFont("NanumGothic-Bold", name_fs)
        c.drawRightString(pill_x + pill_w - 7*mm, pill_y + 5.3*mm, student_name)

        # divider
        line_y = TOP - 19.5 * mm
        c.setLineWidth(2)
        c.setStrokeColor(title_col)
        c.line(L, line_y, W - R, line_y)

        # KPI (축소)
        kpi_h = 24 * mm
        gap = 10 * mm
        kpi_w = (usable_w - gap) / 2
        kpi_gap_from_line = 6 * mm
        kpi_y = line_y - kpi_gap_from_line - kpi_h

        def draw_kpi_card(x, y, w, h, label, score, dt, t):
            draw_round_rect(c, x, y, w, h, 8*mm, colors.white, stroke, 1)

            c.setFillColor(title_col)
            c.setFont("NanumGothic-Bold", 14)
            c.drawString(x + 8*mm, y + h - 9.5*mm, label)

            c.setFont("NanumGothic-Bold", 24)
            c.drawRightString(x + w - 8*mm, y + h - 14.5*mm, str(score))

            c.setFillColor(muted)
            c.setFont("NanumGothic", 7.8)
            c.drawString(x + 8*mm, y + 4.3*mm, f"{dt}")
            c.drawRightString(x + w - 8*mm, y + 4.3*mm, f"{t}")

        draw_kpi_card(L, kpi_y, kpi_w, kpi_h, "Module 1", m1_meta["score"], m1_meta["dt"], m1_meta["time"])
        draw_kpi_card(L + kpi_w + gap, kpi_y, kpi_w, kpi_h, "Module 2", m2_meta["score"], m2_meta["dt"], m2_meta["time"])

        # ---------- 하단 Topic 패널 위치 ----------
        panel_h = 72 * mm
        panel_y = 16 * mm
        panel_x = L
        panel_w = usable_w

        # ---------- 테이블 위치: 패널 위로 ----------
        header_h = 5.2 * mm
        row_h = 4.9 * mm
        top_padding = 4.0 * mm
        bottom_padding = 4.8 * mm
        card_h = top_padding + header_h + (22 * row_h) + bottom_padding

        card_y = panel_y + panel_h + 6*mm  # 패널 바로 위
        card_w = kpi_w
        left_x = L
        right_x = L + card_w + gap

        def draw_table(x, y, w, h, mod_num, ans_dict, wr_dict, wrong_set):
            draw_round_rect(c, x, y, w, h, 10*mm, colors.white, stroke, 1)

            strip_y = y + h - top_padding - header_h
            strip_h = header_h

            c.setLineWidth(1)
            c.setStrokeColor(stroke)
            c.setFillColor(pill_fill)
            c.rect(x + 6*mm, strip_y, w - 12*mm, strip_h, stroke=1, fill=1)

            inner_x = x + 8 * mm
            inner_w = w - 16 * mm

            # [요청] No./Answer/정답률/Result/난이도/단원
            col_no    = 8 * mm
            col_ans   = 18 * mm
            col_wr    = 12 * mm
            col_res   = 11 * mm
            col_diff  = 8 * mm
            col_topic = inner_w - (col_no + col_ans + col_wr + col_res + col_diff)

            centers = {}
            centers["no"]    = inner_x + col_no / 2
            centers["ans"]   = inner_x + col_no + col_ans / 2
            centers["wr"]    = inner_x + col_no + col_ans + col_wr / 2
            centers["res"]   = inner_x + col_no + col_ans + col_wr + col_res / 2
            centers["diff"]  = inner_x + col_no + col_ans + col_wr + col_res + col_diff / 2
            centers["topic"] = inner_x + col_no + col_ans + col_wr + col_res + col_diff + col_topic / 2

            header_text_y = strip_y + 1.35 * mm
            c.setFillColor(muted)
            c.setFont("NanumGothic-Bold", 8.2)
            c.drawCentredString(centers["no"], header_text_y, "No.")
            c.drawCentredString(centers["ans"], header_text_y, "Answer")
            c.drawCentredString(centers["wr"], header_text_y, "정답률")
            c.drawCentredString(centers["res"], header_text_y, "Result")
            c.drawCentredString(centers["diff"], header_text_y, "난이도")
            c.drawCentredString(centers["topic"], header_text_y, "단원")

            start_y = strip_y - 0.35*mm - row_h
            base = 1.05 * mm

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

                diff_txt  = _clean(meta_diff.get((mod_num, q), "")).upper() or "-"
                topic_txt = _clean(meta_topic.get((mod_num, q), "")) or "-"

                # No
                c.setFillColor(title_col)
                c.setFont("NanumGothic", 8.8)
                c.drawCentredString(centers["no"], ry + base, str(q))

                # Answer
                ans_max_w = col_ans - 2.0*mm
                fs = fit_font_size_two_lines(lines, "NanumGothic-Bold", 8.8, 6.2, ans_max_w)
                c.setFont("NanumGothic-Bold", fs)
                c.setFillColor(title_col)
                if len(lines) == 1:
                    c.drawCentredString(centers["ans"], ry + base, lines[0])
                else:
                    c.drawCentredString(centers["ans"], ry + base + 0.45*mm, lines[0])
                    c.drawCentredString(centers["ans"], ry + base - 0.45*mm, lines[1])

                # 정답률 (50% 미만: 굵게 + 남색)
                is_low = False
                try:
                    if rate_val is not None and float(rate_val) < 0.5:
                        is_low = True
                except:
                    pass

                if is_low:
                    c.setFillColor(navy)
                    c.setFont("NanumGothic-Bold", 8.9)
                else:
                    c.setFillColor(title_col)
                    c.setFont("NanumGothic", 8.6)
                c.drawCentredString(centers["wr"], ry + base, wr_txt)

                # Result
                if res_txt:
                    ox_color = red if res_txt == "X" else green
                    c.setFillColor(ox_color)
                    c.setFont("NanumGothic-Bold", 9.2)
                    c.drawCentredString(centers["res"], ry + base, res_txt)

                # 난이도
                c.setFillColor(title_col)
                c.setFont("NanumGothic-Bold", 8.6)
                c.drawCentredString(centers["diff"], ry + base, diff_txt)

                # 단원
                c.setFillColor(title_col)
                c.setFont("NanumGothic", 8.4)
                c.drawCentredString(centers["topic"], ry + base, topic_txt)

        draw_table(left_x, card_y, card_w, card_h, 1, ans_m1, wr_m1, wrong_m1)
        draw_table(right_x, card_y, card_w, card_h, 2, ans_m2, wr_m2, wrong_m2)

        # 하단 Topic 패널 (HTML 스타일)
        draw_topic_panel_html_style(
            c,
            panel_x, panel_y, panel_w, panel_h,
            topic_stats=topic_stats,
            diff_stats=diff_stats,
            stroke=stroke,
            title_col=title_col,
            muted=muted,
        )

        # footer
        if footer_left_text:
            c.setFillColor(title_col)
            c.setFont("NanumGothic", 8)
            lines = str(footer_left_text).splitlines()
            y0 = 10.5 * mm
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

    # ---------- 실행 버튼 ----------
    if st.button("🚀 Tab4 개인 성적표 생성", type="primary", key="t4_btn"):
        if not eta_file4 or not mock_file4:
            st.warning("⚠️ ETA.xlsx와 Mock데이터.xlsx를 모두 업로드해주세요.")
            st.stop()

        if not font_ready:
            st.error("⚠️ fonts 폴더에 NanumGothic.ttf / NanumGothicBold.ttf가 필요합니다.")
            st.stop()

        try:
            eta_xl = pd.ExcelFile(eta_file4)

            # Student Analysis
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

            # QuizResults
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

            # Accuracy / Error Analysis (정답률)
            target_sheet = None
            if "Accuracy Analysis" in eta_xl.sheet_names:
                target_sheet = "Accuracy Analysis"
            elif "Error Analysis" in eta_xl.sheet_names:
                target_sheet = "Error Analysis"

            if target_sheet:
                wr1, wr2 = build_wrong_rate_dict_fixed_ranges(eta_xl, target_sheet)
            else:
                wr1, wr2 = {}, {}

            # Mock Answers + Meta
            ans1, ans2, meta_topic, meta_diff = read_mock_answers_with_meta(mock_file4)

            # PDF 생성
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

                m1_score_txt = _clean(m1.get("score", ""))
                m2_score_txt = _clean(m2.get("score", ""))

                if m1_score_txt == "" or m2_score_txt == "":
                    skipped.append(stu)
                    prog.progress((i+1)/len(students))
                    continue

                subtitle_kw = _clean(m1.get("keyword", "")) or _clean(m2.get("keyword", "")) or "-"
                if subtitle_kw != "-" and common_subtitle == "-":
                    common_subtitle = subtitle_kw

                m1_meta = {"score": m1_score_txt, "dt": m1.get("dt", "-"), "time": m1.get("time", "-")}
                m2_meta = {"score": m2_score_txt, "dt": m2.get("dt", "-"), "time": m2.get("time", "-")}

                wrong1 = set(m1.get("wrong_set", set()))
                wrong2 = set(m2.get("wrong_set", set()))

                topic_stats, diff_stats = compute_topic_and_difficulty_stats(wrong1, wrong2, meta_topic, meta_diff)

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
                    meta_topic=meta_topic,
                    meta_diff=meta_diff,
                    topic_stats=topic_stats,
                    diff_stats=diff_stats,
                    result_blank=False,
                    footer_left_text=FOOTER_LEFT_TEXT,
                )

                made_files.append((stu, pdf_path))

                # PNG
                try:
                    png_bytes = render_pdf_first_page_to_png_bytes(pdf_path, zoom=2.0)
                    png_path = os.path.join(output_dir, f"{stu}_{generated_date4.strftime('%Y%m%d')}.png")
                    with open(png_path, "wb") as f:
                        f.write(png_bytes)
                    made_images.append((stu, png_path))
                except:
                    pass

                prog.progress((i+1)/len(students))

            # 템플릿 1개
            template_pdf = os.path.join(output_dir, f"Report_{generated_date4.strftime('%Y%m%d')}.pdf")
            template_topic_stats = {g: {"correct": 0, "total": 0} for g in range(1, 8)}
            template_diff_stats  = {d: {"correct": 0, "total": 0} for d in ["E", "M", "H"]}

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
                meta_topic=meta_topic,
                meta_diff=meta_diff,
                topic_stats=template_topic_stats,
                diff_stats=template_diff_stats,
                result_blank=True,
                footer_left_text=FOOTER_LEFT_TEXT,
            )
            made_files.append(("Report", template_pdf))

            try:
                png_bytes = render_pdf_first_page_to_png_bytes(template_pdf, zoom=2.0)
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

            # PDF ZIP
            pdf_zip_buf = io.BytesIO()
            with zipfile.ZipFile(pdf_zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for stu, path in made_files:
                    if os.path.exists(path):
                        z.write(path, arcname=os.path.basename(path))
            pdf_zip_buf.seek(0)

            # PNG ZIP
            img_zip_buf = io.BytesIO()
            with zipfile.ZipFile(img_zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for stu, path in made_images:
                    if os.path.exists(path):
                        z.write(path, arcname=os.path.basename(path))
            img_zip_buf.seek(0)

            st.success(f"✅ Tab4 생성 완료: PDF {len(made_files)}개 / 이미지 {len(made_images)}개 (제외: {len(skipped)}명)")
            if skipped:
                with st.expander(f"제외된 학생 ({len(skipped)}명) - 점수 blank"):
                    for s in skipped:
                        st.write(f"- {s}")

            st.download_button(
                "📦 Tab4 개인 성적표 PDF ZIP 다운로드",
                data=pdf_zip_buf,
                file_name=f"개인성적표_Tab4_PDF_{generated_date4.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="t4_download_pdf_zip"
            )

            st.download_button(
                "🖼️ Tab4 개인 성적표 이미지(PNG) ZIP 다운로드",
                data=img_zip_buf,
                file_name=f"개인성적표_Tab4_PNG_{generated_date4.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="t4_download_png_zip"
            )

        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.exception(e)
