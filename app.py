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
            self.add_font(pdf_font_name, '', FONT_REGULAR, uni=True)
            self.add_font(pdf_font_name, 'B', FONT_BOLD, uni=True)
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
        
        # case 2: "21" "."
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

tab1, tab2, tab3 = st.tabs(["📝 오답노트 생성기", "✂️ 문제캡처 ZIP생성기", "📊 개인 성적표"])

# ---------------------------------------------------------
# [Tab 1] 오답노트 생성기
# ---------------------------------------------------------
with tab1:
    st.header("📝 SAT 오답노트 생성기")

    # 세션 상태 초기화
    if 'generated_files' not in st.session_state:
        st.session_state.generated_files = []
    if 'zip_buffer' not in st.session_state:
        st.session_state.zip_buffer = None
    if 'skipped_details' not in st.session_state:
        st.session_state.skipped_details = {}

    st.markdown("---")
    st.subheader("📊 예시 엑셀 양식")
    
    with st.expander("예시 엑셀파일 미리보기 (클릭하여 열기)"):
        st.dataframe(example_input_df(), use_container_width=True)
    
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

    # [수정] 결과 표시 로직을 버튼 밖으로 빼서 다운로드 시에도 유지되게 함
    if st.session_state.generated_files or st.session_state.skipped_details:
        
        # 상세 결과 리포트 출력 (항상 보이게)
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
# [Tab 2] PDF 문제 자르기
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
# [Tab 3] 개인 성적표 (Student Analysis 학생목록 + QuizResults 메타 + Mock/오답률)
# ---------------------------------------------------------
with tab3:
    st.header("📊 개인 성적표")
    st.info("Student Analysis(학생목록) + QuizResults(Date/Time/Time/Score/Wrong/Keyword) + Mock데이터(정답) + Error Analysis(오답률)")

    eta_file = st.file_uploader("ETA 결과 파일 업로드 (ETA.xlsx)", type=["xlsx"], key="t3_eta")
    mock_file = st.file_uploader("Mock 정답 파일 업로드 (Mock데이터.xlsx)", type=["xlsx"], key="t3_mock")

    c1, c2 = st.columns([1, 1])
    with c1:
        report_title = st.text_input("리포트 제목", value="SAT Math Report", key="t3_title")
    with c2:
        generated_date = st.date_input("Generated 날짜", value=datetime.now().date(), key="t3_gen_date")

    st.caption("부제목은 QuizResults의 '검색 키워드'가 학생별로 자동으로 들어갑니다.")

    # =========================
    # 시트/헤더 규칙 (ETA(1).xlsx 기준)
    # =========================
    STUDENT_SHEET = "Student Analysis"
    QUIZ_SHEET = "QuizResults"
    ERROR_SHEET = "Error Analysis"

    SA_HEADER_ROW_IDX = 1  # ✅ Student Analysis: 2행이 헤더
    QZ_HEADER_ROW_IDX = 0  # ✅ QuizResults: 1행이 헤더

    # Student Analysis: 학생목록 ONLY
    SA_NAME_COL = "학생 이름"
    SA_M1_SCORE_COL = "[M1] 점수"
    SA_M2_SCORE_COL = "[M2] 점수"

    # QuizResults: 메타 ONLY (고정 컬럼명)
    QZ_KEYWORD_COL = "검색 키워드"
    QZ_MODULE_COL  = "모듈"
    QZ_NAME_COL    = "학생 이름"
    QZ_DT_COL      = "응답 날짜"
    QZ_TIME_COL    = "소요 시간"
    QZ_SCORE_COL   = "점수"
    QZ_WRONG_COL   = "틀린 문제 번호"

    # =========================
    # Helpers
    # =========================
    def _clean(x):
        if x is None: return ""
        if isinstance(x, float) and pd.isna(x): return ""
        return str(x).replace("\r", "").strip()

    def parse_wrong_list(val):
        """'1,3,5' 단순 문자열 (오답노트 생성기와 동일)"""
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

        def wr_to_text(v):
        """
        정답률 표시용 함수
        - None이면 '-'
        - 0% ~ 100% 모두 표시 (지우지 않음)
        """
            if v is None:
            return "-"
            try:
                v = float(v)
            # [수정] 정답률이므로 0%도 100%도 모두 의미가 있음. 무조건 표시.
            return f"{int(round(v * 100))}%"
            except:
                return "-"

    def score_to_slash22(s):
        """QuizResults 점수가 이미 '19 / 22'면 그대로, 아니면 '점수 / 22'"""
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

    # ✅ Error Analysis 정답률 고정 범위: M1=C3:C24, M2=C26:C47
    def build_wrong_rate_dict_fixed_ranges(eta_xl):
        df = pd.read_excel(eta_xl, sheet_name=ERROR_SHEET, header=None)
        colC = df.iloc[:, 2].tolist()  # C열

        m1_vals = colC[2:24]    # C3:C24 (22개)
        m2_vals = colC[25:47]   # C26:C47 (22개)

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
        """Mock데이터.xlsx 정답 '셀 그대로'(줄바꿈 유지)"""
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
                try: q = int(str(r[c0]).strip())
                except: continue
                out[q] = _clean(r[c1])
            return out, {}

        m2i = m2_idxs[0]
        m1_rows = df.iloc[:m2i]
        m2_rows = df.iloc[m2i+1:]

        def rows_to_ans(rows):
            dct={}
            for _, r in rows.iterrows():
                try: q = int(str(r[c0]).strip())
                except: continue
                dct[q] = _clean(r[c1])
            return dct

        return rows_to_ans(m1_rows), rows_to_ans(m2_rows)

    # =========================
    # ReportLab
    # =========================
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    def ensure_fonts_registered():
        # 중복 등록되어도 크게 문제 없게 try 처리
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
        c.drawString(x_center - tw/2, y_baseline, text)

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
):
    ensure_fonts_registered()
    c = canvas.Canvas(output_path, pagesize=A4)
    W, H = A4

    # === [디자인 컬러 팔레트: 인쇄 친화적 화이트톤] ===
    # 배경은 칠하지 않음 (기본 흰색)
    stroke = colors.Color(203/255, 213/255, 225/255)  # 연한 회색 테두리
    header_line = colors.Color(30/255, 41/255, 59/255) # 진한 네이비 (구분선)
    
    # 텍스트 컬러
    text_main = colors.Color(15/255, 23/255, 42/255)   # 거의 검정
    text_sub = colors.Color(100/255, 116/255, 139/255) # 연한 회색 텍스트
    
    # 정오 표시 컬러
    green = colors.Color(22/255, 101/255, 52/255)
    red = colors.Color(220/255, 38/255, 38/255)       # 조금 더 선명한 빨강
    
    # 테이블 행 배경 (가독성을 위한 아주 연한 줄무늬)
    row_stripe = colors.Color(248/255, 250/255, 252/255) 

    # 여백 설정
    L = 15*mm
    R = 15*mm
    TOP = H - 15*mm
    usable_w = W - L - R

    # 1. 문서 헤더 (심플하게 텍스트와 하단 라인만 사용)
    c.setFillColor(text_sub)
    c.setFont("NanumGothic", 9)
    c.drawRightString(W - R, TOP, f"Generated: {gen_date_str}")

    # 메인 타이틀
    c.setFillColor(text_main)
    c.setFont("NanumGothic-Bold", 24)
    c.drawString(L, TOP - 10*mm, title)

    # 부제 (키워드)
    c.setFillColor(text_sub)
    c.setFont("NanumGothic", 12)
    c.drawString(L, TOP - 17*mm, subtitle)

    # 학생 이름 (오른쪽에 크게 배치)
    c.setFillColor(text_main)
    c.setFont("NanumGothic-Bold", 16)
    c.drawRightString(W - R, TOP - 10*mm, student_name)
    
    # 헤더 구분선 (굵게)
    c.setLineWidth(1.5)
    c.setStrokeColor(header_line)
    line_y = TOP - 22*mm
    c.line(L, line_y, W - R, line_y)

    # 2. KPI 영역 (Module 1 / Module 2 점수)
    kpi_y = line_y - 10*mm
    kpi_h = 25*mm
    gap = 8*mm
    kpi_w = (usable_w - gap) / 2

    def draw_kpi_simple(x, y, w, h, label, score, dt, t):
        # 외곽선 박스
        c.setLineWidth(0.5)
        c.setStrokeColor(stroke)
        c.setFillColor(colors.white)
        c.roundRect(x, y, w, h, 3*mm, fill=1, stroke=1)
        
        # 라벨 (Module 1)
        c.setFillColor(text_sub)
        c.setFont("NanumGothic-Bold", 10)
        c.drawString(x + 5*mm, y + h - 8*mm, label)
        
        # 점수 (크게)
        c.setFillColor(text_main)
        c.setFont("NanumGothic-Bold", 20)
        c.drawRightString(x + w - 5*mm, y + h - 10*mm, str(score))
        
        # 하단 정보 (날짜/시간) - 구분선 추가
        c.setLineWidth(0.5)
        c.setStrokeColor(colors.Color(241/255, 245/255, 249/255))
        c.line(x + 3*mm, y + 9*mm, x + w - 3*mm, y + 9*mm)
        
        c.setFillColor(text_sub)
        c.setFont("NanumGothic", 9)
        c.drawString(x + 5*mm, y + 4*mm, f"Date: {dt}")
        c.drawRightString(x + w - 5*mm, y + 4*mm, f"Time: {t}")

    draw_kpi_simple(L, kpi_y, kpi_w, kpi_h, "Module 1 Results", m1_meta["score"], m1_meta["dt"], m1_meta["time"])
    draw_kpi_simple(L + kpi_w + gap, kpi_y, kpi_w, kpi_h, "Module 2 Results", m2_meta["score"], m2_meta["dt"], m2_meta["time"])

    # 3. 상세 분석 카드 (Analysis Cards)
    # KPI 바로 아래부터 시작
    cards_top = kpi_y - 8*mm 
    card_h = 200*mm # 충분히 길게
    card_y = cards_top - card_h

    def draw_analysis_list(x, y, w, h, module_name, ans_dict, wr_dict, wrong_set):
        # 전체 외곽선 (둥근 모서리 없이 깔끔하게, 혹은 아주 살짝 둥글게)
        c.setLineWidth(0.5)
        c.setStrokeColor(stroke)
        c.rect(x, y, w, h, stroke=1, fill=0)
        
        # 헤더 바 (네이비색 배경으로 강조)
        header_h = 10*mm
        c.setFillColor(header_line)
        c.rect(x, y + h - header_h, w, header_h, stroke=0, fill=1)
        
        c.setFillColor(colors.white)
        c.setFont("NanumGothic-Bold", 11)
        c.drawCentredString(x + w/2, y + h - 6.5*mm, module_name)
        
        # 내부 컬럼 헤더
        sub_header_y = y + h - header_h - 8*mm
        
        col_q = 10*mm
        col_wr = 14*mm
        col_ox = 10*mm
        col_ans = w - (col_q + col_wr + col_ox) # 나머지 공간
        
        # X 좌표 계산
        cx_q = x + col_q/2
        cx_ans = x + col_q + col_ans/2
        cx_wr = x + col_q + col_ans + col_wr/2
        cx_ox = x + col_q + col_ans + col_wr + col_ox/2
        
        c.setFillColor(text_sub)
        c.setFont("NanumGothic-Bold", 9)
        c.drawCentredString(cx_q, sub_header_y, "No.")
        c.drawCentredString(cx_ans, sub_header_y, "Answer")
        c.drawCentredString(cx_wr, sub_header_y, "정답률") # [변경] 오답률 -> 정답률
        c.drawCentredString(cx_ox, sub_header_y, "Result")
        
        # 구분선
        c.setStrokeColor(stroke)
        c.line(x + 2*mm, sub_header_y - 3*mm, x + w - 2*mm, sub_header_y - 3*mm)
        
        # 데이터 리스트
        row_h = 7.5*mm # 행 높이 약간 여유있게
        start_y = sub_header_y - 3*mm - row_h
        
        base_font_size = 10
        
        for i, q in enumerate(range(1, 23)):
            ry = start_y - i * row_h
            
            # 줄무늬 배경 (짝수행만)
            if q % 2 == 0:
                c.setFillColor(row_stripe)
                c.rect(x + 0.5, ry, w - 1, row_h, stroke=0, fill=1)
            
            # 데이터 준비
            ans_raw = _clean(ans_dict.get(q, ""))
            # 정답률 표시 (값이 없으면 -)
            rate_val = wr_dict.get(q, None)
            wr_txt = wr_to_text(rate_val) # 수정된 wr_to_text 사용

            ox = "X" if q in wrong_set else "O"
            
            # 텍스트 그리기 (수직 중앙 정렬)
            text_y = ry + 2.5*mm
            
            # 1. 문제 번호
            c.setFillColor(text_main)
            c.setFont("NanumGothic", base_font_size)
            c.drawCentredString(cx_q, text_y, str(q))
            
            # 2. 정답 (긴 텍스트 처리)
            lines = ans_raw.split("\n") if "\n" in ans_raw else [ans_raw]
            lines = [ln.strip() for ln in lines if ln.strip() != ""]
            if not lines: lines = [""]
            
            # 긴 텍스트 폰트 조절
            c.setFillColor(text_main)
            avail_w = col_ans - 2*mm
            
            if len(lines) == 1:
                fs = fit_font_size(lines[0], "NanumGothic-Bold", base_font_size, 7, avail_w)
                c.setFont("NanumGothic-Bold", fs)
                c.drawCentredString(cx_ans, text_y, lines[0])
            else:
                # 2줄인 경우
                fs = fit_font_size_two_lines(lines, "NanumGothic-Bold", 9, 6, avail_w)
                c.setFont("NanumGothic-Bold", fs)
                c.drawCentredString(cx_ans, text_y + 1.5*mm, lines[0])
                c.drawCentredString(cx_ans, text_y - 1.5*mm, lines[1])
            
            # 3. 정답률 (Accuracy)
            # 100%는 굵게, 나머지는 일반
            c.setFont("NanumGothic", base_font_size)
            c.setFillColor(text_main)
            c.drawCentredString(cx_wr, text_y, wr_txt)
            
            # 4. 정오 (O/X)
            ox_color = red if ox == "X" else green
            c.setFillColor(ox_color)
            c.setFont("NanumGothic-Bold", 11)
            c.drawCentredString(cx_ox, text_y, ox)

    draw_analysis_list(L, card_y, kpi_w, card_h, "Module 1 Analysis", ans_m1, wr_m1, wrong_m1)
    draw_analysis_list(L + kpi_w + gap, card_y, kpi_w, card_h, "Module 2 Analysis", ans_m2, wr_m2, wrong_m2)

    c.showPage()
    c.save()
    return output_path

    # =========================
    # Run
    # =========================
    if st.button("🚀 개인 성적표 생성", type="primary", key="t3_btn"):
        if not eta_file or not mock_file:
            st.warning("⚠️ ETA.xlsx와 Mock데이터.xlsx를 모두 업로드해주세요.")
            st.stop()

        if not font_ready:
            st.error("⚠️ 한글 PDF 생성을 위해 fonts 폴더에 NanumGothic.ttf / NanumGothicBold.ttf가 필요합니다.")
            st.stop()

        try:
            eta_xl = pd.ExcelFile(eta_file)

            # ---- Student Analysis: 학생목록 ONLY ----
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

            # ---- QuizResults: 1행 헤더 ----
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

            # {name: {1:{...}, 2:{...}}}
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

            # ---- Error Analysis 오답률 ----
            if ERROR_SHEET in eta_xl.sheet_names:
                wr1, wr2 = build_wrong_rate_dict_fixed_ranges(eta_xl)
            else:
                wr1, wr2 = {}, {}

            # ---- Mock 정답 ----
            ans1, ans2 = read_mock_answers(mock_file)

            # ---- PDF 생성 ----
            output_dir = "generated_reports"
            os.makedirs(output_dir, exist_ok=True)

            made_files = []
            skipped = []
            prog = st.progress(0)

            for i, stu in enumerate(students):
                q = quiz_map.get(stu, {})
                m1 = q.get(1, {})
                m2 = q.get(2, {})

                m1_score_txt = _clean(m1.get("score", ""))
                m2_score_txt = _clean(m2.get("score", ""))

                # ✅ M1/M2 점수 중 하나라도 blank면 제외
                if m1_score_txt == "" or m2_score_txt == "":
                    skipped.append(stu)
                    prog.progress((i+1)/len(students))
                    continue

                # ✅ 부제목: 검색 키워드 (M1 우선, 없으면 M2)
                subtitle_kw = _clean(m1.get("keyword", "")) or _clean(m2.get("keyword", "")) or "-"

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
                )

                made_files.append((stu, pdf_path))
                prog.progress((i+1)/len(students))

            if not made_files:
                st.warning("생성된 PDF가 없습니다. (QuizResults 점수 blank로 모두 제외되었을 수 있어요)")
                if skipped:
                    with st.expander(f"제외된 학생 ({len(skipped)}명) - 점수 blank"):
                        for s in skipped:
                            st.write(f"- {s}")
                st.stop()

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for stu, path in made_files:
                    if os.path.exists(path):
                        z.write(path, arcname=os.path.basename(path))
            zip_buf.seek(0)

            st.success(f"✅ 생성 완료: {len(made_files)}명 (제외: {len(skipped)}명)")
            if skipped:
                with st.expander(f"제외된 학생 ({len(skipped)}명) - 점수 blank"):
                    for s in skipped:
                        st.write(f"- {s}")

            st.download_button(
                "📦 개인 성적표 ZIP 다운로드",
                data=zip_buf,
                file_name=f"개인성적표_{generated_date.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="t3_download_zip"
            )

        except ModuleNotFoundError as e:
            st.error("❌ reportlab이 설치되어 있지 않습니다. (requirements.txt에 reportlab 추가 필요)")
            st.exception(e)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.exception(e)
