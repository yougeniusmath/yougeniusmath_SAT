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
# [Tab 3] 개인 성적표  (ReportLab 버전)
# ---------------------------------------------------------
with tab3:
    st.header("📊 개인 성적표")
    st.info("ETA.xlsx(Student Analysis) + Mock데이터.xlsx(정답) 기반으로 학생별 성적표 PDF 생성 → ZIP 다운로드")

    eta_file = st.file_uploader("ETA 결과 파일 업로드 (ETA.xlsx)", type=["xlsx"], key="t3_eta")
    mock_file = st.file_uploader("Mock 정답 파일 업로드 (Mock데이터.xlsx)", type=["xlsx"], key="t3_mock")

    c1, c2 = st.columns([1, 1])
    with c1:
        report_title = st.text_input("리포트 제목", value="SAT Math Report", key="t3_title")
    with c2:
        generated_date = st.date_input("Generated 날짜", value=datetime.now().date(), key="t3_gen_date")

    subtitle = st.text_input("부제목(키워드)", value="25 S2 SAT MATH 만점반 Mock Test1", key="t3_subtitle")

    # =========================
    # Helpers (Parsing)
    # =========================
    def _clean(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        return str(x).replace("\r", "").strip()

    def _is_blank(x) -> bool:
        return _clean(x) == ""

    def parse_wrong_list(val):
        if pd.isna(val): return set()
        s = str(val).strip()
        if s == "" or s.upper() in ["X", "Х", "-"]:
            return set()
        s = s.replace("，", ",").replace(";", ",")
        out=set()
        for p in re.split(r"[,\s]+", s):
            p=p.strip()
            if not p: continue
            try:
                out.add(int(p))
            except:
                pass
        return out

    def safe_find_col(df: pd.DataFrame, must_have=None, any_of=None):
        """
        must_have: list[str] - 모두 포함되어야 함
        any_of: list[str] - 하나라도 포함되면 OK
        """
        if df is None or df.empty:
            return None
        cols = list(df.columns)
        norm = {c: str(c).lower().replace(" ", "") for c in cols}

        def ok(cn):
            if must_have:
                for t in must_have:
                    if t.lower().replace(" ", "") not in cn:
                        return False
            if any_of:
                return any(t.lower().replace(" ", "") in cn for t in any_of)
            return True

        for c in cols:
            if ok(norm[c]):
                return c
        return None

    def find_module_score_cols(student_df: pd.DataFrame):
        """
        Student Analysis에서 M1/M2 점수 컬럼을 최대한 유연하게 찾는다.
        """
        # M1 점수
        m1_score_col = safe_find_col(student_df, must_have=["m1"], any_of=["score", "점수"])
        if m1_score_col is None:
            m1_score_col = safe_find_col(student_df, must_have=["module1"], any_of=["score", "점수"])
        if m1_score_col is None:
            m1_score_col = safe_find_col(student_df, must_have=["모듈1"], any_of=["score", "점수"])

        # M2 점수
        m2_score_col = safe_find_col(student_df, must_have=["m2"], any_of=["score", "점수"])
        if m2_score_col is None:
            m2_score_col = safe_find_col(student_df, must_have=["module2"], any_of=["score", "점수"])
        if m2_score_col is None:
            m2_score_col = safe_find_col(student_df, must_have=["모듈2"], any_of=["score", "점수"])

        return m1_score_col, m2_score_col

    def find_module_meta_cols(student_df: pd.DataFrame):
        """
        Student Analysis에서 (가능하면) DateTime/Time/Wrong 컬럼도 찾는다.
        없으면 '-' 처리.
        """
        def pick(mod_tag, any_of):
            # 예: mod_tag="m1", any_of=["date","응답","datetime"]
            c = safe_find_col(student_df, must_have=[mod_tag], any_of=any_of)
            if c is None and mod_tag.startswith("m"):
                c = safe_find_col(student_df, must_have=[f"module{mod_tag[-1]}"], any_of=any_of)
            if c is None and mod_tag.startswith("m"):
                c = safe_find_col(student_df, must_have=[f"모듈{mod_tag[-1]}"], any_of=any_of)
            return c

        m1_dt = pick("m1", ["date", "datetime", "응답", "시간"])
        m2_dt = pick("m2", ["date", "datetime", "응답", "시간"])
        m1_time = pick("m1", ["time", "duration", "소요"])
        m2_time = pick("m2", ["time", "duration", "소요"])
        m1_wrong = pick("m1", ["wrong", "오답", "틀린"])
        m2_wrong = pick("m2", ["wrong", "오답", "틀린"])
        return m1_dt, m2_dt, m1_time, m2_time, m1_wrong, m2_wrong

    def build_wrong_rate_dict(error_df: pd.DataFrame):
        if error_df is None or error_df.empty:
            return {}, {}

        col_q = error_df.columns[0]
        wr_col = None
        for c in error_df.columns:
            cs = str(c).lower()
            if "wrong" in cs or "오답" in cs:
                wr_col = c
                break
        if wr_col is None:
            wr_col = error_df.columns[2] if len(error_df.columns) >= 3 else error_df.columns[-1]

        m2_idxs = error_df.index[error_df[col_q].astype(str).str.contains("M2", case=False, na=False)].tolist()
        if not m2_idxs:
            return {}, {}
        m2_start = m2_idxs[0]

        m1_rows = error_df.iloc[1:m2_start]
        m2_rows = error_df.iloc[m2_start+1:m2_start+23]

        def rows_to_dict(rows):
            dct={}
            for _, r in rows.iterrows():
                try:
                    q = int(str(r[col_q]).strip())
                except:
                    continue
                v = r.get(wr_col, None)
                try:
                    v = float(v)
                except:
                    v = None
                dct[q] = v
            return dct

        return rows_to_dict(m1_rows), rows_to_dict(m2_rows)

    def wr_to_text(v):
        if v is None:
            return "-"
        try:
            v = float(v)
            if abs(v) < 1e-12:
                return "-"  # 0% -> '-'
            return f"{int(round(v*100))}%"
        except:
            return "-"

    def read_mock_answers(mock_bytes) -> tuple[dict, dict]:
        df = pd.read_excel(mock_bytes)
        cols = set(df.columns.astype(str))

        # canonical
        if {"모듈", "문항번호", "정답"}.issubset(cols):
            m1 = df[df["모듈"].astype(str).str.upper().eq("M1")].set_index("문항번호")["정답"].astype(str).to_dict()
            m2 = df[df["모듈"].astype(str).str.upper().eq("M2")].set_index("문항번호")["정답"].astype(str).to_dict()
            m1 = {int(k): _clean(v).replace("\r", "") for k, v in m1.items() if str(k).strip().isdigit()}
            m2 = {int(k): _clean(v).replace("\r", "") for k, v in m2.items() if str(k).strip().isdigit()}
            return m1, m2

        # marker
        c0, c1 = df.columns[0], df.columns[1]
        m2_idxs = df.index[df[c0].astype(str).str.contains("Module2", case=False, na=False)].tolist()
        if not m2_idxs:
            out={}
            for _, r in df.iterrows():
                try:
                    q = int(str(r[c0]).strip())
                except:
                    continue
                out[q] = _clean(r[c1]).replace("\r", "")
            return out, {}

        m2i = m2_idxs[0]
        m1_rows = df.iloc[:m2i]
        m2_rows = df.iloc[m2i+1:]

        def rows_to_ans(rows):
            dct={}
            for _, r in rows.iterrows():
                try:
                    q = int(str(r[c0]).strip())
                except:
                    continue
                dct[q] = _clean(r[c1]).replace("\r", "")
            return dct

        return rows_to_ans(m1_rows), rows_to_ans(m2_rows)

    # =========================
    # ReportLab PDF rendering
    # =========================
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    def ensure_fonts_registered():
        # NanumGothic 등록 (한글)
        pdfmetrics.registerFont(TTFont("NanumGothic", FONT_REGULAR))
        pdfmetrics.registerFont(TTFont("NanumGothic-Bold", FONT_BOLD))

    def str_w(text, font_name, font_size):
        return pdfmetrics.stringWidth(text, font_name, font_size)

    def fit_font_size(text, font_name, max_size, min_size, max_width):
        """한 줄 문자열이 max_width 안에 들어가도록 폰트 크기 축소"""
        s = max_size
        while s >= min_size:
            if str_w(text, font_name, s) <= max_width:
                return s
            s -= 0.5
        return min_size

    def fit_font_size_two_lines(lines, font_name, max_size, min_size, max_width):
        """두 줄 모두 같은 폰트 크기로, 둘 다 폭 제한 만족하도록"""
        if not lines:
            return max_size
        need = max_size
        for ln in lines:
            if ln.strip() == "":
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

        # Colors
        bg = colors.Color(248/255, 250/255, 252/255)
        stroke = colors.Color(226/255, 232/255, 240/255)
        muted = colors.Color(71/255, 85/255, 105/255)
        title_col = colors.Color(15/255, 23/255, 42/255)
        pill_fill = colors.Color(241/255, 245/255, 249/255)
        stripe = colors.Color(248/255, 250/255, 252/255)
        green = colors.Color(22/255, 101/255, 52/255)
        red = colors.Color(153/255, 27/255, 27/255)

        # Background
        c.setFillColor(bg)
        c.rect(0, 0, W, H, stroke=0, fill=1)

        # Margins
        L = 18*mm
        R = 18*mm
        TOP = H - 18*mm
        usable_w = W - L - R

        # Generated (top-right)
        c.setFont("NanumGothic", 9.5)
        c.setFillColor(colors.Color(100/255, 116/255, 139/255))
        gen_text = f"Generated: {gen_date_str}"
        c.drawRightString(W - R, TOP, gen_text)

        # Header card
        header_h = 42*mm
        header_y = TOP - 8*mm - header_h
        draw_round_rect(c, L, header_y, usable_w, header_h, 10*mm, colors.white, stroke, 1)

        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 26)
        c.drawString(L + 10*mm, header_y + header_h - 18*mm, title)

        c.setFillColor(muted)
        c.setFont("NanumGothic", 13)
        c.drawString(L + 10*mm, header_y + header_h - 28*mm, subtitle)

        # Name pill (label left, value right)
        pill_w = 78*mm
        pill_h = 18*mm
        pill_x = L + usable_w - pill_w - 10*mm
        pill_y = header_y + 12*mm
        draw_round_rect(c, pill_x, pill_y, pill_w, pill_h, 7*mm, pill_fill, stroke, 1)

        c.setFillColor(colors.Color(100/255, 116/255, 139/255))
        c.setFont("NanumGothic-Bold", 9.5)
        c.drawString(pill_x + 6*mm, pill_y + 11*mm, "Name")

        # name RIGHT aligned
        c.setFillColor(colors.Color(2/255, 6/255, 23/255))
        c.setFont("NanumGothic-Bold", 15)
        c.drawRightString(pill_x + pill_w - 6*mm, pill_y + 4.8*mm, student_name)

        # KPI cards
        kpi_y = header_y - 8*mm - 24*mm
        kpi_h = 24*mm
        gap = 6*mm
        kpi_w = (usable_w - gap) / 2

        def draw_kpi_card(x, y, label, score, dt, t):
            draw_round_rect(c, x, y, kpi_w, kpi_h, 8*mm, colors.white, stroke, 1)
            c.setFillColor(colors.Color(2/255, 6/255, 23/255))
            c.setFont("NanumGothic-Bold", 12)
            c.drawString(x + 7*mm, y + kpi_h - 9*mm, label)

            c.setFont("NanumGothic-Bold", 20)
            c.setFillColor(title_col)
            c.drawRightString(x + kpi_w - 7*mm, y + kpi_h - 16*mm, str(score))

            c.setFont("NanumGothic", 9)
            c.setFillColor(muted)
            c.drawString(x + 7*mm, y + 5.5*mm, f"Date/Time  {dt}")
            c.drawRightString(x + kpi_w - 7*mm, y + 5.5*mm, f"Time  {t}")

        draw_kpi_card(L, kpi_y, "Module 1", m1_meta.get("score","-"), m1_meta.get("dt","-"), m1_meta.get("time","-"))
        draw_kpi_card(L + kpi_w + gap, kpi_y, "Module 2", m2_meta.get("score","-"), m2_meta.get("dt","-"), m2_meta.get("time","-"))

        # Section
        sec_y = kpi_y - 10*mm
        c.setFillColor(title_col)
        c.setFont("NanumGothic-Bold", 14)
        c.drawString(L, sec_y, "문항별 분석")

        c.setStrokeColor(stroke)
        c.setLineWidth(1.5)
        c.line(L, sec_y - 4*mm, W - R, sec_y - 4*mm)

        # Analysis cards (two columns)
        cards_top = sec_y - 10*mm
        card_h = 120*mm
        card_w = (usable_w - gap) / 2
        left_x = L
        right_x = L + card_w + gap
        card_y = cards_top - card_h

        def draw_analysis_card(x, y, title_txt, ans_dict, wr_dict, wrong_set):
            draw_round_rect(c, x, y, card_w, card_h, 10*mm, colors.white, stroke, 1)

            # Title
            c.setFillColor(title_col)
            c.setFont("NanumGothic-Bold", 15)
            c.drawString(x + 9*mm, y + card_h - 14*mm, title_txt)

            # Header strip
            strip_h = 12*mm
            strip_y = y + card_h - 28*mm
            draw_round_rect(c, x + 7*mm, strip_y, card_w - 14*mm, strip_h, 6*mm, pill_fill, stroke, 1)

            # column layout
            inner_x = x + 9*mm
            inner_w = card_w - 18*mm
            col_q = 12*mm
            col_wr = 18*mm
            col_ox = 12*mm
            col_ans = inner_w - (col_q + col_wr + col_ox)

            # centers (for header)
            q_center = inner_x + col_q/2
            ans_center = inner_x + col_q + col_ans/2
            wr_center = inner_x + col_q + col_ans + col_wr/2
            ox_center = inner_x + col_q + col_ans + col_wr + col_ox/2

            # header text CENTER in each column
            header_y = strip_y + 3.5*mm
            c.setFillColor(muted)
            c.setFont("NanumGothic-Bold", 10.5)
            draw_text_center(c, q_center, header_y, "문항", "NanumGothic-Bold", 10.5, muted)
            draw_text_center(c, ans_center, header_y, "정답", "NanumGothic-Bold", 10.5, muted)
            draw_text_center(c, wr_center, header_y, "오답률", "NanumGothic-Bold", 10.5, muted)
            draw_text_center(c, ox_center, header_y, "정오", "NanumGothic-Bold", 10.5, muted)

            # rows
            row_h = 9.5*mm
            start_y = strip_y - 3*mm - row_h
            for i, q in enumerate(range(1, 23)):
                ry = start_y - i*(row_h + 1.6*mm)
                if ry < y + 8*mm:
                    break

                # row pill bg
                fill = stripe if (q % 2 == 0) else colors.white
                # outline 없는 pill 느낌
                c.setFillColor(fill)
                c.setStrokeColor(fill)
                c.roundRect(x + 7*mm, ry, card_w - 14*mm, row_h, 6*mm, fill=1, stroke=0)

                # values
                ans_raw = _clean(ans_dict.get(q, ""))
                lines = ans_raw.split("\n") if "\n" in ans_raw else [ans_raw]
                lines = [ln.strip() for ln in lines if ln.strip() != ""]
                if not lines:
                    lines = [""]
                if len(lines) > 2:
                    # 2줄까지만: 나머지는 2번째 줄에 합치기
                    lines = [lines[0], " ".join(lines[1:])]

                wr_txt = wr_to_text(wr_dict.get(q, None))
                ox = "X" if q in wrong_set else "O"

                # Q
                c.setFillColor(title_col)
                c.setFont("NanumGothic", 11.5)
                draw_text_center(c, q_center, ry + 3.0*mm, str(q), "NanumGothic", 11.5, title_col)

                # Answer (CENTER, auto shrink, two lines same font size)
                ans_max_w = col_ans - 4*mm
                base_size = 11.5
                min_size = 7.0
                fsize = fit_font_size_two_lines(lines, "NanumGothic-Bold", base_size, min_size, ans_max_w)
                # vertical placement
                if len(lines) == 1:
                    draw_text_center(c, ans_center, ry + 3.0*mm, lines[0], "NanumGothic-Bold", fsize, title_col)
                else:
                    draw_text_center(c, ans_center, ry + 4.2*mm, lines[0], "NanumGothic-Bold", fsize, title_col)
                    draw_text_center(c, ans_center, ry + 1.4*mm, lines[1], "NanumGothic-Bold", fsize, title_col)

                # Wrong rate (CENTER)
                draw_text_center(c, wr_center, ry + 3.0*mm, wr_txt, "NanumGothic", 11.5, title_col)

                # O/X (text only, CENTER)
                ox_color = red if ox == "X" else green
                draw_text_center(c, ox_center, ry + 3.0*mm, ox, "NanumGothic-Bold", 11.5, ox_color)

        draw_analysis_card(left_x, card_y, "Module 1", ans_m1, wr_m1, wrong_m1)
        draw_analysis_card(right_x, card_y, "Module 2", ans_m2, wr_m2, wrong_m2)

        c.showPage()
        c.save()
        return output_path

    # =========================
    # Main action
    # =========================
    if st.button("🚀 개인 성적표 생성", type="primary", key="t3_btn"):
        if not eta_file or not mock_file:
            st.warning("⚠️ ETA.xlsx와 Mock데이터.xlsx를 모두 업로드해주세요.")
            st.stop()

        if not font_ready:
            st.error("⚠️ 한글 PDF 생성을 위해 fonts 폴더에 NanumGothic.ttf / NanumGothicBold.ttf가 필요합니다.")
            st.stop()

        try:
            # 1) Load ETA
            eta_xl = pd.ExcelFile(eta_file)
            if "Student Analysis" not in eta_xl.sheet_names:
                st.error("⚠️ ETA.xlsx에 'Student Analysis' 시트가 없습니다. (학생 목록은 Student Analysis 기준 ONLY)")
                st.stop()

            student_df = pd.read_excel(eta_xl, sheet_name="Student Analysis")
            error_df_ = pd.read_excel(eta_xl, sheet_name="Error Analysis") if "Error Analysis" in eta_xl.sheet_names else None

            # 2) Student list from Student Analysis only
            name_col = safe_find_col(student_df, any_of=["학생", "이름", "name", "student"])
            if name_col is None:
                st.error("⚠️ Student Analysis에서 학생 이름 컬럼을 찾지 못했습니다.")
                st.stop()

            m1_score_col, m2_score_col = find_module_score_cols(student_df)
            if m1_score_col is None or m2_score_col is None:
                st.error("⚠️ Student Analysis에서 M1/M2 점수 컬럼을 찾지 못했습니다. (컬럼명에 M1/M2 + 점수/score 포함 필요)")
                st.stop()

            m1_dt_col, m2_dt_col, m1_time_col, m2_time_col, m1_wrong_col, m2_wrong_col = find_module_meta_cols(student_df)

            # 3) Wrong-rate dicts
            wr1, wr2 = build_wrong_rate_dict(error_df_) if error_df_ is not None else ({}, {})

            # 4) Mock answers
            ans1, ans2 = read_mock_answers(mock_file)

            # 5) Build students + skip blanks (둘 중 하나라도 blank면 제외)
            students_all = [_clean(x) for x in student_df[name_col].dropna().tolist()]
            students_all = [s for s in students_all if s != ""]
            if not students_all:
                st.error("학생 목록이 비어있습니다.")
                st.stop()

            output_dir = "generated_reports"
            os.makedirs(output_dir, exist_ok=True)

            made_files = []
            skipped = []  # excluded due to blank scores
            prog = st.progress(0)

            for i, stu in enumerate(students_all):
                row = student_df[student_df[name_col].astype(str).str.strip().eq(stu)].head(1)
                if row.empty:
                    prog.progress((i+1)/len(students_all))
                    continue
                rr = row.iloc[0]

                m1_score = _clean(rr.get(m1_score_col, ""))
                m2_score = _clean(rr.get(m2_score_col, ""))

                if m1_score == "" or m2_score == "":
                    skipped.append(stu)
                    prog.progress((i+1)/len(students_all))
                    continue

                m1_dt = "-" if (m1_dt_col is None) else _clean(rr.get(m1_dt_col, "-"))
                m2_dt = "-" if (m2_dt_col is None) else _clean(rr.get(m2_dt_col, "-"))
                m1_time = "-" if (m1_time_col is None) else _clean(rr.get(m1_time_col, "-"))
                m2_time = "-" if (m2_time_col is None) else _clean(rr.get(m2_time_col, "-"))
                wrong1 = set() if (m1_wrong_col is None) else parse_wrong_list(rr.get(m1_wrong_col, ""))
                wrong2 = set() if (m2_wrong_col is None) else parse_wrong_list(rr.get(m2_wrong_col, ""))

                m1_meta = {"score": m1_score, "dt": m1_dt or "-", "time": m1_time or "-"}
                m2_meta = {"score": m2_score, "dt": m2_dt or "-", "time": m2_time or "-"}

                pdf_path = os.path.join(output_dir, f"{stu}_{generated_date.strftime('%Y%m%d')}.pdf")

                create_report_pdf_reportlab(
                    output_path=pdf_path,
                    title=report_title,
                    subtitle=subtitle,
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
                prog.progress((i+1)/len(students_all))

            # 6) ZIP
            if not made_files:
                st.warning("생성된 PDF가 없습니다. (M1/M2 점수 blank로 모두 제외되었을 수 있어요)")
                if skipped:
                    with st.expander(f"제외된 학생 ({len(skipped)}명)"):
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
                with st.expander(f"제외된 학생 ({len(skipped)}명) - M1/M2 점수 blank"):
                    for s in skipped:
                        st.write(f"- {s}")

            st.download_button(
                "📦 개인 성적표 ZIP 다운로드",
                data=zip_buf,
                file_name=f"개인성적표_{generated_date.strftime('%Y%m%d')}.zip",
                mime="application/zip",
                key="t3_download_zip"
            )

            # 개별 다운로드
            st.subheader("👁️ 개별 PDF 다운로드")
            student_names = [n for n, _ in made_files]
            selected = st.selectbox("학생 선택", student_names, key="t3_pick")
            if selected:
                mp = {n:p for n,p in made_files}
                pth = mp[selected]
                if os.path.exists(pth):
                    with open(pth, "rb") as f:
                        st.download_button(
                            f"📄 '{selected}' PDF 다운로드",
                            data=f,
                            file_name=os.path.basename(pth),
                            key="t3_down_one"
                        )

        except Exception as e:
            st.error(f"오류 발생: {e}")
