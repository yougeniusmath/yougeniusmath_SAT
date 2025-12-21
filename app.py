import streamlit as st
import pandas as pd
import zipfile
import os
import io
import re
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import fitz  # PyMuPDF

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
        '학생 이름': ['홍길동', '김철수'],
        '[M1] 점수': [100, 90],
        '[M1] 틀린 문제': ['1,3,5', '2,4'],
        '[M2] 점수': [95, 85],
        '[M2] 틀린 문제': ['2,6', '1,3']
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
        img_est_height = 100
        if title == "<Module2>" and pdf.get_y() + 10 + (img_est_height if images else 0) > pdf.page_break_trigger:
            pdf.add_page()
        pdf.set_font(pdf_font_name, size=10)
        pdf.cell(0, 8, txt=title, ln=True)
        if images:
            for img in images:
                temp_filename = f"temp_{datetime.now().timestamp()}_{os.urandom(4).hex()}.jpg"
                img.save(temp_filename)
                pdf.image(temp_filename, w=180)
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
    r"(YOU,\s*GENIUS|700\+\s*MOCK\s*TEST|Kakaotalk|Instagram|010-\d{3,4}-\d{4})",
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
        
        # [수정] 페이지 내 모든 텍스트 블록을 미리 가져옴 (위쪽 글자 충돌 감지용)
        page_blocks = page.get_text("blocks") 

        mid = find_module_on_page(page)
        if mid is not None: current_module = mid
        if current_module not in (1, 2): continue

        anchors = detect_question_anchors(page)
        if not anchors: continue

        for i, (qnum, y0) in enumerate(anchors):
            # 1. 기본 위쪽 여백 계산
            y_start_candidate = clamp(y0 - pad_top, 0, h)
            
            # [수정] 위쪽 여백 공간에 다른 글자가 끼어있는지 확인 (헤더 방지 로직)
            # 번호(y0)보다 위에 있고, 우리가 자르려는 선(y_start_candidate)보다 아래에 끝나는 글자가 있으면
            # 그 글자 바로 밑으로 시작점을 내립니다.
            safe_y = y_start_candidate
            for b in page_blocks:
                # b = (x0, y0, x1, y1, text, block_no, line_no)
                b_y1 = b[3] # 글자 블록의 바닥 좌표
                b_text = b[4]
                
                # 헤더/푸터 힌트가 있는 텍스트는 무조건 피함
                if HEADER_FOOTER_HINT_RE.search(b_text):
                    if b_y1 < y0 and b_y1 > safe_y:
                        safe_y = b_y1 + 2 # 글자 2px 밑에서 자름
                else:
                    # 일반 텍스트라도 번호 바로 위의 여백 영역을 침범하면 피함
                    # (단, 번호 자체인 경우는 제외하기 위해 y0보다 확실히 위에 있는 것만 체크)
                    if b_y1 > safe_y and b_y1 < y0 - 2: 
                        safe_y = b_y1 + 2

            y_start = clamp(safe_y, 0, h)

            # 2. 아래쪽 여백 계산 (기존과 동일)
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

tab1, tab2 = st.tabs(["📝 오답노트 생성기", "✂️ PDF 문제 이미지"])

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

    st.markdown("---")
    st.subheader("📊 예시 엑셀 양식")
    
    # [수정됨] 클릭하면 표가 열리고, 다운로드 버튼이 있는 기존 스타일로 복구
    with st.expander("예시 엑셀파일 미리보기 (클릭하여 열기)"):
        st.dataframe(example_input_df(), use_container_width=True)
    
    example = get_example_excel()
    st.download_button(
        "📥 예시 엑셀파일 다운로드 (Mock결과_양식.xlsx)", 
        example, 
        file_name="Mock결과_양식.xlsx"
    )

    st.markdown("---")

    st.header("📄 문서 제목 입력")
    doc_title = st.text_input("문서 제목 (예: 25 S2 SAT MATH 만점반 Mock Test1)", value="25 S2 SAT MATH 만점반 Mock Test1")

    st.header("📦 파일 업로드")

    st.write("") 
    st.markdown("####  1. 문제 이미지 ZIP 파일")
    st.caption("`m1`, `m2` 폴더가 들어있는 ZIP 파일을 업로드해주세요.")
    img_zip = st.file_uploader("", type="zip", key="t1_zip") # 라벨을 위쪽 마크다운으로 대체하여 깔끔하게

    st.markdown("---") # 구분선

    st.markdown("####  2. 오답 현황 엑셀 파일")
    st.caption("학생들의 결과 데이터가 담긴 엑셀 파일을 업로드해주세요.")
    excel_file = st.file_uploader("", type="xlsx", key="t1_excel")

    st.write("") # 버튼과의 여백



    
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
                progress_bar = st.progress(0)
                
                for idx, row in df.iterrows():
                    name = row['이름']
                    if pd.isna(row['Module1']) or pd.isna(row['Module2']): continue

                    def to_list(x):
                        if pd.isna(x) or str(x).strip() in ["", "X", "x"]: return []
                        s = str(x).replace("，", ",").replace(";", ",")
                        return [t.strip() for t in s.split(",") if t.strip()]

                    m1_nums = to_list(row['Module1'])
                    m2_nums = to_list(row['Module2'])
                    m1_list = [m1_imgs[n] for n in m1_nums if n in m1_imgs]
                    m2_list = [m2_imgs[n] for n in m2_nums if n in m2_imgs]

                    pdf_path = create_student_pdf(name, m1_list, m2_list, doc_title, output_dir)
                    if pdf_path:
                        temp_files.append((name, pdf_path))
                    progress_bar.progress((idx + 1) / len(df))

                st.session_state.generated_files = temp_files

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

    # 다운로드 영역
    if st.session_state.generated_files:
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
                        file_name=f"{selected_student}_오답노트.pdf",
                        key="t1_down_indiv"
                    )

# ---------------------------------------------------------
# [Tab 2] PDF 문제 자르기
# ---------------------------------------------------------
with tab2:
    st.header("✂️문제캡처 ZIP생성기")
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
                    # PDF 파일 읽기
                    pdf_bytes = pdf_file.read()
                    pdf_name = pdf_file.name
                    zip_base = pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name

                    # 계산 로직 실행
                    doc_obj, rects_data = compute_rects_for_pdf(
                        pdf_bytes,
                        zoom=zoom_val,
                        pad_top=pt_val,
                        pad_bottom=pb_val,
                        frq_extra_space_px=frq_val,
                    )

                    # ZIP 생성
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

