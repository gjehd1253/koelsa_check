# build_kb.py
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os

# 1) 사용할 PDF 파일 경로
PDF_FILES = [
    "검사방법 표준화.pdf",
    "승강기 안전기준 연혁집.pdf",
]

# 2) 임베딩 모델 (한국어 지원)
MODEL_NAME = "jhgan/ko-sroberta-multitask"  # 인터넷 필요
model = SentenceTransformer(MODEL_NAME)

def extract_chunks_from_pdf(path, max_chars=800):
    """
    PDF에서 텍스트를 추출해 적당한 길이의 문단(chunk)으로 자른다.
    """
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            # 너무 긴 페이지는 max_chars 단위로 잘라서 저장
            for i in range(0, len(text), max_chars):
                part = text[i:i+max_chars].strip()
                if len(part) < 50:
                    continue
                chunks.append({
                    "source": os.path.basename(path),
                    "page": page_num,
                    "text": part,
                })
    return chunks

def build_knowledge_base():
    all_chunks = []
    for pdf in PDF_FILES:
        if not os.path.exists(pdf):
            print(f"[경고] 파일을 찾을 수 없음: {pdf}")
            continue
        print(f"[INFO] PDF에서 텍스트 추출 중... {pdf}")
        chunks = extract_chunks_from_pdf(pdf)
        all_chunks.extend(chunks)

    print(f"[INFO] 총 {len(all_chunks)}개 문단 chunk 생성")

    # 텍스트 임베딩 벡터 계산
    texts = [c["text"] for c in all_chunks]
    print("[INFO] 임베딩 계산 중...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    kb = {
        "chunks": all_chunks,   # 원문
        "embeddings": embeddings,  # numpy 배열
        "model_name": MODEL_NAME,
    }

    with open("kb_standards.pkl", "wb") as f:
        pickle.dump(kb, f)

    print("[완료] kb_standards.pkl 파일로 저장되었습니다.")

if __name__ == "__main__":
    build_knowledge_base()