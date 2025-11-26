import argparse
import os
import shutil
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

from estimate_cost import estimate_embedding_cost


load_dotenv()

DEFAULT_MARKDOWN_FILE = os.path.join("data", "HS_code_Nomenclature.md")
DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_MODEL = "text-embedding-3-large"


def main() -> None:
    args = parse_arguments()
    process_markdown_file(
        markdown_file=args.markdown_file,
        chroma_db_path=args.chroma_db_path,
        model=args.model,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chroma DB from a Markdown document."
    )
    parser.add_argument(
        "--markdown_file",
        type=str,
        default=DEFAULT_MARKDOWN_FILE,
        help="Path to the Markdown file to embed.",
    )
    parser.add_argument(
        "--chroma_db_path",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help="Directory where the Chroma DB will be stored.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Embedding model name. OpenAI models (e.g., text-embedding-3-large) or SentenceTransformer models (e.g., Salesforce/SFR-Embedding-Mistral).",
    )
    return parser.parse_args()


def process_markdown_file(markdown_file: str, chroma_db_path: str, model: str) -> None:
    documents = load_markdown(markdown_file)
    if not documents:
        print(f"문서 로딩 실패: {markdown_file}")
        return

    chunks = split_text(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    cost, error_code = estimate_embedding_cost(markdown_file, model)
    if error_code == -1:
        print("임베딩 비용 추정 중 오류가 발생했습니다.")
        return
    print(f"Estimated embedding cost: ${cost:.3f}")

    input("계속하려면 Enter 키를 누르세요...")
    print(f"Processing {len(chunks)} chunks...")

    save_to_chroma(chunks, chroma_db_path, model)
    print(f"Saved {len(chunks)} chunks to {chroma_db_path}.")


def load_markdown(markdown_file: str) -> List[Document]:
    if not os.path.isfile(markdown_file):
        print(f"Markdown 파일을 찾을 수 없습니다: {markdown_file}")
        return []

    with open(markdown_file, "r", encoding="utf-8") as file:
        content = file.read()

    document = Document(page_content=content, metadata={"source": markdown_file})
    return [document]


def split_text(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def _is_openai_embedding_model(model_name: str) -> bool:
    """Check if the model is an OpenAI embedding model."""
    return model_name.startswith("text-embedding-")


def save_to_chroma(chunks: List[Document], chroma_db_path: str, model: str) -> None:
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)

    # 모델 타입에 따라 적절한 임베딩 클래스 선택
    if _is_openai_embedding_model(model):
        embedding_model = OpenAIEmbeddings(model=model)
        print(f"Using OpenAI embedding model: {model}")
    else:
        # SentenceTransformer 모델 (예: Salesforce/SFR-Embedding-Mistral)
        embedding_model = SentenceTransformerEmbeddings(
            model_name=model,
            encode_kwargs={
                "normalize_embeddings": True
            }
        )
        print(f"Using SentenceTransformer embedding model: {model}")

    Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=chroma_db_path,
    )


if __name__ == "__main__":
    main()


# OpenAI 모델 사용 예시:
# python RAG_embedding/nomenclature_chroma_embedding.py --markdown_file data/HS_code_Nomenclature.md --chroma_db_path data/chroma_db_nomenclature --model text-embedding-3-large

# Salesforce/SFR-Embedding-Mistral 모델 사용 예시 (4092차원):
# python RAG_embedding/nomenclature_chroma_embedding.py --markdown_file data/HS_code_Nomenclature.md --chroma_db_path data/chroma_db_nomenclature_mistral --model Salesforce/SFR-Embedding-Mistral
