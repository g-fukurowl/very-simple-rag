from llama_cpp import Llama
from colorama import Fore, Style, init
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from huggingface_hub import hf_hub_download
import argparse
import sys
import os
import shutil
import logging
from mcp.server.fastmcp import FastMCP
import io
#import cProfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("very_simple_rag")

mcp = FastMCP("very-simple-rag-mcp")

if getattr(sys, 'frozen', False):
    # PyInstallerでビルドされた実行環境
    SCRIPT_DIR_PATH = os.path.dirname(sys.executable)
else:
    # このスクリプト自身があるディレクトリの絶対パスを取得
    SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# HuggingFace上にアップロードされているembeddingモデル名
EMBEDDING_MODEL_PATH = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cpu" 
# 埋め込みモデルのロード。時間がかかる
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={"device": DEVICE},
)

# HuggingFace上にアップロードされているLLMのリポジトリ名
HF_REPO_NAME = "lmstudio-community/gemma-3-1B-it-qat-GGUF"
# ダウンロードしたいLLMのggufファイル名（大文字・小文字を正確に）
GGUF_FILE_NAME = "gemma-3-1B-it-QAT-Q4_0.gguf"
# ダウンロードした GGUF ファイルへのパス
MODEL_PATH = os.path.join(SCRIPT_DIR_PATH, "models", GGUF_FILE_NAME)

if os.path.exists(MODEL_PATH):
    # Llama インスタンスの生成
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=12000,         # コンテキスト長（トークン）
        n_threads=4,        # 並列スレッド数
        n_gpu_layers=-1,
        verbose=False
    )
else:
    print(f"{GGUF_FILE_NAME} is not found.")
    
# ベクトル化したいドキュメントをロードする
def load_documents(file_path: str):
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.lower().endswith(".csv"):
        loader = CSVLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return loader.load()

# ロードしたドキュメントをチャンク単位に分割
def split_documents(documents, chunk_size=1000, chunk_overlap=0.1):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# ドキュメントをベクトル化してストアする
def embed_and_store(
    raw_docs,
    model_name=EMBEDDING_MODEL_PATH,
    device=DEVICE,
    persist_path="faiss_index"
):
    # ステップ1: 生ドキュメント数を検証
    if not raw_docs:
        raise ValueError("No documents to embed. Check loader output.")
    # ステップ2: チャンク生成
    docs = split_documents(raw_docs, 100)
    print(raw_docs)
    # デバッグ出力
    print(f"Generated {len(docs)} chunks")
    # ステップ3: 空チャンクの除外
    docs = [doc for doc in docs if doc.page_content.strip()]
    if not docs:
        raise ValueError("All chunks are empty after filtering.")
    # ステップ4: 埋め込みモデルの準備
    # 冒頭で初期化済み

    # ステップ5: FAISS に格納
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

def update_vector():
    import glob

    # 例えば「data/sample.txt」というファイルを参照したい場合
    data_path = os.path.join(SCRIPT_DIR_PATH, 'data')
    file_paths = glob.glob(data_path+"/*.*")
    raw_docs = []
    for path in file_paths:
        raw_docs.extend(load_documents(path))
    # 修正版関数の呼び出し
    faiss_store = embed_and_store(
        raw_docs,
        model_name=EMBEDDING_MODEL_PATH,
        device=DEVICE,
        persist_path="faiss_index"
    )
    print(f"Indexed into FAISS at 'faiss_index'")


# あらかじめ作っておいたベクターストアをロード
def load_vectorstore(persist_path="faiss_index",
                     model_name="intfloat/multilingual-e5-large-instruct",
                     device=DEVICE):
    """保存済みFAISSインデックスと埋め込みモデルをロード"""

    vectorstore = FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def search_faiss(query: str, k: int = 5):
    """FAISSでクエリ検索"""
    persist_path = "faiss_index"
    vectorstore = load_vectorstore(persist_path, EMBEDDING_MODEL_PATH)

    print(f"# Searching for: {query}")
    results = vectorstore.similarity_search(query, k=k)

    return results

def chat(prompt: str,
         max_tokens: int = 2048,
         temperature: float = 0.8,
         top_p: float = 0.95,):
    """
    prompt（文字列）を渡して LLM で応答を生成し、
    生成テキストを返す。
    """
    out = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=0.05,
        top_k=40,
        repeat_penalty=1.1,
        stop=["\n\n\n\n"]  # 必要に応じてストップシーケンスを変更
    )
    print(out)
    return out["choices"][0]["text"].strip()

def run():
    """
    ワンショットのFAQを実行する。
    """
    init()
    query = input("# Query: ")
    search_result = search_faiss(query, k=2)
    search_result_str = ""
    for i, doc in enumerate(search_result, 1):
        search_result_str = search_result_str + f"\nResult #{i}"
        search_result_str = search_result_str + doc.page_content
        search_result_str = search_result_str + f"[Metadata] {doc.metadata}"

    prompt = f"### 指示 \nあなたは優秀なアシスタントAIです。常に日本語で応答します。質問「{query}」に簡潔に答えてください。その際、以下の情報を参照してください。\n\n### 情報 \n{search_result_str}\n\n\n\n"
    response = Fore.GREEN + chat(prompt) + Style.RESET_ALL
    print("# Assistant AI:", response)

def setup():
    """
    必要なLLMをダウンロードする
    """

    # 必要なら revision を指定（例: "main"）
    revision = "main"

    # 認証済みトークンを使う場合
    # - 環境変数 HF_TOKEN を設定済みなら use_auth_token=True
    # - 直接文字列を渡す場合は use_auth_token="hf_xxx..."
    path_to_hf_model_cache = hf_hub_download(
        repo_id=HF_REPO_NAME,
        filename=GGUF_FILE_NAME,
        revision=revision,
        use_auth_token=True,
    )
    print(path_to_hf_model_cache)

    path_to_dst_model_dir = os.path.join(SCRIPT_DIR_PATH, "models")
    os.makedirs(path_to_dst_model_dir, exist_ok=True)

    print(f"Moving: {path_to_hf_model_cache} -> {path_to_dst_model_dir}")
    shutil.move(path_to_hf_model_cache, path_to_dst_model_dir)


@mcp.tool()
async def semantic_search(query: str) -> str:
    """事前に構築したベクターストアから情報を検索します。

    Args:
        query: セマンティック検索用のクエリ。調べたい事柄を文として入力してください。
    """
    logger.info(f"Received search query: {query}")
    search_result = search_faiss(query, k=2)
    if not search_result:
        logger.warning("No results returned from FESS server")
        return "検索結果が見つかりませんでした。"
    
    return str(search_result)

def run_mcp_server():
    """
    MCPサーバーとして起動する
    """
    init()
    logger.info("Starting MCP server...")
    mcp.run(transport="sse")
    logger.info("MCP server stopped.")

def main():
    parser = argparse.ArgumentParser(description="Local LLM one‑shot Q&A with semantic vector retrieval.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # run コマンド
    subparsers.add_parser('run', help='1問1答式の対話を実行します。Queryに入力した内容を元にセマンティック検索を行い、これについてLLMが要約して回答します。modelsディレクトリに目的の.ggufファイルを配置してください')

    # update-vector コマンド
    subparsers.add_parser('update-vector', help='ベクトルの更新処理を実行します。dataディレクトリにcsv, pdf, txtなどを配置してください')

    # setup コマンド
    subparsers.add_parser('setup', help='セットアップを行います')

    # run-mcp-server コマンド
    subparsers.add_parser('run-mcp-server', help='セマンティック検索機能を持つMCPサーバーを起動します。')

    # search コマンド
    search_command = subparsers.add_parser('search', help='セマンティック検索機能のみを実行します。')
    search_command.add_argument('--query', type=str, help='検索クエリを入力してください。')
    search_command.add_argument('--k', type=int, default=5, help='検索結果の数を指定します。デフォルトは5です。')



    # 引数がない場合はヘルプを表示して終了
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    print(args)

    if args.command == 'run':
        run()
    elif args.command == 'update-vector':
        update_vector()
    elif args.command == 'setup':
        setup()
    elif args.command == 'run-mcp-server':
        run_mcp_server()
    elif args.command == 'search':
        if not args.query:
            print("Error: --query argument is required for search command.")
            sys.exit(1)
        if args.k <= 0:
            print("Error: --k must be a positive integer.")
            sys.exit(1)
        result = search_faiss(args.query, args.k)
        if not result:
            print("検索結果が見つかりませんでした")
        else:
            for i, doc in enumerate(result, 1):
                # もし doc.page_content が 30文字以下ならば、表示しない
                if len(doc.page_content) <= 30:
                    continue
                print(f"## Result {i}\n### Page Content \n{doc.page_content}\n\n### Metadata \n{doc.metadata}")
                print("\n-----------------------------------------------\n")


if __name__ == "__main__":
    #cProfile.run('main()')
    main()