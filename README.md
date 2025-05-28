以下が修正済みの **README.md** です。
すべての `python very_simple_rag.py` を `uv run very_simple_rag.py` に置き換えました。

---

# 🤖 Very Simple RAG — ローカルLLMによるシンプルな質問応答アプリ

**Very Simple RAG** は、PDF/TXT/CSV ファイルからセマンティック検索を行い、ローカルLLM（Gemma）で質問応答を実現する「超シンプルな RAG（Retrieval-Augmented Generation）」アプリケーションです。

> LangChain + HuggingFace + llama.cpp + FAISS による構成。
> すべてローカルで動作し、インターネット接続なしでも利用できます。

---

## ✅ 特徴

* 🔍 **意味検索対応**：ドキュメントを意味的に分割・検索（FAISS）
* 🧠 **ローカルLLMで応答**：Gemma（GGUF形式）で自然言語の回答生成
* 📄 **対応ファイル形式**：`.pdf`, `.txt`, `.csv`
* 💾 **インデックス永続化**：FAISS による高速な再検索が可能
* 🛠 **PyInstaller対応済み**：CLIツールとして単体バイナリ化可能

---

## 🧪 コマンド一覧

| コマンド                                      | 説明                                                      |
| ----------------------------------------- | ------------------------------------------------------- |
| `uv run very_simple_rag.py setup`         | HuggingFace から LLM (`.gguf`) をダウンロードして `models/` に配置します |
| `uv run very_simple_rag.py update-vector` | `data/` フォルダ内のファイルをベクトル化し、FAISS に保存します                  |
| `uv run very_simple_rag.py run`           | クエリに基づいて検索・要約し、1問1答形式でGemmaが応答します                       |

---

## 🚀 セットアップ手順

### 1. ライブラリのインストール

[**uv**](https://github.com/astral-sh/uv) を使って高速・依存関係解決付きでインストールできます。

```bash
uv pip install .
```

> `pyproject.toml` に定義された依存がインストールされます。

※ 事前に `uv` をインストールしていない場合は以下で導入できます：

```bash
pip install uv
```

---

### 2. モデルのダウンロード

```bash
uv run very_simple_rag.py setup
```

> モデル：`lmstudio-community/gemma-3-1B-it-qat-GGUF`
> 保存先：`models/gemma-3-1B-it-QAT-Q4_0.gguf`
> ※ Hugging Face のアクセストークン（環境変数 `HF_TOKEN`）が必要です

---

### 3. ドキュメントの配置とベクトル化

```bash
mkdir data
# → data に .pdf/.txt/.csv を配置
uv run very_simple_rag.py update-vector
```

---

### 4. 実行と応答確認

```bash
uv run very_simple_rag.py run
```

クエリを入力すると、検索結果をもとにGemmaが回答を生成します。

---

## ⚙ PyInstallerによる実行ファイル化

このスクリプトは PyInstaller に対応しており、　build.bat でスタンドアロン実行形式に変換できます。


---

## 📌 使用モデル情報

| 種別        | モデル名                                                                                                            |
| --------- | --------------------------------------------------------------------------------------------------------------- |
| LLM       | [`lmstudio-community/gemma-3-1B-it-qat-GGUF`](https://huggingface.co/lmstudio-community/gemma-3-1B-it-qat-GGUF) |
| Embedding | [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct)     |

---

## 📖 ライセンス

このリポジトリは MIT ライセンスです。使用するモデルのライセンスは Hugging Face にてご確認ください。

---

## 💡 補足

* 初回実行時はモデルとインデックス構築に時間がかかります。
* すべてローカルで動作するため、プライバシー性の高い利用にも適しています。

