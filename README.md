# 🤖 Very Simple RAG — ローカルLLMによるシンプルな質問応答アプリ

**Very Simple RAG** は、PDF/TXT/CSV ファイルからセマンティック検索を行う、超シンプルな RAG（Retrieval-Augmented Generation）向けツールです。

> LangChain + HuggingFace + llama.cpp + FAISS による構成。
> すべてローカルで動作し、インターネット接続なしでも利用できます。
> MCPサーバーとして動作させることも可能です。

---

## ✅ 特徴

* 🔍 **意味検索対応**：ドキュメントを意味的に分割・検索
* 🧠 **ローカル埋め込みモデル**：HFに公開されているオープンな埋め込みモデルをローカル環境で利用
* 📄 **対応ファイル形式**：`.pdf`, `.txt`, `.csv`
* 💾 **インデックス永続化**：FAISS による高速な再検索が可能
* 🤖 **MCPサーバモード**：MCPサーバーとして動作させることも可能。セマンティック検索ベースのRAGに対応
* 🛠 **PyInstaller対応済み**：CLIツールとして単体バイナリ化可能

---

## 🧪 コマンド一覧

| コマンド                                      | 説明                                                      |
| ----------------------------------------- | ------------------------------------------------------- |
| `uv run very_simple_rag.py update-vector` | `data/` フォルダ内のファイルをベクトル化し、FAISS に保存します                  |
| `uv run very_simple_rag.py search`| 検索のみを実行します。クエリを `--query` オプション、検索結果の数を `--k` オプションで指定可能です。 |
| `uv run very_simple_rag.py run-mcp-server`| MCPサーバモードで起動します。サーバプロセスは 127.0.0.1:8000 にて待ち受けます。クエリを受け取って検索結果をクライアントに返します |

---

## 🚀 セットアップ手順

### 1. ライブラリのインストール

[**uv**](https://github.com/astral-sh/uv) を使って高速・依存関係解決付きでインストールできます。
事前に `uv` をインストールしていない場合は以下で導入できます：

```bash
pip install uv
```
LLM の動作に CUDA を利用したい場合は下記のコマンドで依存関係をインストールしてください。
```bash
uv sync --extra cuda
```
CPUを利用したい場合は下記の方法でインストールしてください。
```bash
uv sync --extra cpu
```


---

### 2. ドキュメントの配置とベクトル化～ベクターストアに格納
dataフォルダの下に配置されたドキュメント群をすべてチャンク分割して埋め込みを行います。

```bash
mkdir data
# → data に １つ以上の.pdf/.txt/.csv を配置してください
uv run very_simple_rag.py update-vector
```

完了するとFAISSのインデックスフォルダが作成されます。
大量のドキュメントについて update-vector を完了するには長い時間が必要になる場合があるため、注意してください。

---

### 4. 実行と応答確認
下記のコマンドによって "This is a test query." という自然文のクエリでセマンティック検索を行います。
2つまでの検索結果を返すように指定しています。

```bash
uv run very_simple_rag.py search --interactive --query "This is a test query." --k 2
```

検索結果が返ると、続けてクエリの入力を要求されます。何も入力せずにEnterキーを押すか、Ctrl-Cによって終了できます。

---

## ⚙ PyInstallerによる実行ファイル化

このスクリプトは PyInstaller に対応しており、　build.bat でスタンドアロン実行形式に変換できます。

---

## 📖 ライセンス

このリポジトリは MIT ライセンスです。使用するモデルのライセンスは Hugging Face にてご確認ください。

---

## 💡 補足

* 初回実行時はモデルとインデックス構築に時間がかかります。
* すべてローカルで動作するため、プライバシー性の高い利用にも適しています。
* CUDA 利用時の動作確認は Windows 11 + Intel Core i7-8750H + NVIDIA GTX 1650 + CUDA Tool kit 12.8 の環境で行いました

