# hooks/hook-llama_cpp.py

from PyInstaller.utils.hooks import collect_data_files
import os

# llama_cpp パッケージのパスを取得
pkg_base, pkg_dir = collect_data_files('llama_cpp', subdir='lib')[0]
# collect_data_files で 'llama_cpp/lib' 以下のファイルを全て回収
datas = collect_data_files('llama_cpp', subdir='lib')
