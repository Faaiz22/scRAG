"""
scRAG/src/utils.py
Small helpers.
"""
import os
from dotenv import load_dotenv
load_dotenv()

def get_env(key, default=None):
    return os.environ.get(key, default)
