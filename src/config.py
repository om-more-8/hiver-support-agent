"""
Central config. Change BRAND to whichever brand you pick after EDA.
Everything downstream reads from here so you only change it in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# ---- Paths ----
RAW_CSV = ROOT / "data" / "raw" / "twcs.csv"          # the Kaggle file, once you download it
PROCESSED_DIR = ROOT / "data" / "processed"
GOLDEN_SET_PATH = ROOT / "data" / "golden_set.jsonl"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"

# ---- Brand selection ----
# Kaggle dataset uses the Twitter handle as author_id for brand-side tweets, e.g. "AmazonHelp", "AppleSupport", "Uber_Support"
BRAND = os.getenv("BRAND", "AmazonHelp")

# ---- LLM ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ---- Retrieval ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3

# ---- Golden set ----
GOLDEN_SET_SIZE = 200
RANDOM_SEED = 42

for d in [PROCESSED_DIR, RESULTS_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
