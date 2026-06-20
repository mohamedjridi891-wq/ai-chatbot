import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / '.env')
except ImportError:
    pass

HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')

if HF_TOKEN:
    os.environ['HUGGINGFACE_HUB_TOKEN'] = HF_TOKEN


def get_hf_token():
    """Return the Hugging Face token from environment variables."""
    return HF_TOKEN


def use_hf_token():
    """Ensure the Hugging Face token is available for huggingface-hub and sentence-transformers."""
    if not HF_TOKEN:
        raise RuntimeError('Hugging Face token not found. Set HF_TOKEN in .env or environment.')
    os.environ['HUGGINGFACE_HUB_TOKEN'] = HF_TOKEN
    return HF_TOKEN
