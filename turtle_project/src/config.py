import os
from pathlib import Path
from dotenv import load_dotenv


from typing import Optional

load_dotenv()  # looks for a .env file in the current and parent directories
print(".env loaded (if present)")

def get_key(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)