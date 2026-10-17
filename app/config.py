"""
config.py -- loads the API key and shared paths. Every other module
imports from here instead of reading os.environ directly, so there's one
place to fix if the env var name or model id ever changes.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # fine if python-dotenv isn't installed -- just relies on a real env var instead

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Copy .env.example to .env in the repo "
        "root and paste in your key from console.anthropic.com."
    )

# claude-sonnet-5-5 is the current balanced model as of Oct 2026 (speed + a
# vision-capable multimodal input) -- check platform.claude.com/docs if this
# assignment runs again later and model names have moved on.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Set this to your robot's IP address (shown in the Reachy Mini Control app).
ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.8.105")
