"""Environment-driven configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

# Gradium
GRADIUM_API_KEY = os.getenv("GRADIUM_API_KEY", "")
GRADIUM_CUSTOM_VOICE_ID = os.getenv("GRADIUM_CUSTOM_VOICE_ID", "")
GRADIUM_FALLBACK_VOICE_ID = os.getenv("GRADIUM_FALLBACK_VOICE_ID", "YTpq7expH9539ERJ")

# Thymia Sentinel
THYMIA_API_KEY = os.getenv("THYMIA_API_KEY", "")
THYMIA_POLICIES = [p.strip() for p in os.getenv("THYMIA_POLICIES", "demo_wellbeing_awareness").split(",") if p.strip()]
THYMIA_BIOMARKERS = [b.strip() for b in os.getenv("THYMIA_BIOMARKERS", "helios,apollo").split(",") if b.strip()]

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-5"

# LiveKit
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

# Behaviour
USE_MOCK = os.getenv("USE_MOCK", "1") == "1"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Paths
STATIC_DIR = ROOT / "static"
SAMPLES_DIR = ROOT / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

# Timing
# Gradium STT expects 24kHz PCM16 mono, 80ms (1920 sample) chunks
GRADIUM_STT_SAMPLE_RATE = 24000
GRADIUM_STT_CHUNK_SAMPLES = 1920

# Thymia Sentinel expects PCM16 @ 16kHz mono
THYMIA_SAMPLE_RATE = 16000

# Claude intelligence cadence
CLAUDE_INTELLIGENCE_INTERVAL = 20  # seconds
CLAUDE_TRANSCRIPT_WINDOW = 40       # lines sent to Claude each analysis


def integration_status() -> dict:
    return {
        "gradium": bool(GRADIUM_API_KEY),
        "thymia": bool(THYMIA_API_KEY),
        "anthropic": bool(ANTHROPIC_API_KEY),
        "livekit": bool(LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
    }
