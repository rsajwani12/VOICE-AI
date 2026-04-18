"""
Thymia Sentinel integration — REAL SDK shape.

Per docs at https://thymia-ai.github.io/thymia-sentinel-integrations/1.1.0/:

    pip install thymia-sentinel

    from thymia_sentinel import SentinelClient

    sentinel = SentinelClient(
        user_label="user-123",
        policies=["demo_wellbeing_awareness"],
        biomarkers=["helios", "apollo"],
        sample_rate=16000,
    )

    @sentinel.on_policy_result
    async def handle_result(result):
        # result["result"]["recommended_actions"]["for_agent"] — guidance for the LLM
        # result["result"]["alerts"] — list of {type, severity, ...}
        # result["result"]["biomarkers"] — raw biomarker readings
        ...

    @sentinel.on_progress
    def handle_progress(result):
        # result["biomarkers"]: {name: {speech_seconds, trigger_seconds, processing}}
        ...

    await sentinel.connect()
    await sentinel.send_user_audio(pcm16_bytes)      # from the human
    await sentinel.send_agent_audio(pcm16_bytes)     # from the AI agent
    await sentinel.send_user_transcript("...")
    await sentinel.send_agent_transcript("...")

Thymia Sentinel is designed for AI-voice-agent-to-human conversations. The
"user" is the client/customer; the "agent" is the AI voice agent (our Gradium
TTS output). Policies are configured with Thymia ahead of time and referenced
by name.

For our use case — a product manager talking to a client, with an AI voice
steering the meeting — we map:
  - client (customer)       → user_label / send_user_audio / send_user_transcript
  - product manager speech  → send_user_transcript with a different user_label
  - AI voice agent output   → send_agent_audio / send_agent_transcript

Sentinel treats each SentinelClient instance as one user_label. To monitor
both the client AND the product manager as separate "users", we spin up two
client instances with different user_label values.
"""
import asyncio
import time
from typing import Callable, Optional

from . import config
from .models import Speaker, BiomarkerUpdate


class ThymiaPersonMonitor:
    """One Sentinel client per human we're monitoring (client and product manager)."""

    def __init__(
        self,
        person_name: str,
        speaker_kind: Speaker,
        on_biomarker: Callable[[BiomarkerUpdate], None],
        on_policy_action: Callable[[dict, Speaker, str], None],
    ):
        self.person_name = person_name
        self.speaker_kind = speaker_kind
        self.on_biomarker = on_biomarker
        self.on_policy_action = on_policy_action

        self.mock = config.USE_MOCK or not config.THYMIA_API_KEY
        self._sdk_client = None
        self._connected = False

    async def connect(self):
        """Establish Sentinel connection for this person."""
        if self.mock:
            self._connected = True
            # Start mock progress loop
            asyncio.create_task(self._mock_progress_loop())
            return

        try:
            from thymia_sentinel import SentinelClient
        except ImportError as e:
            raise RuntimeError("thymia-sentinel not installed. Run: pip install thymia-sentinel") from e

        self._sdk_client = SentinelClient(
            user_label=self.person_name,
            policies=config.THYMIA_POLICIES,
            biomarkers=config.THYMIA_BIOMARKERS,
            sample_rate=config.THYMIA_SAMPLE_RATE,
            progress_updates_frequency=2.0,
        )

        # Register handlers
        @self._sdk_client.on_policy_result
        async def _on_policy(result):
            self._handle_policy_result(result)

        @self._sdk_client.on_progress
        def _on_progress(result):
            self._handle_progress(result)

        await self._sdk_client.connect()
        self._connected = True

    async def send_audio(self, pcm16_bytes: bytes):
        """Stream user audio (the human speaking). PCM16 @ 16kHz."""
        if self.mock or not self._connected:
            return
        await self._sdk_client.send_user_audio(pcm16_bytes)

    async def send_transcript(self, text: str):
        """Send finalised transcript for this person."""
        if self.mock or not self._connected:
            return
        await self._sdk_client.send_user_transcript(text)

    async def close(self):
        if self._sdk_client:
            await self._sdk_client.close()
        self._connected = False

    # =========================================================
    # PAYLOAD HANDLERS
    # =========================================================

    def _handle_policy_result(self, result: dict):
        """Thymia's on_policy_result payload — actions + biomarker summary."""
        inner = result.get("result", {})

        # Raw biomarker snapshot
        biomarkers = inner.get("biomarkers") or inner.get("biomarker_summary") or {}
        if biomarkers:
            update = self._biomarkers_to_update(biomarkers)
            self.on_biomarker(update)

        # Recommended agent-side actions (Thymia's guidance for the AI)
        actions = inner.get("recommended_actions", {})
        if actions.get("for_agent"):
            self.on_policy_action(
                {"for_agent": actions["for_agent"], "urgency": actions.get("urgency", "normal"), "alerts": inner.get("alerts", [])},
                self.speaker_kind,
                self.person_name,
            )

    def _handle_progress(self, result: dict):
        """Thymia's on_progress payload — biomarker collection status."""
        progress = {}
        for name, status in result.get("biomarkers", {}).items():
            collected = status.get("speech_seconds", 0)
            required = max(status.get("trigger_seconds", 1), 0.001)
            progress[name] = min(100, (collected / required) * 100)

        update = BiomarkerUpdate(
            speaker=self.speaker_kind,
            speaker_name=self.person_name,
            ts=result.get("timestamp", time.time()),
            progress=progress,
            raw={"progress": result},
        )
        self.on_biomarker(update)

    def _biomarkers_to_update(self, biomarkers: dict) -> BiomarkerUpdate:
        """
        Normalise Thymia's biomarker output into our UI shape.
        Thymia returns nested biomarkers under model names (helios, apollo, etc);
        we extract the summary fields that make sense in the dashboard.
        """
        def val(key: str) -> Optional[float]:
            # Flat lookup first
            if key in biomarkers and isinstance(biomarkers[key], (int, float)):
                return float(biomarkers[key])
            # Nested lookup (model_name.value)
            for model_name, model_bm in biomarkers.items():
                if isinstance(model_bm, dict) and key in model_bm:
                    v = model_bm[key]
                    if isinstance(v, (int, float)):
                        return float(v)
                    if isinstance(v, dict) and "value" in v:
                        return float(v["value"])
            return None

        return BiomarkerUpdate(
            speaker=self.speaker_kind,
            speaker_name=self.person_name,
            ts=time.time(),
            cognitive_load=val("cognitive_load"),
            stress=val("stress"),
            wellness=val("wellness") or val("wellbeing"),
            engagement=val("engagement"),
            confidence=val("confidence"),
            raw=biomarkers,
        )

    # =========================================================
    # MOCK DATA (for pre-event dev)
    # =========================================================

    async def _mock_progress_loop(self):
        """Emit synthetic biomarker/progress updates every 3 seconds."""
        import random
        t0 = time.time()
        while self._connected:
            await asyncio.sleep(3)
            elapsed = time.time() - t0
            progress_pct = min(100, elapsed * 5)  # hit 100% in 20s

            update = BiomarkerUpdate(
                speaker=self.speaker_kind,
                speaker_name=self.person_name,
                ts=time.time(),
                cognitive_load=0.4 + random.random() * 0.4 if progress_pct > 50 else None,
                stress=0.3 + random.random() * 0.3 if progress_pct > 50 else None,
                wellness=0.6 + random.random() * 0.2 if progress_pct > 50 else None,
                engagement=0.5 + random.random() * 0.3 if progress_pct > 50 else None,
                confidence=0.55 + random.random() * 0.3 if progress_pct > 50 else None,
                progress={"helios": progress_pct, "apollo": min(100, progress_pct * 0.8)},
                raw={"mock": True},
            )
            self.on_biomarker(update)

            # Occasional mock policy action
            if progress_pct > 60 and random.random() < 0.2:
                self.on_policy_action(
                    {
                        "for_agent": (
                            f"[MOCK] {self.person_name} is showing signs of hesitation. "
                            "Slow the pace and offer a clarifying question."
                        ),
                        "urgency": "normal",
                        "alerts": [],
                    },
                    self.speaker_kind,
                    self.person_name,
                )
