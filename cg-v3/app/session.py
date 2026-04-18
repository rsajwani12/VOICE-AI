"""
Session orchestrator — the nervous system of a live meeting.

Coordinates:
  - Per-person Thymia Sentinel monitors (PM + each client stakeholder)
  - Gradium STT (audio in) and TTS (voice agent out)
  - Claude intelligence (dashboard alerts + voice agent decisions)
  - Dashboard WebSocket broadcast
"""
import asyncio
import base64
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import WebSocket

from . import config
from .models import (
    Speaker, TranscriptLine, BiomarkerUpdate, MeetingContext, Alert, AgentUtterance,
)
from .gradium_client import GradiumClient
from .thymia_client import ThymiaPersonMonitor
from .claude_brain import ClaudeBrain
from .documentation import generate_meeting_report


class MeetingSession:
    def __init__(self, context: MeetingContext):
        self.context = context
        self.start_time = time.time()
        self.session_id = str(uuid.uuid4())[:8]

        # Clients
        self.gradium = GradiumClient()
        self.claude = ClaudeBrain()
        self.thymia_monitors: dict[str, ThymiaPersonMonitor] = {}

        # State
        self.transcript: list[TranscriptLine] = []
        self.biomarkers: list[BiomarkerUpdate] = []
        self.alerts: list[Alert] = []
        self.agent_utterances: list[AgentUtterance] = []
        self.ws_clients: set[WebSocket] = set()

        # Loops
        self._intelligence_task: Optional[asyncio.Task] = None
        self._agent_task: Optional[asyncio.Task] = None
        self._stopped = False

        # Audio directory for agent TTS files
        self.audio_dir = config.ROOT / "static" / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # LIFECYCLE
    # ================================================================
    async def start(self):
        # Spin up one Thymia monitor per stakeholder
        for s in self.context.stakeholders:
            monitor = ThymiaPersonMonitor(
                person_name=s.name,
                speaker_kind=Speaker.CLIENT if s.is_client else Speaker.CONSULTANT,
                on_biomarker=self._handle_biomarker,
                on_policy_action=self._handle_thymia_action,
            )
            self.thymia_monitors[s.name] = monitor
            await monitor.connect()

        self._intelligence_task = asyncio.create_task(self._intelligence_loop())
        self._agent_task = asyncio.create_task(self._agent_loop())

    async def stop(self):
        self._stopped = True
        for t in (self._intelligence_task, self._agent_task):
            if t:
                t.cancel()
        for m in self.thymia_monitors.values():
            await m.close()
        await self.gradium.close()

    # ================================================================
    # WEBSOCKET
    # ================================================================
    def add_ws(self, ws: WebSocket):
        self.ws_clients.add(ws)

    def remove_ws(self, ws: WebSocket):
        self.ws_clients.discard(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.discard(ws)

    def _elapsed(self) -> float:
        return time.time() - self.start_time

    # ================================================================
    # TRANSCRIPT INGESTION
    # ================================================================
    async def ingest_transcript(
        self, text: str, speaker: Speaker, speaker_name: str,
        ts_start: float = 0, ts_end: float = 0,
    ):
        line = TranscriptLine(
            text=text,
            speaker=speaker,
            speaker_name=speaker_name,
            ts_start=ts_start or self._elapsed(),
            ts_end=ts_end or self._elapsed(),
        )
        self.transcript.append(line)

        # Forward to the relevant Thymia monitor as a transcript
        monitor = self.thymia_monitors.get(speaker_name)
        if monitor:
            await monitor.send_transcript(text)

        await self.broadcast({
            "type": "transcript",
            "line": line.to_dict(),
            "elapsed": self._elapsed(),
        })

    # ================================================================
    # AUDIO INGESTION
    # ================================================================
    async def ingest_audio_chunk(self, speaker_name: str, pcm16_bytes: bytes):
        """Raw PCM16 @ 16kHz from the browser or LiveKit."""
        monitor = self.thymia_monitors.get(speaker_name)
        if monitor:
            await monitor.send_audio(pcm16_bytes)

    # ================================================================
    # BIOMARKER + THYMIA ACTION HANDLERS
    # ================================================================
    def _handle_biomarker(self, update: BiomarkerUpdate):
        self.biomarkers.append(update)
        # Schedule the broadcast on the event loop
        asyncio.create_task(self.broadcast({
            "type": "biomarker",
            "update": update.to_dict(),
            "elapsed": self._elapsed(),
        }))

    def _handle_thymia_action(self, action: dict, speaker: Speaker, speaker_name: str):
        # Feed this into Claude's context for the next analysis cycle
        self.claude.absorb_thymia_action(action, speaker, speaker_name)

        # Also surface to the dashboard as a Thymia-sourced alert
        for_agent = action.get("for_agent", "")
        if for_agent:
            alert = Alert(
                kind="thymia-action",
                type="coaching",
                type_label=f"Thymia · {speaker_name}",
                title=f"Policy guidance re: {speaker_name}",
                detail=for_agent,
                suggestion="",
                suggestion_label="",
                signals=[{"label": action.get("urgency", "normal"), "class": "warm"}],
                source="thymia",
            )
            self.alerts.append(alert)
            asyncio.create_task(self.broadcast({
                "type": "alert",
                "alert": alert.to_dict(),
                "elapsed": self._elapsed(),
            }))

    # ================================================================
    # CLAUDE INTELLIGENCE LOOP
    # ================================================================
    async def _intelligence_loop(self):
        while not self._stopped:
            await asyncio.sleep(config.CLAUDE_INTELLIGENCE_INTERVAL)
            if self._stopped or not self.transcript:
                continue
            try:
                alerts = await self.claude.generate_dashboard_alerts(
                    context=self.context,
                    transcript=self.transcript,
                    biomarkers=self.biomarkers,
                )
                for a in alerts:
                    self.alerts.append(a)
                    await self.broadcast({
                        "type": "alert",
                        "alert": a.to_dict(),
                        "elapsed": self._elapsed(),
                    })
            except Exception as e:
                print(f"[intelligence_loop] error: {e}")

    async def force_intelligence(self):
        alerts = await self.claude.generate_dashboard_alerts(
            context=self.context,
            transcript=self.transcript,
            biomarkers=self.biomarkers,
        )
        for a in alerts:
            self.alerts.append(a)
            await self.broadcast({
                "type": "alert",
                "alert": a.to_dict(),
                "elapsed": self._elapsed(),
            })

    # ================================================================
    # VOICE AGENT LOOP
    # ================================================================
    async def _agent_loop(self):
        """Every 30s, check if the voice agent should interject."""
        while not self._stopped:
            await asyncio.sleep(30)
            if self._stopped or len(self.transcript) < 3:
                continue
            try:
                utt = await self.claude.generate_agent_utterance(
                    context=self.context,
                    transcript=self.transcript,
                    biomarkers=self.biomarkers,
                )
                if utt:
                    await self._emit_agent_utterance(utt)
            except Exception as e:
                print(f"[agent_loop] error: {e}")

    async def force_agent_speak(self) -> Optional[AgentUtterance]:
        utt = await self.claude.generate_agent_utterance(
            context=self.context,
            transcript=self.transcript,
            biomarkers=self.biomarkers,
        )
        if utt:
            await self._emit_agent_utterance(utt)
        return utt

    async def _emit_agent_utterance(self, utt: AgentUtterance):
        """Synthesize the utterance via Gradium TTS and broadcast to dashboard."""
        try:
            audio_bytes = await self.gradium.synthesize(utt.text, output_format="wav")
            filename = f"agent-{self.session_id}-{int(time.time()*1000)}.wav"
            path = self.audio_dir / filename
            path.write_bytes(audio_bytes)
            utt.audio_url = f"/static/audio/{filename}"
            utt.voice_id = self.gradium.voice_id
        except Exception as e:
            print(f"[agent TTS] error: {e}")

        self.agent_utterances.append(utt)
        await self.broadcast({
            "type": "agent_utterance",
            "utterance": utt.to_dict(),
            "elapsed": self._elapsed(),
        })

    # ================================================================
    # REPORT GENERATION
    # ================================================================
    def generate_report(self) -> str:
        return generate_meeting_report(
            context=self.context,
            transcript=self.transcript,
            alerts=self.alerts,
            biomarkers=self.biomarkers,
        )
