"""
FastAPI application — Common Ground backend.

Routes
------
GET  /                            serves the dashboard
GET  /api/health                  integration status
POST /api/session                 create a new session
POST /api/session/transcript      push a finalised transcript line
POST /api/session/audio           push raw PCM16 audio chunk for a speaker
POST /api/session/analyse         force an immediate Claude analysis
POST /api/session/agent-speak     ask the voice agent to consider speaking now
POST /api/session/end             end the active session and generate a report
WS   /ws                          dashboard subscribes for live events
"""
import base64
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .models import Speaker, MeetingContext, Stakeholder
from .session import MeetingSession


app = FastAPI(title="Common Ground Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

SESSION: Optional[MeetingSession] = None


# ================================================================
# PYDANTIC SCHEMAS
# ================================================================
class StakeholderIn(BaseModel):
    name: str
    role: str = ""
    notes: str = ""
    mbti: str = ""
    is_client: bool = True


class SessionCreate(BaseModel):
    org_name: str
    objective: str
    session_type: str
    stage: str
    stakeholders: list[StakeholderIn] = []
    regulatory_frames: list[str] = []


class TranscriptIn(BaseModel):
    text: str
    speaker: str      # 'consultant' or 'client'
    speaker_name: str
    ts_start: float = 0
    ts_end: float = 0


class AudioIn(BaseModel):
    speaker_name: str
    audio_base64: str


# ================================================================
# ROUTES
# ================================================================
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "use_mock": config.USE_MOCK,
        "integrations": config.integration_status(),
        "session_active": SESSION is not None,
        "thymia_policies": config.THYMIA_POLICIES,
        "thymia_biomarkers": config.THYMIA_BIOMARKERS,
    }


@app.post("/api/session")
async def create_session(payload: SessionCreate):
    global SESSION
    if SESSION is not None:
        await SESSION.stop()

    ctx = MeetingContext(
        org_name=payload.org_name,
        objective=payload.objective,
        session_type=payload.session_type,
        stage=payload.stage,
        stakeholders=[Stakeholder(**s.model_dump()) for s in payload.stakeholders],
        regulatory_frames=payload.regulatory_frames,
    )
    SESSION = MeetingSession(ctx)
    await SESSION.start()
    return {"status": "started", "session_id": SESSION.session_id, "context": ctx.to_dict()}


@app.post("/api/session/transcript")
async def push_transcript(line: TranscriptIn):
    if SESSION is None:
        raise HTTPException(400, "No active session")
    await SESSION.ingest_transcript(
        text=line.text,
        speaker=Speaker(line.speaker),
        speaker_name=line.speaker_name,
        ts_start=line.ts_start,
        ts_end=line.ts_end,
    )
    return {"ok": True}


@app.post("/api/session/audio")
async def push_audio(payload: AudioIn):
    if SESSION is None:
        raise HTTPException(400, "No active session")
    try:
        pcm = base64.b64decode(payload.audio_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 audio")
    await SESSION.ingest_audio_chunk(payload.speaker_name, pcm)
    return {"ok": True, "bytes": len(pcm)}


@app.post("/api/session/analyse")
async def force_analyse():
    if SESSION is None:
        raise HTTPException(400, "No active session")
    await SESSION.force_intelligence()
    return {"ok": True}


class AgentSpeakIn(BaseModel):
    purpose: str | None = None  # 'probe' | 'steer' | 'calm' | 'thought-provoker' | 'summary'


@app.post("/api/session/agent-speak")
async def force_agent_speak(payload: AgentSpeakIn | None = None):
    if SESSION is None:
        raise HTTPException(400, "No active session")
    purpose = payload.purpose if payload else None
    utt = await SESSION.force_agent_speak(purpose=purpose)
    if utt:
        return {"ok": True, "utterance": utt.to_dict()}
    return {"ok": True, "utterance": None}


@app.post("/api/session/end")
async def end_session():
    global SESSION
    if SESSION is None:
        return {"ok": True, "already_ended": True}
    report_path = SESSION.generate_report()
    await SESSION.stop()
    SESSION = None
    # Return a relative URL to the report (served from /reports)
    from pathlib import Path
    report_filename = Path(report_path).name
    return {
        "ok": True,
        "report_url": f"/reports/{report_filename}",
    }


# ================================================================
# WEBSOCKET
# ================================================================
@app.websocket("/ws")
async def dashboard_ws(ws: WebSocket):
    await ws.accept()
    if SESSION is None:
        await ws.send_json({"type": "error", "message": "No active session. POST /api/session first."})
        await ws.close()
        return

    SESSION.add_ws(ws)
    try:
        await ws.send_json({
            "type": "snapshot",
            "context": SESSION.context.to_dict(),
            "transcript": [t.to_dict() for t in SESSION.transcript[-50:]],
            "biomarkers": [b.to_dict() for b in SESSION.biomarkers[-15:]],
            "alerts": [a.to_dict() for a in SESSION.alerts[-15:]],
            "agent_utterances": [u.to_dict() for u in SESSION.agent_utterances[-5:]],
            "elapsed": SESSION._elapsed(),
        })
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        SESSION.remove_ws(ws)


# ================================================================
# STATIC FILES
# ================================================================
@app.get("/")
async def root():
    index = config.STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "Dashboard not built yet.")
    return FileResponse(index)


# Mount static files (dashboard assets + agent TTS audio)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# Mount reports directory
REPORTS_DIR = config.ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=True)
