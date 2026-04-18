# Common Ground

**Voice AI that reads both sides of high-stakes professional conversations.**

A product manager is reviewing work with a client. The client says "yes, that looks good" — but their cognitive load is elevated and they hesitated 3 seconds before answering. Three weeks later the work gets rejected. Common Ground catches that moment in real time, privately alerts the PM, and the voice agent can steer the conversation to surface the real concern.

---

## Architecture (real APIs)

```
┌──────────────────────────────────────────────────────────────┐
│                      Meeting audio                            │
│  Browser mic  ──▶  Web Speech API (dev) or                   │
│                    TinyFish + Gradium STT (prod)              │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
          ┌──────────────────┴──────────────────┐
          │         FastAPI backend             │
          │         (app/main.py)               │
          ├──────────────────┬──────────────────┤
          │                  │                  │
          ▼                  ▼                  ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Thymia Sentinel  │ │ Claude (brain)   │ │  Gradium TTS     │
│ pip install      │ │ anthropic        │ │  pip install     │
│ thymia-sentinel  │ │ (claude-sonnet)  │ │  gradium         │
│                  │ │                  │ │                  │
│ Per-person       │ │ · Dashboard      │ │  Voice agent     │
│ biomarker        │ │   alerts         │ │  speech          │
│ monitors         │ │ · Agent speech   │ │  synthesis       │
│ (client + PM)    │ │   decisions      │ │                  │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │ Dashboard (HTML) │
                   │ WebSocket feed   │
                   └──────────────────┘
```

---

## Run it

**Requirements:** Python 3.10+ (thymia-sentinel requires 3.10, Gradium requires 3.10+).

```bash
# 1. Install
cd cg-v3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Install the real SDKs at the event when you have API keys:
#    pip install gradium thymia-sentinel anthropic
#    (commented out in requirements.txt to avoid blocking the mock-mode setup)

# 3. Configure
cp .env.example .env
# Edit .env — at minimum set ANTHROPIC_API_KEY to get live Claude intelligence

# 4. Run
uvicorn app.main:app --reload

# 5. Open
# http://localhost:8000
```

With `USE_MOCK=1` (default), the backend runs with synthetic Thymia biomarkers and mock Claude alerts so you can smoke-test the full UI. Once real API keys are in `.env`, set `USE_MOCK=0` and restart.

---

## The real SDK shapes

### Gradium

Per [docs.gradium.ai](https://docs.gradium.ai/):

```python
import gradium

client = gradium.client.GradiumClient(api_key="gd_...")

# TTS (voice agent speech)
result = await client.tts(
    setup={"model_name": "default", "voice_id": "YTpq7expH9539ERJ", "output_format": "wav"},
    text="Before we move on — what concerns would we want to raise with compliance?"
)
with open("agent.wav", "wb") as f:
    f.write(result.raw_data)

# STT (streaming transcription)
# Audio spec: PCM16 @ 24kHz mono, 1920-sample (80ms) chunks
stream = await client.stt_stream(
    {"model_name": "default", "input_format": "pcm"},
    audio_generator
)
async for msg in stream._stream:
    if msg["type"] == "text":
        print(msg["text"])
```

**What's implemented:** `GradiumClient` in `app/gradium_client.py` wraps both.

**What you'll do at the event:** clone Patti's voice via the Gradium Studio (upload a 10s WAV sample), copy the returned `voice_id` into `.env` as `GRADIUM_CUSTOM_VOICE_ID`. The voice agent will speak in Patti's voice.

### Thymia Sentinel

Per [thymia-ai.github.io/thymia-sentinel-integrations](https://thymia-ai.github.io/thymia-sentinel-integrations/1.1.0/):

```python
from thymia_sentinel import SentinelClient

sentinel = SentinelClient(
    user_label="maria-chen",
    policies=["demo_wellbeing_awareness"],  # Pre-configured by Thymia
    biomarkers=["helios", "apollo"],
    sample_rate=16000,
)

@sentinel.on_policy_result
async def on_result(result):
    actions = result["result"].get("recommended_actions", {})
    if actions.get("for_agent"):
        # This is the guidance for our Claude brain
        ...

@sentinel.on_progress
def on_progress(result):
    # Collection status per biomarker
    ...

await sentinel.connect()
await sentinel.send_user_audio(pcm16_bytes)       # Human speech
await sentinel.send_user_transcript("text...")    # Finalised transcript
```

**What's implemented:** `ThymiaPersonMonitor` in `app/thymia_client.py`. We spin up one monitor per stakeholder (the client AND the product manager), so biomarkers flow separately for each person.

**What you'll do at the event:** email support@thymia.ai for an API key. Ask them to set up a policy suitable for "business meeting facilitation" use case — the default `demo_wellbeing_awareness` works for the demo but custom policies are more on-brand.

### Claude

Standard Anthropic SDK. Model: `claude-sonnet-4-5`. Two entry points:

- `generate_dashboard_alerts` — every 20s, produces up to 3 actionable items for the PM dashboard
- `generate_agent_utterance` — every 30s, decides if the voice agent should interject (returns None most of the time — this is deliberate)

Both prompts integrate Thymia's `for_agent` guidance from policy results.

### TinyFish (optional for the hack)

The existing code uses browser mic + Web Speech API as a dev fallback. At the event, if you need real multi-participant meeting support, wire the TinyFish agent example from Thymia's docs:

- https://thymia-ai.github.io/thymia-sentinel-integrations/1.1.0/integrations/livekit/

The Thymia team provides a `livekit-plugins-thymia` plugin that auto-captures room audio and forwards to Sentinel. That's the production path.

---

## Files

```
cg-v3/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI routes + WebSocket
│   ├── config.py              # env-driven config
│   ├── models.py              # typed dataclasses
│   ├── gradium_client.py      # Gradium TTS + STT wrapper
│   ├── thymia_client.py       # Per-person Sentinel monitors
│   ├── claude_brain.py        # Dashboard alerts + agent speech
│   ├── session.py             # Session orchestrator
│   └── documentation.py       # End-of-meeting HTML report
├── static/
│   ├── index.html             # Dashboard SPA
│   └── audio/                 # Agent TTS files served here
├── reports/                   # End-of-meeting reports
├── samples/                   # Patti's 10s voice sample for cloning
├── requirements.txt
├── .env.example
└── README.md
```

---

## Integration checklist for the event

Things your backend engineer does in the first 2 hours on site:

### 1. Get keys (first 30 min)
- [ ] Gradium API key from [app.gradium.ai](https://app.gradium.ai)
- [ ] Thymia API key — email support@thymia.ai or find them at the event
- [ ] Anthropic API key (if not using Patti's existing)
- [ ] LiveKit cloud account (optional, only for production multi-participant)

### 2. Clone Patti's voice (10 min)
- [ ] Record 10 seconds of Patti speaking neutrally (any audio editor, export as WAV)
- [ ] Upload through Gradium Studio → get `voice_id`
- [ ] Paste into `.env` as `GRADIUM_CUSTOM_VOICE_ID`

### 3. Set up Thymia policies (30 min)
- [ ] Ask thymia team to enable `demo_wellbeing_awareness` for your API key
- [ ] Optionally request a custom policy for "client meeting facilitation"
- [ ] Update `THYMIA_POLICIES` in `.env` if a custom policy is configured

### 4. Install real SDKs (5 min)
```bash
pip install gradium thymia-sentinel
```

### 5. Flip the switch
- [ ] Set `USE_MOCK=0` in `.env`
- [ ] Restart uvicorn
- [ ] Check the Setup page — all four integration tiles should go green

### 6. Smoke test
- [ ] Start a session with 2-3 stakeholders
- [ ] Talk for 60 seconds holding Alt while "the client" is speaking
- [ ] Watch biomarkers populate
- [ ] Watch Claude alerts surface on the right
- [ ] Press "Ask agent" — voice agent should decide whether to speak

---

## Demo flow (3 minutes)

1. **Cold open (30s)** — tell the lived consulting story. "Four hours on a deck, shouted at at 6pm, five more hours to fix. Nobody listened to what the email actually said."

2. **Setup reveal (30s)** — show the Setup view. Stakeholder cards with behavioural notes ("Maria's silence is processing, not resistance"). Regulatory frames (Consumer Duty, accessibility, vulnerable customers). All four integrations green.

3. **Live demo (90s)** — start recording, 60-second mock workshop with a teammate playing Maria. Trigger the polite-yes moment. Watch:
   - **Biomarkers populate** on Maria's side (Thymia)
   - **Dashboard alerts fire** (Claude interpreting biomarkers + transcript + persona notes)
   - **Voice agent interjects** (Gradium TTS speaking a thoughtful probe in Patti's cloned voice)

4. **Ethics beat (30s)** — "Data serves the professionals in the room, not management. Voice agent never commits on the firm's behalf. Biomarkers inform, they don't decide. We designed deliberately against the surveillance model identified in Cardon et al's 2023 research."

---

## Known scoping

**In scope and working:**
- Browser mic capture + transcription (Web Speech API fallback)
- Per-person Thymia Sentinel biomarker monitoring
- Live Claude intelligence alerts on the dashboard
- Voice agent that can speak in Patti's cloned voice via Gradium TTS
- Per-session report generation at end
- Behavioural-observation stakeholder profiles (not MBTI — MBTI accepted as weak input only)

**Out of scope for the hack:**
- Multi-participant LiveKit rooms (doable but bigger infrastructure lift)
- Actual deployment / user accounts / persistence
- Voice cloning automation (done manually through Gradium Studio)
- TinyFish web agent for documentation (using local HTML renderer instead — real TinyFish integration is a natural next step)

**Honest about the voice agent:**
The agent is designed to be **rare** — it should not constantly interject. Its job is to steer when the conversation drifts, probe when signal is being missed, and calm when tension rises. Claude is prompted conservatively to return `{"speak": false}` most of the time. If you want it to speak on demand, use the "Ask agent" button.

---

## Org psych grounding

Cite these in the pitch — they give intellectual standing:

- **Common ground theory** (Herbert Clark) — communication requires shared mental models
- **Sensemaking** (Karl Weick) — organisations fail when interpretations diverge without reconciliation
- **Psychological safety** (Amy Edmondson) — AI meeting tools mostly reduce safety because they serve management; we invert this
- **Collective intelligence** (Anita Woolley) — team IQ correlates with equal turn-taking
- **AI meeting tensions** (Cardon et al 2023) — we designed against the five tensions identified

---

## Ethics commitments (say these in the pitch)

- Data goes to the professionals in the conversation, not to management dashboards
- Voice cloning is used only in the cloned person's own voice, speaking as a facilitator. It never commits on behalf of the firm, never represents someone who isn't present
- Thymia biomarkers are used as conversation signal proxies, not as clinical diagnoses
- Client-side biomarker data is aggregated at the session level, not attributed to named individuals in any cross-session context
- Consent-first: the client-facing recording of a workshop follows standard workshop recording practices

---

*What we hear and what we're told are not always the same thing.*
