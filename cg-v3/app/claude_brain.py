"""
Claude intelligence layer.

Two distinct outputs:

1. DASHBOARD ALERTS — for the product manager's private screen.
   Claude reviews transcript + biomarkers + Thymia policy actions and surfaces
   up to 3 actionable pieces of intelligence every ~20 seconds. These include
   persona-adapted reads, regulatory blind spots, polite-yes detection, etc.

2. AGENT SPEECH — what the Gradium voice agent should say.
   Claude decides when the agent should interject with a question, probe,
   or calming remark. It also integrates guidance from Thymia's
   `recommended_actions.for_agent` payload to modulate its behaviour.
"""
import json
import time
from typing import Optional

from . import config
from .models import (
    Alert, TranscriptLine, BiomarkerUpdate, MeetingContext, AgentUtterance, Speaker,
)


REGULATORY_PROMPTS = {
    "consumer-duty": "FCA Consumer Duty — four outcomes: products/services, price/value, consumer understanding, consumer support. Flag gaps.",
    "accessibility": "Accessibility (WCAG 2.2 AA, Equality Act 2010) — probe screen reader support, cognitive accessibility, reasonable adjustments for disabled customers.",
    "vulnerable-customers": "FCA FG21/1 Vulnerable Customers — identification of vulnerability drivers (health, life events, resilience, capability), staff training, outcomes monitoring.",
    "data-protection": "UK GDPR — lawful basis, DPIA status, data minimisation, consumer rights handling.",
    "fair-value": "Fair Value Assessment — price vs. benefit analysis, target market definition, distribution strategy.",
    "complaints-handling": "Complaints handling (DISP sourcebook) — root cause analysis, trend identification, MI reporting.",
}


class ClaudeBrain:
    """
    Wraps Anthropic API for both dashboard intelligence and voice agent speech.
    """

    def __init__(self):
        self.api_key = config.ANTHROPIC_API_KEY
        self.mock = config.USE_MOCK or not self.api_key
        self._client = None
        self._prior_alert_sigs: list[str] = []
        self._thymia_hints: list[dict] = []  # recent Thymia policy actions

    def _sdk(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic") from e
        self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    # ====================================================================
    # THYMIA HINT INJECTION
    # ====================================================================
    def absorb_thymia_action(self, action: dict, speaker: Speaker, speaker_name: str):
        """
        Thymia Sentinel returned a `recommended_actions.for_agent` string.
        Queue it to influence the next dashboard analysis + agent speech.
        """
        self._thymia_hints.append({
            "ts": time.time(),
            "speaker": speaker.value,
            "speaker_name": speaker_name,
            "for_agent": action.get("for_agent", ""),
            "urgency": action.get("urgency", "normal"),
            "alerts": action.get("alerts", []),
        })
        # Keep last 5
        self._thymia_hints = self._thymia_hints[-5:]

    # ====================================================================
    # DASHBOARD INTELLIGENCE — generates Alert list
    # ====================================================================
    async def generate_dashboard_alerts(
        self,
        context: MeetingContext,
        transcript: list[TranscriptLine],
        biomarkers: list[BiomarkerUpdate],
    ) -> list[Alert]:
        """Produce up to 3 actionable intelligence items for the PM dashboard."""
        if not transcript:
            return []
        if self.mock:
            return self._mock_alerts()

        prompt = self._build_dashboard_prompt(context, transcript, biomarkers)
        try:
            client = self._sdk()
            msg = await client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            print(f"[ClaudeBrain] dashboard call failed: {e}")
            return []

        text = msg.content[0].text if msg.content else ""
        return self._parse_alerts(text)

    def _build_dashboard_prompt(
        self,
        context: MeetingContext,
        transcript: list[TranscriptLine],
        biomarkers: list[BiomarkerUpdate],
    ) -> str:
        stakeholder_block = "\n".join(self._fmt_stakeholder(s) for s in context.stakeholders)
        reg_block = "\n".join(
            f"- {REGULATORY_PROMPTS.get(f, f)}" for f in context.regulatory_frames
        ) or "None active."
        transcript_block = "\n".join(
            f"[{l.speaker_name} · {'CLIENT' if l.speaker.value == 'client' else 'US'}]: {l.text}"
            for l in transcript[-config.CLAUDE_TRANSCRIPT_WINDOW:]
        )
        bio_block = self._summarise_biomarkers(biomarkers) or "Biomarker baseline calibrating."
        thymia_block = self._format_thymia_hints()

        return f"""You are Common Ground, the live intelligence layer for a product manager in a client meeting.

# Meeting context
- Organisation: {context.org_name}
- Session type: {context.session_type}
- Delivery stage: {context.stage}
- Objective: {context.objective}

# Stakeholders
{stakeholder_block}

# Active regulatory / domain frames
{reg_block}

# Recent biomarker signal from Thymia Sentinel
{bio_block}

# Recent Thymia policy guidance (their AI's read on the conversation)
{thymia_block}

# Recent transcript
{transcript_block}

# Your task
Produce 0-3 pieces of live, actionable intelligence for the product manager's private dashboard. Prioritise:

1. **Polite-yes detection** — client agreement with elevated cognitive load/hesitation. Use behavioural notes to interpret correctly.
2. **Persona-adapted reads** — use the specific behavioural notes to interpret silence/hesitation for this person.
3. **Regulatory blind spots** — active frames that apply to what's being discussed but aren't being addressed.
4. **Value signals** — where the client showed real enthusiasm worth building on.
5. **Probe-deeper moments** — ambiguity that needs follow-up.
6. **Wellbeing watchdog** — Thymia biomarkers suggesting the PM or client is getting stressed, fatigued, or losing focus. Coaching suggestion.

Return JSON only, no prose, strict schema:
```
{{
  "items": [
    {{
      "kind": "polite-yes" | "persona-read" | "regulatory" | "value" | "probe" | "wellbeing" | "self-coach",
      "type": "critical" | "warning" | "coaching" | "regulatory" | "value",
      "type_label": "4 words max — e.g. 'Polite-yes · Maria'",
      "title": "Headline, 10 words max",
      "detail": "2-3 sentences grounded in what was just said",
      "suggestion": "Specific phrase or question the PM could use next",
      "suggestion_label": "Probe / Regulatory probe / Reframe / Value lever / Self-coaching",
      "signals": [{{"label": "Cog. load elev.", "class": "hot"}}]
    }}
  ]
}}
```

Return `{{"items": []}}` if nothing new since last analysis. Do not repeat prior observations."""

    def _fmt_stakeholder(self, s) -> str:
        side = "CLIENT" if s.is_client else "OUR TEAM (PM/consultant)"
        parts = [f"- {s.name or '(unnamed)'} · {s.role or 'role unspecified'} · {side}"]
        if s.notes:
            parts.append(f"  Behavioural notes: {s.notes}")
        if s.mbti:
            parts.append(f"  Optional type input: {s.mbti} (one weak signal — behaviour outweighs this)")
        return "\n".join(parts)

    def _summarise_biomarkers(self, updates: list[BiomarkerUpdate]) -> str:
        if not updates:
            return ""
        recent = updates[-12:]
        lines = []
        for u in recent:
            signals = []
            if u.cognitive_load is not None and u.cognitive_load > 0.65:
                signals.append(f"cog_load {u.cognitive_load:.2f}")
            if u.stress is not None and u.stress > 0.6:
                signals.append(f"stress {u.stress:.2f}")
            if u.wellness is not None and u.wellness < 0.45:
                signals.append(f"wellness {u.wellness:.2f} (low)")
            if u.engagement is not None and u.engagement < 0.45:
                signals.append(f"engagement {u.engagement:.2f} (dropping)")
            if signals:
                lines.append(f"- {u.speaker_name} ({u.speaker.value}): {', '.join(signals)}")
        return "\n".join(lines) if lines else "No significant deviations."

    def _format_thymia_hints(self) -> str:
        if not self._thymia_hints:
            return "No policy actions yet."
        return "\n".join(
            f"- ({h['urgency']}) re: {h['speaker_name']}: {h['for_agent']}"
            for h in self._thymia_hints[-3:]
        )

    # ====================================================================
    # INTERVIEW MODE — summary followed by 2-3 adaptive follow-up questions
    # ====================================================================
    async def generate_interview_sequence(
        self,
        context: MeetingContext,
        transcript: list[TranscriptLine],
        biomarkers: list[BiomarkerUpdate],
    ) -> list[AgentUtterance]:
        """
        Return a sequence of utterances for a summary-then-questions flow:
          [0] A conversational summary of what's been discussed
          [1..n] 2-3 follow-up questions that probe the summary
        Each question is spoken in turn, waiting for the person to respond between.
        """
        if self.mock:
            return [
                AgentUtterance(text="So to summarise — we've explored the six-month migration with parallel-run as the safety net, and the client's Q3 board timeline as the pressure point. Let me ask a few follow-ups.", purpose="summary"),
                AgentUtterance(text="What would need to be true for the six-month window to feel comfortable, rather than tight?", purpose="probe"),
                AgentUtterance(text="If we had to simplify the board narrative, what's the one thing you'd want them to walk away hearing?", purpose="probe"),
                AgentUtterance(text="And — what's still unsaid that you'd want to put on the table before we close?", purpose="probe"),
            ]

        if not transcript:
            return []

        prompt = self._build_interview_prompt(context, transcript, biomarkers)
        try:
            client = self._sdk()
            msg = await client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            print(f"[ClaudeBrain] interview sequence call failed: {e}")
            return []

        text = msg.content[0].text if msg.content else ""
        return self._parse_interview_sequence(text)

    def _build_interview_prompt(
        self,
        context: MeetingContext,
        transcript: list[TranscriptLine],
        biomarkers: list[BiomarkerUpdate],
    ) -> str:
        transcript_block = "\n".join(
            f"[{l.speaker_name}]: {l.text}"
            for l in transcript[-30:]
        )
        bio_block = self._summarise_biomarkers(biomarkers) or "Biomarkers calibrating."

        return f"""You are the voice of an AI facilitator in a client meeting. The product manager ({self._pm_name(context)}) has asked you to run a short summary-and-questions cycle.

# Context
- Organisation: {context.org_name}
- Objective: {context.objective}
- Session type: {context.session_type}

# Recent conversation
{transcript_block}

# Biomarker signal
{bio_block}

# Your task
Produce ONE conversational summary, then 2 or 3 follow-up questions that probe what's been said. Choose 2 questions if the conversation is straightforward; choose 3 if there are real tensions or blind spots worth surfacing.

# Tone
- Conversational and natural — as if you were a skilled facilitator speaking aloud
- Not formal, not clinical
- Short sentences. You are speaking, not writing.
- Don't preamble ("Let me summarise for you..."). Just summarise.

# Rules
- Each utterance is 1-2 sentences. Short.
- Questions must be SPECIFIC to what was actually said, not generic.
- Never commit on behalf of the company.
- Never disclose biomarker data to the room.

# Output format
Return JSON only, strict schema:
```
{{
  "utterances": [
    {{"purpose": "summary", "text": "The summary, spoken aloud, 2-3 sentences max"}},
    {{"purpose": "probe", "text": "First follow-up question"}},
    {{"purpose": "probe", "text": "Second follow-up question"}},
    {{"purpose": "probe", "text": "Optional third follow-up question"}}
  ]
}}
```

Return only the JSON."""

    def _parse_interview_sequence(self, text: str) -> list[AgentUtterance]:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        out = []
        for item in data.get("utterances", []):
            t = (item.get("text") or "").strip()
            if not t:
                continue
            out.append(AgentUtterance(
                text=t,
                purpose=item.get("purpose", "probe"),
            ))
        # Cap at 4 (summary + 3 questions)
        return out[:4]

    # ====================================================================
    # VOICE AGENT SPEECH — decides if/what the Gradium agent should say
    # ====================================================================
    async def generate_agent_utterance(
        self,
        context: MeetingContext,
        transcript: list[TranscriptLine],
        biomarkers: list[BiomarkerUpdate],
        forced_purpose: Optional[str] = None,
    ) -> Optional[AgentUtterance]:
        """
        Decide whether the voice agent should interject. If yes, return the
        utterance text + purpose. If no, return None.

        The voice agent speaks in the meeting (via Gradium TTS) to:
        - Ask thought-provoking questions when conversation stalls
        - Steer back to the objective when drift is detected
        - Calm the room when stress biomarkers spike
        - Introduce structure at natural pauses

        If `forced_purpose` is set (e.g. 'probe', 'steer', 'thought-provoker'),
        the agent MUST speak with that purpose. This is used by the UI when
        the PM explicitly clicks "Probe" / "Steer" / etc.

        Thymia's for_agent guidance is incorporated directly.
        """
        if self.mock:
            return self._mock_utterance(transcript, biomarkers, forced_purpose)
        if not transcript:
            return None

        prompt = self._build_agent_prompt(context, transcript, biomarkers, forced_purpose)
        try:
            client = self._sdk()
            msg = await client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            print(f"[ClaudeBrain] agent call failed: {e}")
            return None

        text = msg.content[0].text if msg.content else ""
        return self._parse_agent_utterance(text)

    def _build_agent_prompt(
        self,
        context: MeetingContext,
        transcript: list[TranscriptLine],
        biomarkers: list[BiomarkerUpdate],
        forced_purpose: Optional[str] = None,
    ) -> str:
        transcript_block = "\n".join(
            f"[{l.speaker_name}]: {l.text}"
            for l in transcript[-20:]
        )
        bio_block = self._summarise_biomarkers(biomarkers) or "Biomarkers calibrating."
        thymia_block = self._format_thymia_hints()

        forced_clause = ""
        if forced_purpose:
            forced_clause = (
                f"\n\n# IMPORTANT OVERRIDE\n"
                f"The product manager has explicitly asked you to speak with purpose = \"{forced_purpose}\". "
                f"You MUST set `speak: true` and `purpose: \"{forced_purpose}\"`. "
                f"Generate an appropriate utterance for this purpose based on the conversation."
            )

        return f"""You are the voice of an AI facilitator embedded in a client meeting. The product manager ({self._pm_name(context)}) is discussing {context.objective} with the client team. You rarely speak unprompted, but when you do, it is to move the conversation forward, keep it on track, surface something that's being missed, or de-escalate tension.

# Rules
- You must be rare when speaking on your own judgement. If the conversation is flowing well, stay silent.
- Your voice is calm, professional, neutral. You are not a participant — you are a facilitator.
- You never make commitments on behalf of the company.
- You never disclose client-side biomarkers to the room.
- You speak in short sentences (1-2 max). Your goal is to open space, not take it.

# Purposes
- "probe": ask a specific follow-up on something just said
- "steer": redirect if the conversation is drifting off-topic
- "calm": slow the pace if tension has risen
- "thought-provoker": open a new angle / ask "what if" / surface what's being missed
- "summary": pull together what has been agreed or explored

# Current conversation
{transcript_block}

# Biomarker signal
{bio_block}

# Thymia policy guidance
{thymia_block}{forced_clause}

# Your decision
If no `forced_purpose` is set above, decide whether to speak. Return JSON only:
```
{{
  "speak": true | false,
  "purpose": "probe" | "steer" | "calm" | "thought-provoker" | "summary",
  "text": "What to say, 1-2 short sentences",
  "reason": "Why now (one sentence, for logs)"
}}
```

Examples of good utterances by purpose:
- probe: "Before we move on — what concerns would we want to raise with the compliance team here?"
- steer: "It feels like we've touched on two different priorities. Would it help to decide which to focus on?"
- calm: "Let's pause for a moment. Any reservations that haven't been said?"
- thought-provoker: "If we paused this project for a week — what would we regret not having asked today?"
- summary: "So we've aligned on the six-month timeline with the parallel-run safety net. Fair summary?"

Return only the JSON."""

    def _pm_name(self, context: MeetingContext) -> str:
        pm = next((s for s in context.stakeholders if not s.is_client), None)
        return pm.name if pm else "the product manager"

    # ====================================================================
    # PARSING
    # ====================================================================
    def _parse_alerts(self, text: str) -> list[Alert]:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        out = []
        for item in data.get("items", []):
            sig = f"{item.get('kind')}::{(item.get('title') or '')[:40].lower()}"
            if sig in self._prior_alert_sigs:
                continue
            self._prior_alert_sigs.append(sig)
            out.append(Alert(
                kind=item.get("kind", "coaching"),
                type=item.get("type", "coaching"),
                type_label=item.get("type_label", "Intelligence"),
                title=item.get("title", ""),
                detail=item.get("detail", ""),
                suggestion=item.get("suggestion", ""),
                suggestion_label=item.get("suggestion_label", ""),
                signals=item.get("signals", []),
                source="claude",
            ))
        return out

    def _parse_agent_utterance(self, text: str) -> Optional[AgentUtterance]:
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if not data.get("speak"):
            return None
        return AgentUtterance(
            text=data.get("text", ""),
            purpose=data.get("purpose", "probe"),
        )

    # ====================================================================
    # MOCKS
    # ====================================================================
    def _mock_alerts(self) -> list[Alert]:
        import random
        if random.random() > 0.5:
            return []
        templates = [
            Alert(
                kind="polite-yes",
                type="critical",
                type_label="Polite-yes · mock",
                title="Client agreement masks hesitation",
                detail="Biomarkers show cognitive load rising before the 'yes'. Worth probing before locking.",
                suggestion="Before we lock this in — what's giving you pause on the timeline?",
                suggestion_label="Probe in your voice",
                signals=[{"label": "Cog. load elev.", "class": "hot"}, {"label": "Latency 2.8s", "class": "hot"}],
                source="claude",
            ),
            Alert(
                kind="regulatory",
                type="regulatory",
                type_label="Consumer Duty",
                title="Fair value assessment not addressed",
                detail="Pricing came up but fair value for the target market has not been discussed.",
                suggestion="How are you evidencing fair value for the target market?",
                suggestion_label="Regulatory probe",
                signals=[{"label": "Consumer Duty", "class": "warm"}],
                source="claude",
            ),
            Alert(
                kind="wellbeing",
                type="coaching",
                type_label="Wellbeing watch",
                title="Your stress is climbing — client is reading it",
                detail="Your own stress biomarker has risen 30% since minute 4. The client's engagement is trending down in response.",
                suggestion="Pause for a breath. Your parallel-run point is your strongest — it carries.",
                suggestion_label="Self-coaching",
                signals=[{"label": "Your stress +30%", "class": "warm"}],
                source="claude",
            ),
        ]
        return [random.choice(templates)]

    def _mock_utterance(self, transcript, biomarkers, forced_purpose=None) -> Optional[AgentUtterance]:
        templates = {
            "probe": "Before we move on — what concerns would we want to raise with compliance on this?",
            "steer": "It sounds like two priorities have come into view. Shall we pin one first?",
            "calm": "Let's pause for a moment. Any reservations that haven't been said?",
            "thought-provoker": "If we paused the project for a week — what would we regret not having asked today?",
            "summary": "So we've aligned on the six-month timeline with the parallel-run safety net. Fair summary?",
        }
        if forced_purpose:
            text = templates.get(forced_purpose, templates["probe"])
            return AgentUtterance(text=text, purpose=forced_purpose)
        if not transcript or len(transcript) < 4:
            return None
        import random
        if random.random() > 0.25:
            return None
        purpose = random.choice(list(templates.keys()))
        return AgentUtterance(text=templates[purpose], purpose=purpose)

    # ====================================================================
    # CLOSING DEBRIEF — spoken summary + open questions at session end
    # ====================================================================
    async def generate_closing_debrief(
        self,
        context: MeetingContext,
        transcript: list[TranscriptLine],
        alerts: list[Alert],
        biomarkers: list[BiomarkerUpdate],
    ) -> str:
        """
        Produce a short debrief (~30 seconds spoken) that the voice agent
        delivers at the end of the session.

        Structure:
        - 1-2 sentences summarising what was agreed / covered
        - 2-3 open reflection questions to leave the team with

        Returns the text — the session will pipe this through Gradium TTS.
        """
        if not transcript:
            return "No conversation was recorded in this session. Thank you for your time."
        if self.mock:
            return self._mock_debrief(context, transcript)

        prompt = self._build_debrief_prompt(context, transcript, alerts, biomarkers)
        try:
            client = self._sdk()
            msg = await client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            print(f"[ClaudeBrain] debrief call failed: {e}")
            return self._mock_debrief(context, transcript)

        text = msg.content[0].text if msg.content else ""
        # Strip any preamble / markdown
        return text.strip().strip('"').strip()

    def _build_debrief_prompt(
        self,
        context: MeetingContext,
        transcript: list[TranscriptLine],
        alerts: list[Alert],
        biomarkers: list[BiomarkerUpdate],
    ) -> str:
        transcript_block = "\n".join(
            f"[{l.speaker_name}]: {l.text}"
            for l in transcript[-60:]
        )
        alerts_block = "\n".join(
            f"- ({a.type}) {a.title}: {a.detail}"
            for a in alerts[-8:]
        ) or "No alerts recorded."

        return f"""You are the voice of the AI facilitator closing a client meeting. The session has just ended. Your job is to deliver a short closing debrief that the whole room will hear — spoken aloud via voice synthesis.

# Meeting context
- Organisation: {context.org_name}
- Objective: {context.objective}
- Session type: {context.session_type}

# Transcript
{transcript_block}

# Private intelligence flagged during the session
{alerts_block}

# Your task
Produce a spoken closing debrief of about 30 seconds (around 70-90 words total). Structure:

1. **Recap (1-2 sentences)**: What was actually agreed or explored? Ground in what was said, not what could have been.

2. **2-3 open reflection questions for the team**: Not rhetorical. Real questions the room could answer now or take away. These should:
   - Draw on ambiguity or unresolved tension from the conversation
   - Invite the team to reflect together, not to be graded
   - Be phrased conversationally, the way a good facilitator would ask

# Rules
- Spoken delivery — short sentences, natural rhythm, no bullet points, no markdown
- Neutral, warm, professional tone — you are a facilitator, not a participant
- Do NOT disclose biomarker data or private intelligence alerts to the room
- Do NOT commit on behalf of anyone or make decisions
- Do NOT fabricate things that weren't said
- Open with something natural like "To close us out..." or "Before we wrap..."

Return only the spoken text. No JSON, no preamble, no explanations."""

    def _mock_debrief(self, context: MeetingContext, transcript: list[TranscriptLine]) -> str:
        return (
            f"To close us out — we've explored {context.objective.split('.')[0].lower()} "
            f"and surfaced the key tensions between timeline and board reporting. "
            f"A few questions to leave you with: What's the one concern each of you wants "
            f"answered before we meet again? Is there anyone not in this room whose view "
            f"we should hear before the next step? And if we paused this for a week, "
            f"what would we regret not asking today? Thank you for your time."
        )
