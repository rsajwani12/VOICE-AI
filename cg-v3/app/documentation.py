"""
TinyFish integration — meeting documentation.

TinyFish is a web agent platform. For our use case we use it as the
*documentation layer*: at the end of a session, the backend hands TinyFish
the transcript + alerts + biomarker summary and TinyFish produces an
HTML report we can share.

For the hack we use a local renderer for the documentation (since full
TinyFish integration requires web agent orchestration beyond our time
budget). The placeholder matches the shape that would integrate with
TinyFish's actual web agent API when credentials are wired up.
"""
import json
import time
from pathlib import Path

from . import config
from .models import MeetingContext, TranscriptLine, Alert, BiomarkerUpdate


def generate_meeting_report(
    context: MeetingContext,
    transcript: list[TranscriptLine],
    alerts: list[Alert],
    biomarkers: list[BiomarkerUpdate],
) -> str:
    """
    Produce an HTML meeting report and return the absolute path.
    """
    out_dir = config.ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    ts = int(time.time())
    out_path = out_dir / f"session-{ts}.html"

    html = _render_report(context, transcript, alerts, biomarkers)
    out_path.write_text(html)
    return str(out_path)


def _render_report(
    context: MeetingContext,
    transcript: list[TranscriptLine],
    alerts: list[Alert],
    biomarkers: list[BiomarkerUpdate],
) -> str:
    flagged = [a for a in alerts if a.type in ("critical", "warning")]
    coaching = [a for a in alerts if a.type == "coaching"]
    regulatory = [a for a in alerts if a.type == "regulatory"]
    value = [a for a in alerts if a.type == "value"]

    def render_alerts(group, heading):
        if not group:
            return ""
        items = "".join(
            f'<div class="alert"><div class="alert-head"><span class="alert-type">{a.type_label}</span></div>'
            f'<div class="alert-title">{_esc(a.title)}</div>'
            f'<div class="alert-detail">{_esc(a.detail)}</div>'
            + (f'<div class="alert-suggestion"><strong>{_esc(a.suggestion_label)}:</strong> {_esc(a.suggestion)}</div>' if a.suggestion else '')
            + '</div>'
            for a in group
        )
        return f'<h2>{heading}</h2>{items}'

    transcript_lines = "".join(
        f'<div class="line {l.speaker.value}"><span class="time">{int(l.ts_start)//60:02d}:{int(l.ts_start)%60:02d}</span>'
        f'<span class="speaker">{_esc(l.speaker_name)}</span>'
        f'<span class="text">{_esc(l.text)}</span></div>'
        for l in transcript
    )

    stakeholder_rows = "".join(
        f'<tr><td>{_esc(s.name)}</td><td>{_esc(s.role)}</td>'
        f'<td>{"Client" if s.is_client else "Our team"}</td>'
        f'<td>{_esc(s.notes or "—")}</td></tr>'
        for s in context.stakeholders
    )

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Meeting Report — {_esc(context.org_name)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300..900;1,300..900&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "JetBrains Mono", monospace;
      background: #faf8f4;
      color: #1a1a1d;
      padding: 48px;
      max-width: 960px;
      margin: 0 auto;
      font-size: 13px;
      line-height: 1.6;
    }}
    h1, h2, h3 {{ font-family: "Fraunces", serif; font-weight: 400; letter-spacing: -0.02em; }}
    h1 {{ font-size: 40px; margin-bottom: 8px; }}
    h1 em {{ font-style: italic; color: #b8895a; }}
    h2 {{ font-size: 22px; margin: 32px 0 14px; padding-bottom: 6px; border-bottom: 1px solid #e6e2d8; }}
    .meta {{
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
      margin: 24px 0; padding: 18px; background: #fff; border: 1px solid #e6e2d8;
    }}
    .meta .k {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.18em; color: #888; }}
    .meta .v {{ font-size: 16px; font-family: "Fraunces", serif; font-style: italic; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #e6e2d8; font-size: 12px; }}
    th {{ font-size: 9px; text-transform: uppercase; letter-spacing: 0.14em; color: #888; }}
    .alert {{
      padding: 14px 18px; margin-bottom: 10px; border: 1px solid #e6e2d8;
      border-left: 3px solid #c9553a; background: rgba(201,85,58,0.04);
    }}
    .alert-head {{ display: flex; justify-content: space-between; font-size: 9px; text-transform: uppercase; letter-spacing: 0.16em; color: #c9553a; }}
    .alert-title {{ font-family: "Fraunces", serif; font-size: 17px; font-weight: 500; margin: 6px 0; }}
    .alert-detail {{ font-size: 12px; color: #444; margin-bottom: 8px; }}
    .alert-suggestion {{ font-size: 12px; padding: 10px; background: #fff; border: 1px dashed #e6e2d8; }}
    .line {{ display: grid; grid-template-columns: 50px 90px 1fr; gap: 12px; padding: 8px 0; border-bottom: 1px solid #f0ece2; font-size: 13px; }}
    .line .time {{ color: #aaa; font-variant-numeric: tabular-nums; }}
    .line .speaker {{ font-weight: 500; font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; padding-top: 2px; }}
    .line.consultant .speaker {{ color: #b8895a; }}
    .line.client .speaker {{ color: #3d5878; }}
    .footer {{ margin-top: 60px; padding-top: 20px; border-top: 1px solid #e6e2d8; font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 0.18em; }}
  </style>
</head>
<body>
  <h1>Meeting <em>report</em></h1>
  <p style="font-family: 'Fraunces', serif; font-style: italic; font-size: 18px; color: #555;">
    {_esc(context.org_name)} · {_esc(context.session_type)}
  </p>

  <div class="meta">
    <div><div class="k">Duration</div><div class="v">{(int(transcript[-1].ts_end) // 60) if transcript else 0} min</div></div>
    <div><div class="k">Delivery stage</div><div class="v">{_esc(context.stage)}</div></div>
    <div><div class="k">Utterances</div><div class="v">{len(transcript)}</div></div>
    <div><div class="k">Flagged</div><div class="v">{len(flagged)}</div></div>
  </div>

  <h2>Objective</h2>
  <p style="font-family: 'Fraunces', serif; font-style: italic; font-size: 16px;">{_esc(context.objective)}</p>

  <h2>Stakeholders</h2>
  <table>
    <tr><th>Name</th><th>Role</th><th>Side</th><th>Behavioural notes</th></tr>
    {stakeholder_rows}
  </table>

  {render_alerts(flagged, "Flagged moments")}
  {render_alerts(regulatory, "Regulatory signal")}
  {render_alerts(value, "Value signal")}
  {render_alerts(coaching, "Coaching notes")}

  <h2>Transcript</h2>
  <div style="background: #fff; border: 1px solid #e6e2d8; padding: 18px;">
    {transcript_lines or '<em style="color: #aaa;">No transcript captured.</em>'}
  </div>

  <div class="footer">
    Common Ground · Private to consulting team · Generated {time.strftime("%d %b %Y %H:%M")}
  </div>
</body>
</html>"""


def _esc(s) -> str:
    s = str(s or "")
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
