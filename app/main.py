import base64, json, os, re, html, time, threading, random
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# load persona
try:
    from .persona_prompts import PERSONA_BLESSED_BOY
except Exception:
    from persona_prompts import PERSONA_BLESSED_BOY

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "ap-south-1")
BEDROCK_MODEL  = os.getenv("BEDROCK_MODEL",  "anthropic.claude-3-haiku-20240307-v1:0")
POLLY_REGION   = os.getenv("POLLY_REGION",   "ap-south-1")
POLLY_FALLBACK_REGION = os.getenv("POLLY_FALLBACK_REGION", "us-east-1")
POLLY_VOICE    = os.getenv("POLLY_VOICE",    "Ruth")
POLLY_RATE     = os.getenv("POLLY_RATE",     "medium")
POLLY_PITCH    = os.getenv("POLLY_PITCH",    "+4%")

# Add adaptive retries on clients
bedrock = boto3.client(
    "bedrock-runtime",
    config=Config(region_name=BEDROCK_REGION, retries={"max_attempts": 3, "mode": "adaptive"})
)
polly   = boto3.client(
    "polly",
    config=Config(region_name=POLLY_REGION, retries={"max_attempts": 3, "mode": "standard"})
)

# Optional fallback region (keeps the same voice, different region)
polly_fb = None
if POLLY_FALLBACK_REGION and POLLY_FALLBACK_REGION != POLLY_REGION:
    try:
        polly_fb = boto3.client(
            "polly",
            config=Config(region_name=POLLY_FALLBACK_REGION, retries={"max_attempts": 3, "mode": "standard"})
        )
    except Exception:
        polly_fb = None

def _polly_clients():
    # Try primary region first, then fallback if configured
    return [c for c in (polly, polly_fb) if c is not None]

# ---- app --------------------------------------------------------------------
app = FastAPI()
logger = logging.getLogger("blessedboy")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def root():
    # Disable caching for the shell HTML so updates (like audio/animation fixes) deploy reliably
    return FileResponse(str(STATIC_DIR / "index.html"), headers={"Cache-Control": "no-store"})

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/version")
def version():
    return {
        "image_tag": os.getenv("IMAGE_TAG", "unknown"),
    "build": os.getenv("APP_BUILD", "unknown"),
        "bedrock_model": BEDROCK_MODEL,
        "polly_region": POLLY_REGION,
        "polly_voice": POLLY_VOICE,
    }

# ---- rolling memory (per session_id) ----------------------------------------
MAX_TURNS = 10  # user+assistant pairs
_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS*2))
def add_turn(session_id: str, role: str, content: str):
    _history[session_id].append({"role": role, "content": content})

def get_msgs(session_id: str) -> List[Dict]:
    msgs = []
    for m in _history[session_id]:
        msgs.append({
            "role": "user" if m["role"] == "user" else "assistant",
            "content": [{"type": "text", "text": m["content"]}]
        })
    return msgs

# ---- models -----------------------------------------------------------------
class ChatIn(BaseModel):
    text: str
    session_id: str = "local"
    style: Optional[str] = None  # conversational style (e.g., 'witty','precise','empathetic')

class TTSIn(BaseModel):
    text: str
    lang: Optional[str] = None   # e.g., 'en','es','fr'
    mode: Optional[str] = None   # 'auto' chooses a female voice by language; default uses Ruth

class ChatStreamIn(BaseModel):
    text: str
    session_id: str = "local"
    style: Optional[str] = None

class SingIn(BaseModel):
    text: str  # user-provided lyrics only
    lang: Optional[str] = None
    mode: Optional[str] = None  # 'auto' to select a native female voice for that language

# ---- helpers ----------------------------------------------------------------
ACTION_PATTERNS = [
    r"\*[^*]{0,120}\*", r"\[[^\]]{0,120}\]",
    r"\((?:smiles|laughs|chuckles|sighs|clears throat|giggles)[^)]*\)"
]
def strip_stage(text: str) -> str:
    t = text
    for pat in ACTION_PATTERNS:
        t = re.sub(pat, "", t, flags=re.I)
    t = re.sub(r"^As an AI(?: language model)?[, ]*", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def enforce_identity(text: str) -> str:
    t = re.sub(r"\bClaude\b", "Rem", text, flags=re.I)
    t = re.sub(r"\bBlessed Boy\b", "Rem", t, flags=re.I)
    t = re.sub(r"\bAnthropic\b", "my team", t, flags=re.I)
    # Remove leading assistant name tags like "Rem:", "Rem -", "Rem." at the start
    t = re.sub(r"^\s*Rem\s*[:\-–—.,]\s*", "", t, flags=re.I)
    return strip_stage(t)

def clamp_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(parts[:n]).strip()

# ---- LLM --------------------------------------------------------------------
BEDROCK_MAX_RETRIES = int(os.getenv("BEDROCK_MAX_RETRIES", "3"))
CHAT_MAX_CONCURRENCY = int(os.getenv("CHAT_MAX_CONCURRENCY", "4"))
_chat_gate = threading.Semaphore(CHAT_MAX_CONCURRENCY)

STYLE_GUIDES = {
    "witty": "Style: Witty, playful, and concise. Use tasteful humor, friendly teasing, and natural contractions.",
    "precise": "Style: Direct and clear. Short sentences, minimal filler.",
    "empathetic": "Style: Warm, supportive, and validating. Brief, caring responses.",
    "spicy": (
        "Style: Flirty, cheeky, and charming (PG-13). Be respectful and consensual. No explicit content or slurs; "
        "keep it playful with compliments and light banter only. Politely decline crossing any boundary."
    ),
}

def _compose_system(base: str, style: Optional[str]) -> str:
    s = (style or "").strip().lower()
    guide = STYLE_GUIDES.get(s)
    return f"{base}\n\n{guide}" if guide else base

def _user_for_style(text: str, style: Optional[str]) -> str:
    s = (style or "").strip().lower()
    if s in {"witty","spicy"}:
        # Nudge to conversational phrasing without changing intent
        return text
    return text

def bedrock_reply(system_prompt: str, session_id: str, user_text: str, style: Optional[str] = None) -> str:
    messages = get_msgs(session_id)
    messages.append({"role":"user","content":[{"type":"text","text":_user_for_style(user_text, style)}]})
    s = (style or "").strip().lower()
    temp = 0.7
    if s in ("witty","spicy"): temp = 0.9
    elif s == "precise": temp = 0.4
    elif s == "empathetic": temp = 0.7
    body = {
        "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 480,
        "temperature": temp,     # style-aware variety
        "top_p": 0.9,
        "system": _compose_system(system_prompt, style),
        "messages": messages,
    }
    last_err = None
    for attempt in range(BEDROCK_MAX_RETRIES):
        try:
            r = bedrock.invoke_model(
                modelId=BEDROCK_MODEL, accept="application/json",
                contentType="application/json", body=json.dumps(body)
            )
            data = json.loads(r["body"].read())
            break
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "ClientError")
            if code in {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"}:
                last_err = e; _retry_sleep(attempt); continue
            raise
    else:
        raise last_err or RuntimeError("Bedrock retries exhausted")
    out = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            out += block.get("text") or ""
    return clamp_sentences(enforce_identity(out) or "I'm here.")

def _stream_bedrock_text(model_id: str, system_prompt: str, messages: list, style: Optional[str] = None):
    s = (style or "").strip().lower()
    temp = 0.7
    if s in ("witty","spicy"): temp = 0.9
    elif s == "precise": temp = 0.4
    elif s == "empathetic": temp = 0.7
    body = {
        "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 480,
        "temperature": temp,
        "top_p": 0.9,
        "system": system_prompt,
        "messages": messages,
    }
    last_err = None
    for attempt in range(BEDROCK_MAX_RETRIES):
        try:
            resp = bedrock.invoke_model_with_response_stream(
                modelId=model_id, accept="application/json",
                contentType="application/json", body=json.dumps(body)
            )
            for ev in resp["body"]:
                chunk = ev.get("chunk", {}).get("bytes")
                if not chunk:
                    continue
                data = json.loads(chunk.decode("utf-8"))
                if data.get("type") == "content_block_delta":
                    d = data.get("delta", {})
                    if d.get("type") == "text_delta":
                        yield d.get("text", "")
            return
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "ClientError")
            if code in {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"}:
                last_err = e; _retry_sleep(attempt); continue
            raise
    raise last_err or RuntimeError("Bedrock stream retries exhausted")

# ---- Polly TTS (SSML pacing + visemes) --------------------------------------
def _normalize_lang(lang: Optional[str]) -> Optional[str]:
    if not lang: return None
    l = lang.strip().lower()
    # Normalize to IETF-like codes Polly expects
    m = {
        "es": "es-ES", "es-es": "es-ES", "es-mx": "es-MX",
        "fr": "fr-FR", "fr-fr": "fr-FR", "fr-ca": "fr-CA",
        "hi": "hi-IN", "hi-in": "hi-IN",
        "en": "en-US",
    }
    return m.get(l, None)

def make_ssml(text: str, lang: Optional[str] = None) -> str:
    import re as _re
    # Sentence-level phrasing with short breaks for natural pacing
    sents = [html.escape(s.strip()) for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    inner = "<break time='80ms'/>".join(f"<s>{s}</s><break time='120ms'/>" for s in sents)

    # Language-aware pacing
    def _rate_pitch_for_lang(l: Optional[str]):
        rate, pitch = POLLY_RATE, POLLY_PITCH
        n = _normalize_lang(l) or ""
        if n.startswith("es-") or n == "es-ES":
            rate = "-10%"  # Spanish a bit slower
        elif n.startswith("fr-") or n == "fr-FR":
            rate = "-10%"  # French a bit slower
        elif n.startswith("hi-") or n == "hi-IN":
            rate = "-5%"   # Slightly slower Hindi
        return rate, pitch

    rate, pitch = _rate_pitch_for_lang(lang)
    lang_norm = _normalize_lang(lang)
    if lang_norm:
        return f"<speak><lang xml:lang='{lang_norm}'><prosody rate='{rate}' pitch='{pitch}'>{inner}</prosody></lang></speak>"
    return f"<speak><prosody rate='{rate}' pitch='{pitch}'>{inner}</prosody></speak>"



def make_sing_ssml(text: str, lang: Optional[str] = None) -> str:
    """A more melodic singing SSML: phrase contours + gentle rests to feel musical.
    Polly can't truly sing; this simulates melody with pitch steps and pacing.
    """
    import re as _re
    # Split into lines and phrases; keep musical rests between phrases
    lines = _re.split(r"\n+", (text or "").strip())
    phrases: List[str] = []
    for line in lines:
        phrases.extend([p.strip() for p in _re.split(r"[.!?;,]\s*", line) if p.strip()])

    # Slightly varied contours to avoid feeling repetitive
    contours = [
        [+2, +5, +9,  +7, +4,  +2,  0,  +3],
        [+1, +3, +6,  +8, +10, +8, +5, +2],
        [ 0, +2, +4,  +7, +9,  +7, +4, +1],
    ]

    def _stylize_word(tok: str, step: int) -> str:
        t = html.escape(tok)
        # Emulate sustained vowels for longer words
        if len(tok) >= 7 and any(v in tok.lower() for v in "aeiou"):
            return f"<prosody pitch='{step}%'><prosody rate='-10%'>{t}</prosody></prosody>"
        return f"<prosody pitch='{step}%'>{t}</prosody>"

    parts: List[str] = ["<amazon:auto-breaths>\n"]
    for idx, ph in enumerate(phrases):
        tokens = [t for t in _re.split(r"(\s+)", ph) if t]
        contour = contours[idx % len(contours)]
        k = 0
        # Add a touch of emphasis to phrase starts
        parts.append("<emphasis level='moderate'>")
        for tok in tokens:
            if tok.isspace():
                parts.append(tok)
            else:
                pitch = contour[k % len(contour)]
                parts.append(_stylize_word(tok, pitch))
                k += 1
        parts.append("</emphasis>")
        # small rest at end of phrase
        parts.append("<break time='260ms'/>")
    parts.append("\n</amazon:auto-breaths>")

    inner = "".join(parts) or html.escape(text)

    # Language-aware base rate for singing (slower, musical)
    lang_norm = _normalize_lang(lang) or ""
    base_rate = "-15%"
    if lang_norm.startswith("es-") or lang_norm == "es-ES":
        base_rate = "-20%"
    elif lang_norm.startswith("fr-") or lang_norm == "fr-FR":
        base_rate = "-18%"
    elif lang_norm.startswith("hi-") or lang_norm == "hi-IN":
        base_rate = "-12%"

    body = f"<prosody rate='{base_rate}'>{inner}</prosody>"
    if lang_norm:
        return f"<speak><lang xml:lang='{lang_norm}'>{body}</lang></speak>"
    return f"<speak>{body}</speak>"

_VOICES_CACHE = {}
def _get_voices(client=None):
    """Cache voices per region client."""
    c = client or polly
    key = getattr(c.meta, "region_name", "default")
    if key not in _VOICES_CACHE:
        try:
            _VOICES_CACHE[key] = c.describe_voices().get("Voices", [])
        except Exception:
            _VOICES_CACHE[key] = []
    return _VOICES_CACHE[key]

def _choose_voice(preferred: str, engine: str) -> str:
    """Return the preferred voice only (strict). Engine compatibility is handled by callers.

    We purposely do NOT fall back to other voices to avoid switching timbre mid-session.
    If the preferred voice is unavailable for the requested engine, the synth call will raise
    (e.g., EngineNotSupportedException or InvalidVoiceIdException) and the caller may try
    another engine (e.g., standard instead of neural).
    """
    return preferred

def _synthesize_audio(clean: str, voice: str, lang: Optional[str] = None) -> Tuple[bytes, str, str, any]:
    """Return (audio_bytes, engine_used, voice_used, polly_client_used).

    Always uses the preferred voice (POLLY_VOICE). Tries neural/standard and
    will switch to a fallback region if necessary, but never changes the voice.
    """
    plan = [
        ("neural",   "ssml", make_ssml(clean, lang)),
        ("standard", "ssml", make_ssml(clean, lang)),
        ("neural",   "text", clean),
        ("standard", "text", clean),
    ]
    last = None
    for client in _polly_clients():
        for engine, text_type, text in plan:
            voice_id = voice
            try:
                r = client.synthesize_speech(
                    VoiceId=voice_id, OutputFormat="mp3",
                    Text=text, TextType=text_type, Engine=engine
                )
                # If we had to use fallback region, log once
                try:
                    region = getattr(client.meta, "region_name", "")
                    if region and region != POLLY_REGION:
                        logger.warning(
                            "Polly voice %s synthesized from fallback region %s (primary %s)",
                            voice_id, region, POLLY_REGION
                        )
                except Exception:
                    pass
                return r["AudioStream"].read(), engine, voice_id, client
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in {"InvalidSsmlException","EngineNotSupportedException","TextLengthExceededException",
                            "InvalidVoiceIdException","UnsupportedPlsAlphabetException","LanguageNotSupportedException",
                            "ValidationException"}:
                    last = e; continue
                raise
        # try next client (fallback region)
    if last: raise last
    raise RuntimeError("Polly synthesis failed")

def _synthesize_ssml(ssml: str, voice: str) -> Tuple[bytes, str, str, any]:
    plan = [
        ("neural",   "ssml"),
        ("standard", "ssml"),
    ]
    last = None
    for client in _polly_clients():
        for engine, text_type in plan:
            voice_id = voice
            try:
                r = client.synthesize_speech(
                    VoiceId=voice_id, OutputFormat="mp3",
                    Text=ssml, TextType=text_type, Engine=engine
                )
                try:
                    region = getattr(client.meta, "region_name", "")
                    if region and region != POLLY_REGION:
                        logger.warning(
                            "Polly voice %s synthesized from fallback region %s (primary %s)",
                            voice_id, region, POLLY_REGION
                        )
                except Exception:
                    pass
                return r["AudioStream"].read(), engine, voice_id, client
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in {"InvalidSsmlException","EngineNotSupportedException","TextLengthExceededException",
                            "InvalidVoiceIdException","UnsupportedPlsAlphabetException","LanguageNotSupportedException",
                            "ValidationException"}:
                    last = e; continue
                raise
    if last: raise last
    raise RuntimeError("Polly SSML synthesis failed")

def _visemes(clean: str, engine: str, voice: str, client=None) -> list:
    c = client or polly
    try:
        r = c.synthesize_speech(
            Text=clean, VoiceId=voice, OutputFormat="json",
            SpeechMarkTypes=["viseme"], Engine=engine
        )
        return [json.loads(x) for x in r["AudioStream"].read().decode("utf-8").splitlines() if x]
    except Exception:
        return []

VOICE_MAP = {
    # English
    "en": "Ruth",
    # Spanish
    "es": "Lucia",      # Spain (Neural female)
    "es-mx": "Mia",     # Mexico (Neural female)
    # French
    "fr": "Lea",        # France (Neural female) — fallback to Celine if unavailable
    "fr-fr": "Lea",
    "fr-ca": "Chantal",  # Canada (female)
    # Hindi
    "hi": "Aditi",       # Bilingual hi-IN / en-IN female
    # Other examples kept
    "de": "Vicki",
    "it": "Bianca",
    "pt": "Camila",
    "ja": "Mizuki",
    "ko": "Seoyeon",
    "zh": "Zhiyu",
    "ar": "Zeina",
    "nl": "Lotte",
    "sv": "Astrid",
    "da": "Naja",
    "nb": "Liv",
    "pl": "Maja",
    "ru": "Tatyana",
    "tr": "Filiz",
}

def _voice_candidates(lang_hint: Optional[str], mode: Optional[str]) -> List[str]:
    if (mode or "").lower() != "auto":
        return [POLLY_VOICE]
    hint = (lang_hint or "en").lower()
    base = hint.split("-")[0]
    # Preference lists per dialect
    prefs = {
        "es-mx": ["Mia", "Lucia"],
        "es-es": ["Lucia", "Mia"],
        "es":    ["Lucia", "Mia"],
        "fr-ca": ["Chantal", "Lea", "Celine"],
        "fr-fr": ["Lea", "Celine", "Chantal"],
        "fr":    ["Lea", "Celine"],
        "hi-in": ["Aditi"],
        "hi":    ["Aditi"],
        "en":    [POLLY_VOICE],
    }
    return prefs.get(hint) or prefs.get(f"{base}-{'mx' if base=='es' else 'fr' if base=='fr' else 'in' if base=='hi' else ''}") or prefs.get(base) or [VOICE_MAP.get(hint) or VOICE_MAP.get(base) or VOICE_MAP["en"]]

def polly_tts_with_visemes(text: str, lang: Optional[str] = None, mode: Optional[str] = None) -> Tuple[str, list]:
    clean = strip_stage(text) or text
    candidates = _voice_candidates(lang, mode)
    last_exc = None
    for v in candidates:
        try:
            audio, engine, used_voice, client = _synthesize_audio(clean, v, lang)
            marks = _visemes(clean, engine, used_voice, client)
            return base64.b64encode(audio).decode("ascii"), marks
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"InvalidVoiceIdException","LanguageNotSupportedException","ValidationException","EngineNotSupportedException"}:
                last_exc = e; continue
            raise
    if last_exc: raise last_exc
    # Fallback to default
    audio, engine, used_voice, client = _synthesize_audio(clean, POLLY_VOICE, lang)
    marks = _visemes(clean, engine, used_voice, client)
    return base64.b64encode(audio).decode("ascii"), marks

def polly_sing_with_visemes(text: str, lang: Optional[str] = None, mode: Optional[str] = None) -> Tuple[str, list]:
    # Reject explicit lyrics (basic filter)
    bad = re.compile(r"\b(fuck|shit|bitch|asshole|slut|whore|dick|pussy|cunt|rape|kill|suicide)\b", re.I)
    if bad.search(text or ""):
        raise HTTPException(400, "I can't sing these lyrics.")
    # Normalize and lightly de-duplicate repeated lines
    cleaned_lines = []
    last = None
    for line in (text or "").splitlines():
        L = line.strip()
        if L and L != last:
            cleaned_lines.append(L)
            last = L
    clean = strip_stage("\n".join(cleaned_lines) or text) or text
    ssml = make_sing_ssml(clean, lang)
    candidates = _voice_candidates(lang, mode)
    last_exc = None
    for v in candidates:
        try:
            # Try singing SSML first
            try:
                audio, engine, used_voice, client = _synthesize_ssml(ssml, v)
                marks = _visemes(clean, engine, used_voice, client)
                return base64.b64encode(audio).decode("ascii"), marks
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                # Graceful fallback: if SSML is invalid or not supported, use regular TTS SSML
                if code in {"InvalidSsmlException", "ValidationException"}:
                    audio2, engine2, used_voice2, client2 = _synthesize_audio(clean, v, lang)
                    marks2 = _visemes(clean, engine2, used_voice2, client2)
                    return base64.b64encode(audio2).decode("ascii"), marks2
                else:
                    raise
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"InvalidVoiceIdException","LanguageNotSupportedException","ValidationException","EngineNotSupportedException","InvalidSsmlException"}:
                last_exc = e; continue
            raise
    if last_exc: raise last_exc
    audio, engine, used_voice, client = _synthesize_ssml(ssml, POLLY_VOICE)
    marks = _visemes(clean, engine, used_voice, client)
    return base64.b64encode(audio).decode("ascii"), marks

# ---- API --------------------------------------------------------------------
@app.post("/api/chat")
def chat(payload: ChatIn):
    txt = payload.text.strip()
    sid = (payload.session_id or "local").strip()
    if not txt:
        raise HTTPException(400, "Empty text")
    try:
        if not _chat_gate.acquire(timeout=10):
            raise HTTPException(429, "Chat busy, try again shortly")
        try:
            q = txt.lower()
            from datetime import datetime
            if "date" in q and "update" not in q:
                reply = datetime.now().strftime("Today is %B %d, %Y.")
            elif "time" in q:
                reply = datetime.now().strftime("It's %I:%M %p.")
            elif q in {"what's your name","whats your name","your name?","who are you"}:
                reply = "Rem."
            else:
                reply = bedrock_reply(_compose_system(PERSONA_BLESSED_BOY, payload.style), sid, txt, payload.style)

            add_turn(sid, "user", txt)
            add_turn(sid, "assistant", reply)
            return {"reply": reply}
        finally:
            try: _chat_gate.release()
            except Exception: pass
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        raise HTTPException(500, f"Bedrock error: {code}")
    except Exception as e:
        raise HTTPException(500, f"Chat failure: {e.__class__.__name__}")

@app.post("/api/chat_stream")
def chat_stream(payload: ChatStreamIn):
    txt = payload.text.strip()
    sid = (payload.session_id or "local").strip()
    if not txt:
        raise HTTPException(400, "Empty text")

    messages = get_msgs(sid) + [{"role": "user", "content": [{"type": "text", "text": txt}]}]
    system_prompt = _compose_system(PERSONA_BLESSED_BOY, payload.style)

    def gen():
        try:
            buff = []
            acquired = _chat_gate.acquire(timeout=10)
            if not acquired:
                yield (json.dumps({"error": "Chat busy, try again shortly"}) + "\n").encode("utf-8")
                return
            try:
                for token in _stream_bedrock_text(BEDROCK_MODEL, system_prompt, messages, payload.style):
                    token = token.replace("\n", " ")
                    buff.append(token)
                    yield (json.dumps({"delta": token}) + "\n").encode("utf-8")
            finally:
                try: _chat_gate.release()
                except Exception: pass

            final = clamp_sentences(enforce_identity("".join(buff)) or "I'm here.")
            add_turn(sid, "user", txt)
            add_turn(sid, "assistant", final)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "ClientError")
            if code in {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"}:
                # Fallback: get a full reply non-streaming and send once
                try:
                    reply = bedrock_reply(_compose_system(PERSONA_BLESSED_BOY, payload.style), sid, txt, payload.style)
                    yield (json.dumps({"delta": reply}) + "\n").encode("utf-8")
                except Exception:
                    yield (json.dumps({"error": f"Bedrock error: {code}"}) + "\n").encode("utf-8")
            else:
                yield (json.dumps({"error": f"Bedrock error: {code}"}) + "\n").encode("utf-8")
        except Exception as e:
            yield (json.dumps({"error": f"Stream failure: {e.__class__.__name__}"}) + "\n").encode("utf-8")

    return StreamingResponse(gen(), media_type="application/jsonl")

_TTS_MAX_CONCURRENCY = int(os.getenv("TTS_MAX_CONCURRENCY", "3"))
_tts_gate = threading.Semaphore(_TTS_MAX_CONCURRENCY)

# Simple in-memory cache: text -> (expires_epoch, audio_b64, marks)
_TTS_TTL_SECONDS = int(os.getenv("TTS_CACHE_TTL", "900"))  # 15 minutes
_tts_cache: Dict[str, Tuple[float, str, list]] = {}
_tts_cache_lock = threading.Lock()

def _tts_cache_get(key: str):
    now = time.time()
    with _tts_cache_lock:
        rec = _tts_cache.get(key)
        if not rec:
            return None
        exp, audio_b64, marks = rec
        if exp < now:
            _tts_cache.pop(key, None)
            return None
        return audio_b64, marks

def _tts_cache_put(key: str, audio_b64: str, marks: list):
    with _tts_cache_lock:
        _tts_cache[key] = (time.time() + _TTS_TTL_SECONDS, audio_b64, marks)

def _retry_sleep(attempt: int):
    base = 0.25 * (2 ** attempt)
    time.sleep(base + random.random() * 0.2)

@app.post("/api/tts")
def tts(payload: TTSIn):
    txt = payload.text.strip()
    if not txt:
        raise HTTPException(400, "Empty text")
    try:
        # Cache by normalized text, including language and mode to avoid cross-voice collisions
        norm_txt = re.sub(r"\s+", " ", txt).strip().lower()
        lang_key = (payload.lang or "").strip().lower()
        mode_key = (payload.mode or "").strip().lower()
        key = f"{lang_key}|{mode_key}|{norm_txt}"
        cached = _tts_cache_get(key)
        if cached:
            audio_b64, marks = cached
            return {"audio_b64": audio_b64, "marks": marks}

        acquired = _tts_gate.acquire(timeout=10)
        if not acquired:
            raise HTTPException(429, "TTS busy, try again shortly")
        try:
            # Retry Polly on throttling / transient failures
            last_err = None
            for attempt in range(3):
                try:
                    audio_b64, marks = polly_tts_with_visemes(txt, payload.lang, payload.mode)
                    _tts_cache_put(key, audio_b64, marks)
                    break
                except ClientError as e:
                    code = e.response.get("Error", {}).get("Code", "ClientError")
                    if code in {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"}:
                        last_err = e; _retry_sleep(attempt); continue
                    raise
                except Exception as e:
                    last_err = e; _retry_sleep(attempt)
            else:
                raise last_err or RuntimeError("TTS retries exhausted")
        finally:
            try:
                _tts_gate.release()
            except Exception:
                pass
        return {"audio_b64": audio_b64, "marks": marks}
    except ClientError as e:
        err = e.response.get("Error", {})
        code = err.get("Code", "ClientError")
        msg = err.get("Message", "")
        logger.error("Polly error %s: %s", code, msg)
        raise HTTPException(500, f"Polly error: {code} - {msg}")
    except Exception as e:
        raise HTTPException(500, f"TTS failure: {e.__class__.__name__}")
@app.post("/api/sing")
def sing(payload: SingIn):
    txt = (payload.text or "").strip()
    if not txt:
        raise HTTPException(400, "Provide lyrics to sing.")
    # This feature uses user-provided lyrics. We do not fetch or provide copyrighted lyrics.
    try:
        # Cache key includes a 'sing:' prefix
        norm_txt = re.sub(r"\s+", " ", txt).strip().lower()
        lang_key = (payload.lang or "").strip().lower()
        mode_key = (payload.mode or "").strip().lower()
        key = f"sing:{lang_key}|{mode_key}|{norm_txt}"
        cached = _tts_cache_get(key)
        if cached:
            audio_b64, marks = cached
            return {"audio_b64": audio_b64, "marks": marks}

        acquired = _tts_gate.acquire(timeout=10)
        if not acquired:
            raise HTTPException(429, "TTS busy, try again shortly")
        try:
            last_err = None
            for attempt in range(3):
                try:
                    audio_b64, marks = polly_sing_with_visemes(txt, payload.lang, payload.mode or 'auto')
                    _tts_cache_put(key, audio_b64, marks)
                    break
                except ClientError as e:
                    code = e.response.get("Error", {}).get("Code", "ClientError")
                    if code in {"ThrottlingException","TooManyRequestsException","ServiceUnavailableException"}:
                        last_err = e; _retry_sleep(attempt); continue
                    raise
                except Exception as e:
                    last_err = e; _retry_sleep(attempt)
            else:
                raise last_err or RuntimeError("SING retries exhausted")
        finally:
            try: _tts_gate.release()
            except Exception: pass
        return {"audio_b64": audio_b64, "marks": marks}
    except ClientError as e:
        err = e.response.get("Error", {})
        code = err.get("Code", "ClientError")
        msg = err.get("Message", "")
        logger.error("Polly error %s: %s", code, msg)
        raise HTTPException(500, f"Polly error: {code} - {msg}")
    except Exception as e:
        raise HTTPException(500, f"Sing failure: {e.__class__.__name__}")

@app.get("/api/polly/voices")
def list_voices():
    try:
        voices = _get_voices()
        out = [
            {
                "Id": v.get("Id"),
                "LanguageCode": v.get("LanguageCode"),
                "LanguageName": v.get("LanguageName"),
                "Gender": v.get("Gender"),
                "SupportedEngines": v.get("SupportedEngines", []),
            }
            for v in voices
        ]
        return {"region": POLLY_REGION, "preferred": POLLY_VOICE, "voices": out}
    except Exception as e:
        raise HTTPException(500, f"List voices failed: {e.__class__.__name__}")
