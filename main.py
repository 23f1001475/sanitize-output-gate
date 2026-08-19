"""
LLM Output Handling Gate (OWASP LLM05)
POST /sanitize-output
Allowed hosts: cdn-aphsg5b.example, app-2b0ft0g.example
"""

import re
from urllib.parse import urlparse, unquote
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        {"safe": False, "reason": "INVALID_SCHEMA"},
        status_code=exc.status_code,
    )

# ── constants ──────────────────────────────────────────────────────────────────
VALID_CHANNELS = {"html", "markdown", "url", "sql", "shell"}
ALLOWED_HOSTS  = {"cdn-aphsg5b.example", "app-2b0ft0g.example"}
MAX_OUTPUT_LEN = 20_000

# Named HTML entities we must decode (spec lists exactly these five)
HTML_ENTITIES = {
    "&lt;":   "<",
    "&gt;":   ">",
    "&quot;": '"',
    "&apos;": "'",
    "&amp;":  "&",
}


# ── decoding helpers ───────────────────────────────────────────────────────────

def clean_string(s: str) -> str:
    """
    Remove characters that could be used to bypass detection:
    - Control characters (except tab, newline, carriage return which are valid whitespace)
    - Zero-width characters (\u200b, \u200c, \u200d, \ufeff, etc.)
    - Other format characters that don't visually render
    """
    result = []
    for c in s:
        code = ord(c)
        # Keep normal printable chars, tab, newline, carriage return
        if c >= ' ' or c in '\t\n\r':
            # Exclude zero-width and other invisible Unicode chars
            # Zero-width space (200B), zero-width non-joiner (200C), zero-width joiner (200D)
            # Zero-width no-break space/BOM (FEFF), Word joiner (2060), etc.
            if code not in (0x200B, 0x200C, 0x200D, 0xFEFF, 0x2060, 0x180E):
                # Also exclude other format characters (Cf category covers more)
                # But for performance, just check common bypass chars
                result.append(c)
    return ''.join(result)


def decode_once(s: str) -> str:
    """
    Decode in order (single pass only, per spec):
      1. percent-escapes  (%XX)
      2. HTML entities    (numeric &#NN; / &#xNN; and the five named ones)
      3. \\uXXXX escapes
    Return the resulting string (may equal the original if nothing changed).
    """
    # 1. percent-decode
    try:
        step1 = unquote(s, errors="replace")
    except Exception:
        step1 = s

    # 2. numeric HTML entities  &#NN;  &#xNN;
    def replace_numeric_entity(m: re.Match) -> str:
        raw = m.group(1)
        try:
            if raw.startswith(("x", "X")):
                cp = int(raw[1:], 16)   # strip leading x/X before hex parse
            else:
                cp = int(raw, 10)
            return chr(cp)
        except (ValueError, OverflowError):
            return m.group(0)

    step2 = re.sub(r"&#([xX][0-9a-fA-F]+|\d+);", replace_numeric_entity, step1)

    # named entities (do &amp; last to avoid double-decoding)
    # order matters: &amp; must come after the others
    for entity, char in [("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                          ("&apos;", "'"), ("&amp;", "&")]:
        step2 = step2.replace(entity, char)

    # 3. \uXXXX escapes
    def replace_unicode_escape(m: re.Match) -> str:
        try:
            return chr(int(m.group(1), 16))
        except (ValueError, OverflowError):
            return m.group(0)

    step3 = re.sub(r"\\u([0-9a-fA-F]{4})", replace_unicode_escape, step2)

    return step3


# ── URL extraction helpers ─────────────────────────────────────────────────────

# matches quoted src= and href= attributes only (per spec)
# Requires whitespace or < before attribute
_HTML_URL_RE = re.compile(
    r"""(?:[\s<])(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)')""",
    re.IGNORECASE,
)

_MARKDOWN_URL_RE = re.compile(r"""\]\(([^)]*)\)""")


def extract_html_urls(text: str) -> list[str]:
    urls = []
    for m in _HTML_URL_RE.finditer(text):
        # group 1: double-quoted, group 2: single-quoted
        url = m.group(1) or m.group(2)
        if url:
            urls.append(url)
    return urls


def extract_markdown_urls(text: str) -> list[str]:
    urls = []
    for m in _MARKDOWN_URL_RE.finditer(text):
        url = m.group(1)
        # Markdown allows titles: [text](url "title") or [text](url 'title')
        # Strip the title part if present
        url = url.strip()
        # Remove title: anything after space followed by quote
        if ' "' in url:
            url = url.split(' "')[0]
        elif " '" in url:
            url = url.split(" '")[0]
        if url:
            urls.append(url)
    return urls


# ── scheme / host checks ───────────────────────────────────────────────────────

# "optional whitespace before the colon"
_DANGEROUS_SCHEME_RE = re.compile(
    r"\b(?:javascript|data|vbscript)\s*:", re.IGNORECASE
)


def is_dangerous_scheme_text(text: str) -> bool:
    """True if the raw text contains javascript:, data:, or vbscript: (with optional ws before :)."""
    # Clean string to remove bypass characters
    cleaned = clean_string(text)
    return bool(_DANGEROUS_SCHEME_RE.search(cleaned))


def check_url(url: str) -> str | None:
    """
    Return a reason code if the URL is problematic, else None.
    Checks:
      - DANGEROUS_SCHEME  (scheme != http / https, or known bad keyword)
      - EXTERNAL_EXFIL    (hostname not in ALLOWED_HOSTS)
    Relative URLs (no scheme, no //) → None (safe).
    Protocol-relative //host/path    → treated as https://host/path.
    """
    url = url.strip()
    if not url:
        return None

    # protocol-relative
    if url.startswith("//"):
        url = "https:" + url

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    scheme = parsed.scheme.lower()

    # No scheme → relative reference → safe
    if not scheme:
        return None

    # Bad scheme
    if scheme not in ("http", "https"):
        return "DANGEROUS_SCHEME"

    # Hostname check — use parsed.hostname (strips port and lowercases)
    hostname = parsed.hostname or ""
    if hostname not in ALLOWED_HOSTS:
        return "EXTERNAL_EXFIL"

    return None


# ── channel rule implementations ──────────────────────────────────────────────

def check_html(text: str) -> str | None:
    # Clean string to remove bypass characters
    cleaned = clean_string(text)
    
    # 1. SCRIPT_TAG — opening <script>, <iframe>, <object>, <embed>
    if re.search(r"<(?:script|iframe|object|embed)[\s>\/]", cleaned, re.IGNORECASE):
        return "SCRIPT_TAG"

    # 2. EVENT_HANDLER — on…= attribute  (e.g. onload=, onclick=)
    if re.search(r"\bon\w+\s*=", cleaned, re.IGNORECASE):
        return "EVENT_HANDLER"

    # 3. DANGEROUS_SCHEME — raw text check (catches javascript: in href values etc.)
    if is_dangerous_scheme_text(cleaned):
        return "DANGEROUS_SCHEME"

    # 4. EXTERNAL_EXFIL — check extracted URLs
    for url in extract_html_urls(cleaned):
        reason = check_url(url)
        if reason == "DANGEROUS_SCHEME":
            return "DANGEROUS_SCHEME"
        if reason == "EXTERNAL_EXFIL":
            return "EXTERNAL_EXFIL"

    return None


def check_markdown(text: str) -> str | None:
    # Clean string to remove bypass characters
    cleaned = clean_string(text)
    
    # 1. DANGEROUS_SCHEME — raw text check
    if is_dangerous_scheme_text(cleaned):
        return "DANGEROUS_SCHEME"

    # 2. EXTERNAL_EXFIL — extracted URLs from ](…)
    for url in extract_markdown_urls(cleaned):
        reason = check_url(url)
        if reason == "DANGEROUS_SCHEME":
            return "DANGEROUS_SCHEME"
        if reason == "EXTERNAL_EXFIL":
            return "EXTERNAL_EXFIL"

    return None


def check_url_channel(text: str) -> str | None:
    # Clean string to remove bypass characters
    url = clean_string(text.strip())
    if not url:
        return None

    # raw text dangerous scheme check first
    if is_dangerous_scheme_text(url):
        return "DANGEROUS_SCHEME"

    reason = check_url(url)
    return reason  # None, DANGEROUS_SCHEME, or EXTERNAL_EXFIL


_SQL_METACHAR_RE = re.compile(
    r"""'|"|;|--|/\*|\bunion\b|or\s+1\s*=\s*1""",
    re.IGNORECASE,
)

_SHELL_METACHAR_RE = re.compile(
    r"[;&|`<>\n\r]|\$\(|\$\{",
)


def check_sql(text: str) -> str | None:
    # Clean string to remove bypass characters
    cleaned = clean_string(text)
    if _SQL_METACHAR_RE.search(cleaned):
        return "SQL_METACHAR"
    return None


def check_shell(text: str) -> str | None:
    # Clean string to remove bypass characters
    cleaned = clean_string(text)
    if _SHELL_METACHAR_RE.search(cleaned):
        return "SHELL_METACHAR"
    return None


CHANNEL_CHECKERS = {
    "html":     check_html,
    "markdown": check_markdown,
    "url":      check_url_channel,
    "sql":      check_sql,
    "shell":    check_shell,
}


# ── main gate logic ────────────────────────────────────────────────────────────

def sanitize(channel: str, output: str) -> dict:
    if channel not in VALID_CHANNELS:
        return {"safe": False, "reason": "INVALID_SCHEMA"}

    if not isinstance(output, str):
        return {"safe": False, "reason": "INVALID_SCHEMA"}

    if len(output) > MAX_OUTPUT_LEN:
        return {"safe": False, "reason": "INVALID_SCHEMA"}

    # Rule 2: ENCODED_PAYLOAD
    decoded = decode_once(output)
    if decoded != output:
        # Run all channel rules against the decoded string
        reason = CHANNEL_CHECKERS[channel](decoded)
        if reason:
            return {"safe": False, "reason": "ENCODED_PAYLOAD"}

    # Rules 3+: apply channel rules to the ORIGINAL output
    reason = CHANNEL_CHECKERS[channel](output)
    if reason:
        return {"safe": False, "reason": reason}

    return {"safe": True, "reason": "SAFE"}


# ── endpoint ───────────────────────────────────────────────────────────────────

@app.post("/sanitize-output")
@app.post("/sanitize-output/", include_in_schema=False)
async def sanitize_output_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"safe": False, "reason": "INVALID_SCHEMA"})

    # Rule 1: INVALID_SCHEMA
    if not isinstance(body, dict):
        return JSONResponse({"safe": False, "reason": "INVALID_SCHEMA"})

    channel = body.get("channel")
    output  = body.get("output")

    if channel not in VALID_CHANNELS:
        return JSONResponse({"safe": False, "reason": "INVALID_SCHEMA"})

    if not isinstance(output, str):
        return JSONResponse({"safe": False, "reason": "INVALID_SCHEMA"})

    if len(output) > MAX_OUTPUT_LEN:
        return JSONResponse({"safe": False, "reason": "INVALID_SCHEMA"})

    result = sanitize(channel, output)
    return JSONResponse(result)


@app.get("/")
async def root():
    return {"status": "ok", "service": "sanitize-output gate"}
