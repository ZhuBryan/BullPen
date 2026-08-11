"""
ai.py — "describe a strategy in plain English" translator.

The user types something like "buy when it's oversold but still in an uptrend."
We ask a language model to turn that into Bullpen's SAFE rule language (the same
grammar in rules.py), then we VALIDATE the model's output with the real parser
before it can ever run. The model never produces code — only strings in a tiny
whitelisted grammar that rules.py checks. If the model returns something the
parser rejects, we send the error back to the model once and let it fix itself.

Design choices:
  * Standard library only (urllib) — no new dependencies, no SDKs.
  * The API key is used in memory for a single request and never stored, cached,
    or logged. We do not print it anywhere.
  * Four provider families: OpenAI, Anthropic, Google Gemini, and any
    OpenAI-compatible local server (Ollama, LM Studio, vLLM, ...).
"""

import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

from backend.rules import validate_rules

# Model names go into a URL path (Gemini) and a JSON body. Anything outside this
# set is rejected rather than escaped, because a control character in a URL makes
# http.client raise an error whose message contains the FULL url -- including the
# API key -- and that message would then be returned to the caller.
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:\-/]{1,80}$")

# The grammar we allow, handed to the model as its rulebook.
GRAMMAR = (
    "INDICATORS: CLOSE (today's close), SMA(n), EMA(n), RSI(n), MOM(n) where n is "
    "an integer number of days.\n"
    "OPERATORS: < > <= >= for comparisons; AND, OR to combine; parentheses allowed.\n"
    "A rule is a boolean expression. buy_rule true => go long. sell_rule true => go to cash.\n"
    "Examples: 'RSI(14) < 30 AND CLOSE > SMA(200)', 'SMA(20) > SMA(50)', 'MOM(60) > 0'."
)

SYSTEM_PROMPT = (
    "You convert a plain-English stock trading idea into two rules in a tiny, fixed grammar.\n\n"
    + GRAMMAR
    + "\n\nRespond with ONLY a JSON object, no prose, no code fences, shaped exactly like:\n"
    '{"buy_rule": "<expression>", "sell_rule": "<expression>", "explanation": "<one short sentence>"}\n\n'
    "Rules: use ONLY the indicators and operators above. Do not invent functions. "
    "Keep each rule realistic and simple. If the idea only describes when to buy, "
    "choose a sensible complementary sell rule (and vice-versa)."
)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-1.5-flash",
    "local": "llama3.1",
}

MAX_DESC = 600
TIMEOUT = 45


class AIError(Exception):
    """A user-facing problem (bad key, provider down, unparseable reply)."""


def _check_local_url(base_url: str) -> str:
    """Allow a custom endpoint ONLY when it points at this machine.

    `base_url` is user input. Without this check the server will happily POST to
    any address the caller names -- cloud metadata endpoints, internal admin
    ports, other hosts on the private network -- which is server-side request
    forgery. Resolving the hostname first is what stops a public name that
    resolves to 169.254.169.254 from sneaking through.
    """
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise AIError("Local server URL must be a plain http(s) address.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 80,
                                   proto=socket.IPPROTO_TCP)
    except OSError:
        raise AIError("Could not resolve that local server URL.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_loopback:
            raise AIError(
                "For safety the local model URL must point at this machine "
                "(localhost or 127.0.0.1)."
            )
    return base_url.rstrip("/")


def _post(url: str, headers: dict, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Deliberately do NOT echo the response body. It is attacker-controlled
        # when base_url is, which would turn this into a readable SSRF oracle.
        if e.code in (401, 403):
            raise AIError("The provider rejected the API key (unauthorized).")
        if e.code == 429:
            raise AIError("The provider is rate limiting this key. Try again shortly.")
        raise AIError(f"The provider returned an error (HTTP {e.code}).")
    except urllib.error.URLError:
        raise AIError("Could not reach the model provider.")
    except json.JSONDecodeError:
        raise AIError("The provider returned a response that was not JSON.")
    except AIError:
        raise
    except Exception:  # noqa: BLE001
        # Never interpolate the exception: for Gemini the key is in the URL and
        # urllib puts the URL in the message.
        raise AIError("Unexpected error calling the provider.")


def _call_model(provider: str, api_key: str, model: str, base_url: str, user_msg: str) -> str:
    """Return the raw text the model produced (expected to be JSON)."""
    provider = (provider or "openai").lower()
    if not _MODEL_RE.match(model or ""):
        raise AIError("That model name contains characters that are not allowed.")

    try:
        if provider == "anthropic":
            out = _post(
                "https://api.anthropic.com/v1/messages",
                {"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
                {"model": model, "max_tokens": 400, "system": SYSTEM_PROMPT,
                 "messages": [{"role": "user", "content": user_msg}]},
            )
            return "".join(b.get("text", "") for b in out.get("content", []) or [])

        if provider == "gemini":
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent"
                   f"?key={urllib.parse.quote(api_key, safe='')}")
            out = _post(
                url, {"content-type": "application/json"},
                {"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                 "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
                 "generationConfig": {"temperature": 0.2,
                                      "responseMimeType": "application/json"}},
            )
            cands = out.get("candidates") or []
            if not cands:
                raise AIError("The model returned no content.")
            parts = cands[0].get("content", {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts)

        # openai OR local/openai-compatible (Ollama, LM Studio, vLLM)
        headers = {"content-type": "application/json"}
        if provider == "local":
            base = _check_local_url(base_url) if base_url else "http://localhost:11434/v1"
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            # A custom base_url is NOT honoured for cloud providers: it would
            # send the user's API key to an arbitrary host of the caller's
            # choosing. Only the local provider may redirect the endpoint.
            base = "https://api.openai.com/v1"
            headers["Authorization"] = f"Bearer {api_key}"
        out = _post(
            f"{base}/chat/completions", headers,
            {"model": model, "temperature": 0.2,
             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}]},
        )
        choices = out.get("choices") or []
        if not choices:
            raise AIError("The model returned no content.")
        content = (choices[0].get("message") or {}).get("content")
        if not content:
            raise AIError("The model returned an empty reply.")
        return content
    except AIError:
        raise
    except (KeyError, IndexError, TypeError, AttributeError):
        # A malformed provider payload must not surface as an unhandled 500.
        raise AIError("The provider returned a response in an unexpected shape.")


def _extract_json(text: str) -> dict:
    """Pull the first {...} object out of the model's reply and parse it."""
    text = text.strip()
    if not text:
        raise AIError("The model returned an empty reply.")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise AIError("The model did not return JSON rules.")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        raise AIError("The model's reply was not valid JSON.")


def translate(description: str, provider: str = "openai", api_key: str = "",
              model: str = "", base_url: str = "") -> dict:
    """
    Turn a plain-English description into validated buy/sell rules.

    Returns: {"buy_rule", "sell_rule", "explanation"} on success.
    Raises AIError with a user-friendly message on any failure.
    """
    description = (description or "").strip()
    if not description:
        raise AIError("Please describe the strategy first.")
    if len(description) > MAX_DESC:
        description = description[:MAX_DESC]

    provider = (provider or "openai").lower()
    model = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    if provider != "local" and not api_key:
        raise AIError("An API key is required for this provider.")

    user_msg = f"Trading idea: {description}\n\nReturn the JSON rules now."

    # First attempt, then one self-correcting retry if the parser rejects it.
    last_err = ""
    for attempt in range(2):
        msg = user_msg if attempt == 0 else (
            f"{user_msg}\n\nYour previous rules were invalid: {last_err}\n"
            "Fix them and return corrected JSON."
        )
        raw = _call_model(provider, api_key, model, base_url, msg)
        obj = _extract_json(raw)
        buy = str(obj.get("buy_rule", "")).strip()
        sell = str(obj.get("sell_rule", "")).strip()
        explanation = str(obj.get("explanation", "")).strip()

        ok, err = validate_rules(buy, sell)
        if ok:
            return {"buy_rule": buy, "sell_rule": sell, "explanation": explanation}
        last_err = err

    raise AIError(f"The model could not produce valid rules ({last_err}).")
