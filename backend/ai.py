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

import json
import re
import urllib.request
import urllib.error

from backend.rules import validate_rules

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


def _post(url: str, headers: dict, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # Surface the provider's message but never echo the key.
        body = ""
        try:
            body = e.read().decode("utf-8")[:400]
        except Exception:
            pass
        if e.code in (401, 403):
            raise AIError("The provider rejected the API key (unauthorized).")
        raise AIError(f"Provider error {e.code}: {body or e.reason}")
    except urllib.error.URLError as e:
        raise AIError(f"Could not reach the provider: {e.reason}")
    except Exception as e:  # noqa: BLE001
        raise AIError(f"Unexpected error calling the provider: {e}")


def _call_model(provider: str, api_key: str, model: str, base_url: str, user_msg: str) -> str:
    """Return the raw text the model produced (expected to be JSON)."""
    provider = (provider or "openai").lower()

    if provider == "anthropic":
        out = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
            {"model": model, "max_tokens": 400, "system": SYSTEM_PROMPT,
             "messages": [{"role": "user", "content": user_msg}]},
        )
        return "".join(b.get("text", "") for b in out.get("content", []))

    if provider == "gemini":
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={api_key}")
        out = _post(
            url, {"content-type": "application/json"},
            {"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
             "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
             "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}},
        )
        cands = out.get("candidates", [])
        if not cands:
            raise AIError("The model returned no content.")
        return "".join(p.get("text", "") for p in cands[0]["content"]["parts"])

    # openai OR local/openai-compatible (Ollama, LM Studio, vLLM)
    if provider == "local":
        base = (base_url or "http://localhost:11434/v1").rstrip("/")
    else:
        base = (base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"content-type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    out = _post(
        f"{base}/chat/completions", headers,
        {"model": model, "temperature": 0.2,
         "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_msg}]},
    )
    return out["choices"][0]["message"]["content"]


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
