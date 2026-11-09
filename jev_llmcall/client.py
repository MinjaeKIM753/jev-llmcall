from __future__ import annotations

import json
import urllib.error
import urllib.request


class JevAuthError(Exception):
    """The TypeSafe key was missing or rejected. The router does not guess a model."""


class JevCallError(Exception):
    """The decision call failed or took too long. The router may use its fallback."""


class JevClient:
    """One POST to TypeSafe. The connection is not reused across processes."""

    url = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        if not api_key or not api_key.strip():
            raise JevAuthError("TYPESAFE_API_KEY is empty.")
        self._api_key = api_key.strip()
        self._model = model
        self._timeout_s = timeout_s

    def ask(self, state: str, questions: dict) -> dict:
        return self.ask_with_usage(state, questions)[0]

    def ask_with_usage(self, state: str, questions: dict) -> tuple[dict, dict]:
        """``(answers, usage)``. ``usage`` is TypeSafe's billed token count
        (``input_tokens``; output is not billed), or ``{}`` if absent."""
        body = json.dumps(
            {"model": self._model, "state": state, "questions": questions}
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code in (401, 403):
                raise JevAuthError(f"TypeSafe rejected the key ({exc.code}).") from exc
            raise JevCallError(f"TypeSafe HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise JevCallError(str(exc)) from exc
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise JevCallError("TypeSafe response had no answers.")
        usage = payload.get("usage")
        return answers, dict(usage) if isinstance(usage, dict) else {}
