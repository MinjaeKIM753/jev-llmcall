from __future__ import annotations

import os
from dataclasses import dataclass

from .bundle import load_bundle
from .client import JevAuthError, JevCallError, JevClient
from .decide import Choice, Judgment, build_questions, choose, parse_answers, state_text

__all__ = ["Decision", "JevAuthError", "NoCandidate", "Router"]


class NoCandidate(LookupError):
    """The allowlist had no priced card that can take this request."""


@dataclass(frozen=True)
class Decision:
    model_id: str
    reasoning_effort: str | None
    reason: str
    task: str | None
    difficulty: float | None
    stakes: float | None
    output_size: str | None
    estimated_cost_usd: float | None
    sticky: bool
    skipped: tuple[str, ...]
    jev_model: str
    usage: dict | None = None

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "reasoning_effort": self.reasoning_effort,
            "reason": self.reason,
            "task": self.task,
            "difficulty": self.difficulty,
            "stakes": self.stakes,
            "output_size": self.output_size,
            "estimated_cost_usd": self.estimated_cost_usd,
            "sticky": self.sticky,
            "skipped": list(self.skipped),
            "jev_model": self.jev_model,
            "usage": dict(self.usage) if self.usage else None,
        }


class Router:
    """Select a model the caller can already invoke.

    ``typesafe_api_key`` is sent only to TypeSafe. Provider keys stay in the
    caller's own adapters.
    """

    def __init__(self, typesafe_api_key: str | None = None) -> None:
        key = typesafe_api_key if typesafe_api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
        self._policy, self._tasks, self._catalogue = load_bundle()
        timeout_s = float(self._policy["limits"]["timeout_ms"]) / 1000.0
        self._client = JevClient(key, str(self._policy["jev_model"]), timeout_s)
        self._state_limit = int(self._policy["limits"]["max_state_chars"])

    def select(
        self,
        prompt: str,
        models: list[str],
        *,
        input_tokens: int | None = None,
        needs_tools: bool = False,
        needs_image: bool = False,
        previous_model: str | None = None,
        previous_task: str | None = None,
        previous_effort: str | None = None,
        continue_turn: bool = False,
    ) -> Decision:
        """Return the id and reasoning effort for one user message.

        Set ``continue_turn`` on tool-result follow-ups in the same turn.
        That path does not call Jev and keeps ``previous_model``.
        """
        if not models:
            raise NoCandidate("models is empty.")
        if continue_turn:
            if not previous_model:
                raise NoCandidate("continue_turn requires previous_model.")
            if previous_model not in models:
                raise NoCandidate(f"{previous_model} is not in models.")
            return Decision(
                model_id=previous_model,
                reasoning_effort=previous_effort,
                reason="continue_turn",
                task=previous_task,
                difficulty=None,
                stakes=None,
                output_size=None,
                estimated_cost_usd=None,
                sticky=True,
                skipped=(),
                jev_model=str(self._policy["jev_model"]),
            )

        judgment = None
        fallback = False
        usage: dict | None = None
        try:
            answers, usage = self._client.ask_with_usage(
                state_text(prompt, self._state_limit),
                build_questions(self._tasks),
            )
            judgment = parse_answers(answers)
        except JevAuthError:
            raise
        except (JevCallError, ValueError):
            fallback = True

        try:
            choice = choose(
                policy=self._policy,
                catalogue=self._catalogue,
                models=list(models),
                prompt=prompt,
                judgment=judgment,
                input_tokens=input_tokens,
                needs_tools=needs_tools,
                needs_image=needs_image,
                previous_model=previous_model,
                previous_task=previous_task,
                fallback=fallback,
            )
        except LookupError as exc:
            raise NoCandidate(str(exc)) from exc
        return _decision(choice, judgment, str(self._policy["jev_model"]), usage)

    @property
    def bundled_catalogue(self) -> dict:
        return self._catalogue

    @property
    def jev_model(self) -> str:
        return str(self._policy["jev_model"])

    def judge(self, prompt: str) -> tuple[Judgment | None, dict | None]:
        """Jev's reading of the prompt without picking a model: ``(judgment,
        usage)``, or ``(None, None)`` when the call fails. For callers that
        keep their own table but want Jev's task label."""
        try:
            answers, usage = self._client.ask_with_usage(
                state_text(prompt, self._state_limit),
                build_questions(self._tasks),
            )
            return parse_answers(answers), usage
        except JevAuthError:
            raise
        except (JevCallError, ValueError):
            return None, None

    def choose_for(
        self,
        judgment: Judgment,
        prompt: str,
        models: list[str],
        *,
        input_tokens: int | None = None,
        needs_tools: bool = False,
        needs_image: bool = False,
        usage: dict | None = None,
        catalogue: dict | None = None,
    ) -> Decision:
        """The :meth:`select` pick for a judgment from :meth:`judge`, without
        calling Jev again. ``catalogue`` (same shape as the bundled
        ``models.example.yaml``) replaces the bundled cards for this call, for
        callers that build cards from their own model list."""
        if not models:
            raise NoCandidate("models is empty.")
        try:
            choice = choose(
                policy=self._policy,
                catalogue=catalogue if catalogue is not None else self._catalogue,
                models=list(models),
                prompt=prompt,
                judgment=judgment,
                input_tokens=input_tokens,
                needs_tools=needs_tools,
                needs_image=needs_image,
                previous_model=None,
                previous_task=None,
                fallback=False,
            )
        except LookupError as exc:
            raise NoCandidate(str(exc)) from exc
        return _decision(choice, judgment, str(self._policy["jev_model"]), usage)


def _decision(choice: Choice, judgment, jev_model: str, usage: dict | None = None) -> Decision:
    return Decision(
        model_id=choice.model_id,
        reasoning_effort=choice.reasoning_effort,
        reason=choice.reason,
        task=None if judgment is None else judgment.task,
        difficulty=None if judgment is None else judgment.difficulty,
        stakes=None if judgment is None else judgment.stakes,
        output_size=None if judgment is None else judgment.output_size,
        estimated_cost_usd=choice.estimated_cost_usd,
        sticky=choice.sticky,
        skipped=choice.skipped,
        jev_model=jev_model,
        usage=usage,
    )
