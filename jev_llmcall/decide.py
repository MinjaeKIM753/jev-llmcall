from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


_EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
_QUALITY_RANK = {"weak": 0, "adequate": 1, "strong": 2}


@dataclass(frozen=True)
class Judgment:
    task: str
    task_confidence: float
    difficulty: float
    stakes: float
    output_size: str


@dataclass(frozen=True)
class Choice:
    model_id: str
    reasoning_effort: str | None
    reason: str
    estimated_cost_usd: float | None
    sticky: bool
    skipped: tuple[str, ...]


def output_band(score: float) -> str:
    if score <= 0.5:
        return "short"
    if score <= 1.5:
        return "medium"
    return "long"


def build_questions(tasks: dict) -> dict:
    questions = {
        "task": {
            "type": "choice",
            "instructions": tasks["task"]["instructions"],
            "criteria": dict(tasks["task"]["criteria"]),
        },
        "difficulty": {
            "type": "score",
            "instructions": tasks["difficulty"]["instructions"],
            "criteria": list(tasks["difficulty"]["criteria"]),
        },
        "stakes": {
            "type": "noul",
            "instructions": tasks["stakes"]["instructions"],
        },
        "output_size": {
            "type": "score",
            "instructions": tasks["output_size"]["instructions"],
            "criteria": list(tasks["output_size"]["criteria"]),
        },
    }
    return questions


def parse_answers(answers: dict) -> Judgment:
    task = answers.get("task") or {}
    difficulty = answers.get("difficulty") or {}
    stakes = answers.get("stakes") or {}
    output = answers.get("output_size") or {}
    label = task.get("choice")
    if not isinstance(label, str) or not label:
        raise ValueError("Jev did not return a task label.")
    return Judgment(
        task=label,
        task_confidence=float(task.get("confidence") or 0.0),
        difficulty=float(difficulty.get("score") or 0.0),
        stakes=float(stakes.get("noul") or 0.0),
        output_size=output_band(float(output.get("score") or 0.0)),
    )


def state_text(prompt: str, limit: int) -> str:
    text = (prompt or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit]


def _cards_by_id(catalogue: dict) -> dict[str, dict]:
    cards = {}
    for card in catalogue.get("models") or []:
        if isinstance(card, dict) and card.get("id"):
            cards[str(card["id"])] = card
    return cards


def _tokens(input_tokens: int | None, prompt: str, output_name: str, policy: dict) -> tuple[int, int]:
    incoming = input_tokens if input_tokens is not None else max(1, len(prompt) // 4)
    guesses = policy["output_token_guess"]
    return int(incoming), int(guesses[output_name])


def priced_card(card: dict, today: _dt.date | None = None) -> dict:
    """The card with its ``scheduled`` price steps applied for ``today``.

    A step is ``{from: YYYY-MM-DD, price_in, price_out, ...}``. The latest
    step whose date has arrived overrides the base fields it names.
    """
    steps = card.get("scheduled") or []
    if not steps:
        return card
    day = (today or _dt.date.today()).isoformat()
    out = dict(card)
    for step in sorted((s for s in steps if isinstance(s, dict)), key=lambda s: str(s.get("from"))):
        if str(step.get("from")) <= day:
            out.update({k: v for k, v in step.items() if k != "from"})
    return out


def _list_cost(card: dict, incoming: int, outgoing: int) -> float | None:
    card = priced_card(card)
    price_in = card.get("price_in")
    price_out = card.get("price_out")
    if price_in is None or price_out is None:
        return None
    threshold = card.get("long_context_after")
    if threshold is not None and incoming > int(threshold):
        price_in = card.get("long_price_in", price_in)
        price_out = card.get("long_price_out", price_out)
    return (float(price_in) * incoming + float(price_out) * outgoing) / 1_000_000


def _multiplier(card: dict, task: str | None, policy: dict) -> float:
    if task is None:
        return 1.0
    quality = (card.get("quality") or {}).get(task)
    table = policy["quality_multiplier"]
    if quality not in table:
        return 1.0
    return float(table[quality])


def _eligible(
    card: dict,
    *,
    task: str | None,
    drop_economy: bool,
    needs_tools: bool,
    needs_image: bool,
    incoming: int,
    outgoing: int,
) -> bool:
    if _list_cost(card, incoming, outgoing) is None:
        return False
    context = int(card.get("context") or 0)
    if incoming + outgoing > context:
        return False
    if needs_tools and not card.get("tools"):
        return False
    if needs_image and not card.get("image"):
        return False
    if task is not None:
        quality = (card.get("quality") or {}).get(task)
        if quality in (None, "none"):
            return False
    if drop_economy and card.get("tier") == "economy":
        return False
    return True


def _drop_economy(judgment: Judgment | None, policy: dict) -> bool:
    if judgment is None:
        return True
    rules = policy["exclude_economy"]
    if judgment.difficulty >= float(rules["min_difficulty"]):
        return True
    if judgment.stakes >= float(rules["min_stakes"]):
        return True
    if judgment.task_confidence < float(policy["limits"]["task_confidence_floor"]):
        return True
    return False


def _clamp_effort(effort: str, card: dict) -> str | None:
    declared = [str(item) for item in (card.get("efforts") or [])]
    if not declared:
        return None
    if effort in declared:
        return effort
    if effort not in _EFFORT_ORDER:
        return declared[-1]
    wanted = _EFFORT_ORDER.index(effort)
    lower = [item for item in declared if item in _EFFORT_ORDER and _EFFORT_ORDER.index(item) <= wanted]
    if lower:
        return lower[-1]
    return declared[0]


def reasoning_effort(judgment: Judgment | None, card: dict, policy: dict, fallback: str | None) -> str | None:
    if judgment is None:
        return _clamp_effort(fallback or "medium", card)
    chosen = "high"
    for band in policy["reasoning_from_difficulty"]:
        if "max_score" not in band:
            continue
        if judgment.difficulty <= float(band["max_score"]):
            chosen = str(band["effort"])
            break
    else:
        chosen = "high"
    top_at = float(policy["stakes_unlocks_top"])
    # The loop above stops at the last max_score band. A higher difficulty
    # stays on that band's effort unless stakes unlock the card's top step.
    last_cap = max(float(band["max_score"]) for band in policy["reasoning_from_difficulty"] if "max_score" in band)
    if judgment.difficulty > last_cap and judgment.stakes >= top_at:
        declared = [str(item) for item in (card.get("efforts") or [])]
        if declared:
            return declared[-1]
    return _clamp_effort(chosen, card)


def _pick(scored: list[tuple[dict, float, float]], policy: dict) -> dict:
    """scored items are (card, list_cost, adjusted_cost)."""
    best = min(item[2] for item in scored)
    window = float(policy["limits"]["tie_relative"])
    if best == 0:
        tied = [item for item in scored if item[2] == 0]
    else:
        tied = [item for item in scored if (item[2] - best) / best < window]

    def sort_key(item: tuple[dict, float, float]) -> tuple:
        card, _list_cost, adjusted = item
        quality_name = None
        # Multiplier is already in adjusted. Rank uses the task quality when present.
        rank = _QUALITY_RANK.get(getattr(card, "_picked_quality", ""), -1)
        return (adjusted, -rank, -int(card.get("context") or 0), str(card["id"]))

    # Attach quality rank via a side field without mutating the catalogue card.
    decorated = []
    for card, list_cost, adjusted in tied:
        quality_name = card.get("_picked_quality")
        rank = _QUALITY_RANK.get(quality_name or "", -1)
        decorated.append((adjusted, -rank, -int(card.get("context") or 0), str(card["id"]), card))
    decorated.sort()
    return decorated[0][4]


def choose(
    *,
    policy: dict,
    catalogue: dict,
    models: list[str],
    prompt: str,
    judgment: Judgment | None,
    input_tokens: int | None,
    needs_tools: bool,
    needs_image: bool,
    previous_model: str | None,
    previous_task: str | None,
    fallback: bool,
) -> Choice:
    cards = _cards_by_id(catalogue)
    requested = [str(model) for model in models]
    skipped = tuple(model for model in requested if model not in cards)
    output_name = "medium" if judgment is None else judgment.output_size
    incoming, outgoing = _tokens(input_tokens, prompt, output_name, policy)
    task = None if judgment is None else judgment.task
    drop_economy = True if fallback else _drop_economy(judgment, policy)

    def collect(exclude_economy: bool) -> list[tuple[dict, float, float]]:
        found = []
        for model_id in requested:
            card = cards.get(model_id)
            if card is None:
                continue
            if not _eligible(
                card,
                task=task,
                drop_economy=exclude_economy,
                needs_tools=needs_tools,
                needs_image=needs_image,
                incoming=incoming,
                outgoing=outgoing,
            ):
                continue
            amount = _list_cost(card, incoming, outgoing)
            if amount is None:
                continue
            quality_name = None if task is None else (card.get("quality") or {}).get(task)
            adjusted = amount * _multiplier(card, task, policy)
            stamped = dict(card)
            stamped["_picked_quality"] = quality_name
            found.append((stamped, amount, adjusted))
        return found

    pool = collect(drop_economy)
    if not pool and drop_economy and not fallback:
        pool = collect(False)
    if not pool:
        raise LookupError("No priced catalogue card in the allowlist can take this request.")

    reason = "lowest_cost"
    sticky = False
    chosen_card = None
    if (
        not fallback
        and previous_model
        and previous_task
        and judgment is not None
        and not policy.get("sticky_downgrade", False)
    ):
        ranks = policy["demand_rank"]
        new_rank = int(ranks.get(judgment.task, 0))
        old_rank = int(ranks.get(previous_task, 0))
        if new_rank <= old_rank:
            for card, _amount, _adjusted in pool:
                if card["id"] == previous_model:
                    chosen_card = card
                    reason = "sticky"
                    sticky = True
                    break
    if chosen_card is None:
        chosen_card = _pick(pool, policy)
        if fallback:
            reason = str(policy["fallback"]["reason"])

    adjusted_cost = next(item[2] for item in pool if item[0]["id"] == chosen_card["id"])
    effort = reasoning_effort(
        None if fallback else judgment,
        chosen_card,
        policy,
        None if not fallback else str(policy["fallback"]["reasoning_effort"]),
    )
    return Choice(
        model_id=str(chosen_card["id"]),
        reasoning_effort=effort,
        reason=reason,
        estimated_cost_usd=adjusted_cost,
        sticky=sticky,
        skipped=skipped,
    )
