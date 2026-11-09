"""python -m jev_llmcall --prompt "..." --models gpt-6.1-sol,grok-4.7"""

from __future__ import annotations

import argparse
import json
import sys

from .router import JevAuthError, NoCandidate, Router


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pick a model id with a TypeSafe Jev key.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--models", required=True, help="Comma-separated model ids from the catalogue.")
    parser.add_argument("--input-tokens", type=int, default=None)
    parser.add_argument("--needs-tools", action="store_true")
    parser.add_argument("--needs-image", action="store_true")
    parser.add_argument("--key", default=None, help="TypeSafe key. Defaults to TYPESAFE_API_KEY.")
    args = parser.parse_args(argv)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    try:
        decision = Router(args.key).select(
            args.prompt,
            models,
            input_tokens=args.input_tokens,
            needs_tools=args.needs_tools,
            needs_image=args.needs_image,
        )
    except (JevAuthError, NoCandidate) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(decision.as_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
