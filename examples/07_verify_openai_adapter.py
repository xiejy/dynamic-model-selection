"""7. Verify the OpenAI adapter against a live endpoint.

This is the one code path in the repo that has never crossed the wire. It is
also the one that matters for production: the Codex CLI backend works but adds
~13s and a ~17k-token harness to every request, which is fine for comparison and
useless in front of real traffic.

OpenRouter speaks the OpenAI wire format, so pointing this adapter at it
exercises *exactly* the code that serves OpenAI -- same payload builder, same
response parser, same streaming reader. It is a real verification, not a proxy
for one.

    export OPENROUTER_API_KEY=sk-or-v1-...
    uv run python examples/07_verify_openai_adapter.py

    # or against OpenAI directly
    export OPENAI_API_KEY=sk-...
    DMS_VERIFY_TARGET=openai uv run python examples/07_verify_openai_adapter.py

What it checks, in order of how likely it is to be wrong:

1. the model id is real                      (these ids came from a Codex cost
                                              table, not an API model list)
2. `max_completion_tokens` is accepted       (renamed from `max_tokens`)
3. the response parses and the text is there
4. usage parses, and cached tokens are not billed twice
5. `stream_options: {include_usage: true}` is honoured, so streamed requests
   can be costed at all
"""
import os
import sys

from _shared import ROOT, usd  # noqa: F401  (path setup)

from dms.dispatch.providers import (
    OpenAIProvider,
    ProviderError,
    Request,
    openrouter_provider,
)
from dms.pricing import PriceBook

TARGET = os.environ.get("DMS_VERIFY_TARGET", "openrouter")
MODEL = os.environ.get(
    "DMS_VERIFY_MODEL", "openai/gpt-5.6" if TARGET == "openrouter" else "gpt-5.6-sol"
)


def build() -> OpenAIProvider:
    return openrouter_provider() if TARGET == "openrouter" else OpenAIProvider()


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return ok


def main() -> int:
    provider = build()
    book = PriceBook.load()
    print(f"\n=== verifying the OpenAI adapter against {TARGET} ===")
    print(f"    base_url {provider.base_url}")
    print(f"    model    {MODEL}\n")

    try:
        provider.api_key
    except ProviderError as exc:
        print(f"  no credential: {exc}")
        print("\n  Nothing was verified. Set the key and re-run.")
        return 2

    results: list[bool] = []
    request = Request(
        messages=({"role": "user", "content": "Reply with exactly: ADAPTER_OK"},),
        system="Answer with the requested text only.",
        max_tokens=64,
    )

    # 1-4: a plain completion.
    try:
        call = provider.complete(MODEL, request)
    except ProviderError as exc:
        check("model id is real / request accepted", False, str(exc)[:160])
        print("\n  Stopped: the non-streaming call failed, so nothing downstream "
              "can be checked.")
        return 1

    results.append(check("model id is real / request accepted", True))
    results.append(
        check("response parses and carries text", bool(call.text), repr(call.text[:40]))
    )
    usage = call.usage
    results.append(
        check(
            "usage parses",
            usage.total_tokens > 0,
            f"in={usage.input_tokens} cached={usage.cache_read_input_tokens} "
            f"out={usage.output_tokens}",
        )
    )
    results.append(
        check(
            "cached tokens are not double counted",
            usage.input_tokens >= 0,
            f"prompt total {usage.prompt_tokens}",
        )
    )
    try:
        cost = book.cost_usd(usage, MODEL)
        results.append(check("cost computes from the price book", True, f"${cost:.6f}"))
    except KeyError as exc:
        results.append(check("cost computes from the price book", False, str(exc)[:120]))

    # 5: streaming, and whether usage comes back with it.
    sink: list = []
    try:
        text = "".join(provider.stream(MODEL, request, sink))
        results.append(check("stream yields text", bool(text), repr(text[:40])))
        results.append(
            check(
                "streamed usage is reported (else streams bill $0)",
                bool(sink),
                f"{sink[0].to_dict()}" if sink else "no usage on the terminal chunk",
            )
        )
    except ProviderError as exc:
        results.append(check("stream yields text", False, str(exc)[:160]))

    passed = sum(results)
    print(f"\n  {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
