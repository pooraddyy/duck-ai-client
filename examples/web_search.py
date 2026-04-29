"""Use the WebSearch tool from duck.ai to ground answers on current info.

WebSearch is opt-in per call. It is only available on a subset of models;
the flag is silently dropped for models that don't support it.
"""

from duck_ai import DuckChat, gpt5_mini, model_supports_web_search


def main() -> None:
    assert model_supports_web_search(gpt5_mini), (
        "gpt5_mini should support web search"
    )

    with DuckChat(model=gpt5_mini) as duck:
        answer = duck.ask(
            "What were the top tech news headlines this week?",
            web_search=True,
        )
        print(answer)


if __name__ == "__main__":
    main()
