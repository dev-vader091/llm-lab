import ollama


MODEL = "qwen3:8b"


def run_inference(prompt: str) -> None:
    stream = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True,
    )

    print("\nResponse:")

    final_chunk = None

    for chunk in stream:
        print(chunk.message.content, end="", flush=True)
        final_chunk = chunk

    print()

    if final_chunk is not None:
        print_stats(final_chunk)


def print_stats(response: ollama.ChatResponse) -> None:
    generation_seconds = response.eval_duration / 1_000_000_000
    tokens_per_second = response.eval_count / generation_seconds

    print("\n--- Inference Stats ---")
    print(f"Prompt tokens: {response.prompt_eval_count}")
    print(f"Generated tokens: {response.eval_count}")
    print(
        f"Prompt evaluation time: "
        f"{response.prompt_eval_duration / 1_000_000_000:.2f}s"
    )
    print(f"Generation time: {generation_seconds:.2f}s")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/s")


def main() -> None:
    prompt = input("Prompt: ")
    run_inference(prompt)