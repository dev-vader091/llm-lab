from llm_lab.inference.ollama_client import run_inference


def main() -> None:
    prompt = input("Prompt: ")
    run_inference(prompt)