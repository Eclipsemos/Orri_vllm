import os

# from nanovllm import LLM, SamplingParams 
from nanovllm import LLM

from transformers import AutoTokenizer

def main():
    path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--QWen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)

    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)



    print(path)


if __name__ == "__main__":
    main()