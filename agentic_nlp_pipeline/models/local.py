import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
from transformers import TextIteratorStreamer


class LocalModel:
    """Wrapper for locally running language models.

    Attributes:
        tokenizer: Model-specific tokenizer for encoding and decoding messages.
        model: A pre-trained language model loaded from huggingface.
    """

    def __init__(
        self, model_id: str, device: str = "auto", enable_thinking: bool = False
    ):
        """Load model and tokenizer.

        Args:
            model_id: The huggingface model-id.
            device: The device where the model is supposed to be loaded onto.
        """
        print(f"Loading model `{model_id}`")
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
        )
        self.model.eval()

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        existing = self.model.generation_config.eos_token_id
        if isinstance(existing, int):
            existing = [existing]
        elif existing is None:
            existing = []
        if im_end_id not in existing:
            existing.append(im_end_id)
        self.model.generation_config.eos_token_id = existing

        print(self.tokenizer.decode([self.model.generation_config.eos_token_id]))
        print(f"Loaded model onto `{next(self.model.parameters()).device}`")

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 200,
        tools: list[dict] | None = None,
    ) -> str:
        """Generate a model response.

        Args:
            messages: A list of messages, each with a `role` and a `content`.
            max_new_tokens: The maximum number of tokens to be generated.

        Returns:
            The model response as a string.
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Settings for thinking enabled as recommended for Qwen
        do_sample = self.enable_thinking
        temperature = 1.0 if self.enable_thinking else None
        top_p = 0.95 if self.enable_thinking else None
        top_k = 20 if self.enable_thinking else None
        repetition_penalty = 1.1 if self.enable_thinking else None

        # Putting together generation arguments
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)  # type: ignore
        thread.start()

        parts: list[str] = []
        for token_text in streamer:
            print(token_text, end="", flush=True)
            parts.append(token_text)
        print()

        thread.join()
        return "".join(parts).strip()


# Little test
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen3-0.6B"
    SYSTEM_PROMPT = "You are a precise, concise linguistics expert."

    model = LocalModel(MODEL_ID)

    turn = "Who are you?"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": turn},
    ]
    reply = model.generate(messages)

    print(f"\n👨‍🦲 User: {turn}")
    print(f"🤖 Bot : {reply}")
