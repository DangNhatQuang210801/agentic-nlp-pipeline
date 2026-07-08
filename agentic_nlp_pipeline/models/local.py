from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalModel:
    """Wrapper for locally running language models.

    Attributes:
        tokenizer: Model-specific tokenizer for encoding and decoding messages.
        model: A pre-trained language model loaded from huggingface.
    """

    def __init__(self, model_id: str, device: str = "auto"):
        """Load model and tokenizer.

        Args:
            model_id: The huggingface model-id.
            device: The device where the model is supposed to be loaded onto.
        """
        print(f"Loading model `{model_id}`")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
        )
        self.model.eval()
        print(f"Loaded model onto `{next(self.model.parameters()).device}`")

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 200,
        tools: Optional[list[dict]] = None,
    ) -> str:
        """Generate a model response.

        Args:
            messages: A list of messages, each with a `role` and a `content`.
            max_new_tokens: The maximum number of tokens to be generated.

        Returns:
            The model response as a string.
        """
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        out_text = self.tokenizer.decode(
            out[0][prompt_len:],
            skip_special_tokens=True,
        )
        return out_text.strip()


# Little test
if __name__ == "__main__":
    print("=" * 64)

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
