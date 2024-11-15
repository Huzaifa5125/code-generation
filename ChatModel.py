import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class ChatModel:
    def __init__(self, model="EmTpro01/CodeLlama-7b-finetuned-16bit"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=quantization_config,
            device_map="cpu",
            cache_dir="./models",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, use_fast=True, padding_side="left"
        )

        self.history = []
        self.history_length = 1

        # Adjusting system prompt format
        self.DEFAULT_SYSTEM_PROMPT = """\
You are a Python code analyst, generating Python code to complete the task below.


"""

    def generate(
        self, user_prompt, system_prompt=None, top_p=0.9, temperature=0.1, max_new_tokens=512
    ):
        # Use default system prompt if none provided
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT.format(user_prompt)

        # Build instruction format
        texts = [f"{system_prompt}"]
        for old_prompt, old_response in self.history:
            texts.append(f"\n### Instruction:\n{old_prompt}\n\n### Response:\n{old_response.strip()}")
        texts.append(f"\n### Instruction:\n{user_prompt}\n\n### Response:")
        prompt = "".join(texts)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to("cuda")

        output = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=50,
            temperature=temperature,
        )
        output = output[0].to("cpu")
        response = self.tokenizer.decode(output[inputs["input_ids"].shape[1] : -1])
        self.append_to_history(user_prompt, response)
        return response

    def append_to_history(self, user_prompt, response):
        self.history.append((user_prompt, response))
        if len(self.history) > self.history_length:
            self.history.pop(0)
