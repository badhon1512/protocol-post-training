import torch


class InferencePipeline:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.device = next(model.parameters()).device

        # Left padding keeps generation aligned when prompts have different lengths.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.7,
        system_prompt: str | None = None,
    ) -> list[str]:
        conversations = []
        for prompt in prompts:
            conversation = []
            if system_prompt:
                conversation.append({"role": "system", "content": system_prompt})
            conversation.append({"role": "user", "content": prompt})
            conversations.append(conversation)
        texts = [
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in conversations
        ]
        inputs = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generation_options = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_options["temperature"] = temperature

        outputs = self.model.generate(**inputs, **generation_options)
        prompt_length = inputs["input_ids"].shape[1]

        return self.tokenizer.batch_decode(
            outputs[:, prompt_length:],
            skip_special_tokens=True,
        )
