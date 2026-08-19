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

    @staticmethod
    def _text_content_messages(messages: list[dict]) -> list[dict]:
        converted = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            converted.append({"role": message["role"], "content": content})
        return converted

    def _apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool,
    ) -> str:
        template_owner = (
            self.processor
            if hasattr(self.processor, "apply_chat_template")
            else self.tokenizer
        )
        try:
            return template_owner.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except (TypeError, ValueError, KeyError):
            return template_owner.apply_chat_template(
                self._text_content_messages(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        prompt_style: str = "chat",
    ) -> list[str]:
        if prompt_style == "chat":
            conversations = []
            for prompt in prompts:
                conversation = []
                if system_prompt:
                    conversation.append({"role": "system", "content": system_prompt})
                conversation.append({"role": "user", "content": prompt})
                conversations.append(conversation)
            texts = [
                self._apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                )
                for conversation in conversations
            ]
        elif prompt_style == "sft_text":
            texts = [
                (
                    f"System: {system_prompt}\n"
                    if system_prompt
                    else ""
                )
                + f"User: {prompt}\nAssistant:"
                for prompt in prompts
            ]
        else:
            raise ValueError("prompt_style must be 'sft_text' or 'chat'")
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

        responses = self.tokenizer.batch_decode(
            outputs[:, prompt_length:],
            skip_special_tokens=True,
        )
        return [response.strip() for response in responses]
