from transformers import AutoModelForCausalLM, AutoProcessor


class Qwen:
    DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B"

    def load_model(self, model_name: str = DEFAULT_MODEL_NAME):
        return AutoModelForCausalLM.from_pretrained(model_name)

    def load_processor(self, model_name: str = DEFAULT_MODEL_NAME):
        return AutoProcessor.from_pretrained(model_name)


class Gemma:
    DEFAULT_MODEL_NAME = "google/gemma-4-E4B-it"

    def load_model(self, model_name: str = DEFAULT_MODEL_NAME):
        return AutoModelForCausalLM.from_pretrained(model_name)

    def load_processor(self, model_name: str = DEFAULT_MODEL_NAME):
        return AutoProcessor.from_pretrained(model_name)
