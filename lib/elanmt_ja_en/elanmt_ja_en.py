import os
import re
import unicodedata

from transformers import pipeline

class MitsuaTranslator:
    def __init__(self, model_path=None):
        self.model_path = model_path or "Mitsua/elan-mt-bt-ja-en"

        self.translator = pipeline(
            "translation",
            model=self.model_path,
            tokenizer=self.model_path,
            device=0  # GPU
        )

    def translate_ja2en(self, text: str) -> str:
        if not text.strip():
            return text

        text = re.sub(r'[\r\n]+', ' ', text)
        text = unicodedata.normalize('NFKC', text).strip()

        result = self.translator(text)

        translated = result[0]["translation_text"]

        return translated