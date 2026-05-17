from transformers import pipeline
import torch


class KoreanTranslator:
    def __init__(
        self,
        model_name="Helsinki-NLP/opus-mt-ko-en",
        device=None
    ):
        """
        Korean -> English Translator
        """

        self.model_name = model_name
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.device = device
        self.pipe = pipeline(
            "translation",
            model=model_name,
            device=device
        )

    def translate(
        self,
        text,
        max_length=512,
        clean=True
    ):
        """
        Translate 1 string
        """

        if not text or not text.strip():
            return ""

        try:
            translation = self.pipe(
                text,
                max_length=max_length
            )

            result = translation[0]["translation_text"]

            if clean:
                result = result.strip()

            return result

        except Exception as e:
            print(f"KR Translate Error: {e}")
            return text

    def translate_batch(
        self,
        texts,
        max_length=512
    ):
        """
        Translate list[str]
        """

        if not texts:
            return []

        try:
            translations = self.pipe(
                texts,
                max_length=max_length
            )

            results = [
                x["translation_text"].strip()
                for x in translations
            ]

            return results

        except Exception as e:
            print(f"KR Batch Translate Error: {e}")
            return texts


# =========================
# Example
# =========================

if __name__ == "__main__":

    translator = KoreanTranslator(
        model_name=r""
    )

    # Single
    text = "안녕하세요"

    result = translator.translate(text)

    print("Single:")
    print(result)

    # Batch
    texts = [
        "안녕하세요",
        "오늘 날씨가 좋네요",
        "이 만화 정말 재미있어요"
    ]

    results = translator.translate_batch(texts)

    print("\nBatch:")

    for r in results:
        print(r)