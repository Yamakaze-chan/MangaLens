import torch
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor
from typing import List, Union

class MangaTranslator:
    def __init__(self, model_path="jzhang533/PaddleOCR-VL-For-Manga", device=None):
        print(f"Loading PaddleOCR-VL from {model_path}...")
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=True
        )
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.read_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        ).eval()
        if hasattr(torch, 'compile'):
            try:
                print("Compiling model for faster inference...")
                self.read_model = torch.compile(self.read_model)
            except Exception as e:
                print(f"Compilation failed: {e}. Running in eager mode.")

        if self.read_model.generation_config.pad_token_id is None:
            self.read_model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        self.prompt_template = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}]}],
            tokenize=False, 
            add_generation_prompt=True
        )

    def _preprocess_images(self, img_list: List[Union[Image.Image, np.ndarray]]) -> List[Image.Image]:
        processed = []
        for img in img_list:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            processed.append(img)
        return processed

    def get_ocr_text_batch(self, img_list: List[Union[Image.Image, np.ndarray]], batch_size: int = 4) -> List[str]:
        """
        Xử lý batch tối ưu với phân đoạn (chunking) để tránh OOM.
        batch_size: Đề xuất 4-8 cho GPU 8GB-12GB, 1-2 cho CPU.
        """
        if not img_list:
            return []

        all_results = []
        for i in range(0, len(img_list), batch_size):
            batch_imgs = img_list[i : i + batch_size]
            processed_images = self._preprocess_images(batch_imgs)
            prompts = [self.prompt_template] * len(processed_images)
            inputs = self.processor(
                text=prompts, 
                images=processed_images, 
                return_tensors="pt", 
                padding=True
            ).to(self.read_model.device)

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.read_model.dtype)

            # Inference
            with torch.inference_mode():
                generated_ids = self.read_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.read_model.generation_config.pad_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            decoded_batch = self.processor.batch_decode(
                generated_ids[:, input_len:], 
                skip_special_tokens=True
            )
            
            all_results.extend([t.strip() for t in decoded_batch])
            
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return all_results

    def get_ocr_text(self, img: Union[Image.Image, np.ndarray]) -> str:
        return self.get_ocr_text_batch([img], batch_size=1)[0]

# --- Cách sử dụng ---
# translator = MangaTranslator()
# images = [Image.open("p1.jpg"), Image.open("p2.jpg"), Image.open("p3.jpg")]
# results = translator.get_ocr_text_batch(images, batch_size=4)
# for r in results:
#     print(f"OCR: {r}")