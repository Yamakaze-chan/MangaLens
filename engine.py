import os
import re
import torch
import unicodedata
from ultralytics import YOLO
from llama_cpp import Llama
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from lib.quickmt_zh_en.translator import Translator
from lib.mangatranslator.translator import MangaTranslator
from lib.manga_predictor.MangaLanguageDetector import MangaLanguageDetector
from lib.kr_ocr.kr_ocr import KoreanOCR
from lib.kr_en.KR_Translator import KoreanTranslator
from lib.elanmt_ja_en.elanmt_ja_en import MitsuaTranslator

try:
    from lib.manga_ocr import MangaOcr
except ImportError:
    from manga_ocr import MangaOcr

from utils import text_wrap


class TranslationEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tải cấu hình mô hình
        self.detect_model = YOLO(os.getenv('YOLO_weight'))
        self.languages_detector = MangaLanguageDetector(os.getenv('MANGA_LANGUAGE_DETECTION_MODEL'), device='cpu')
        self.jp_read_model = MangaOcr(os.getenv('MangaOCR_weight'))
        self.zh_read_model = MangaTranslator(model_path=os.getenv('ZH_READ_MODEL'))
        
        # self.ja_model_path = os.getenv('JA_TRANS_MODEL')
        # self.ja_llm = Llama(
        #     model_path=self.ja_model_path,
        #     n_ctx=2048,
        #     n_threads=1,
        #     verbose=False,
        #     n_batch=8,
        #     stop=["\n"],
        #     echo=False
        # )

        self.ja_en_model = MitsuaTranslator(model_path=os.getenv('JA_ELANMT_MODEL'))

        self.zh_en_model = Translator(
            os.getenv('ZH_TRANS_MODEL'), 
            device=self.device
        )
        self.kr_read_model = KoreanOCR()
        self.kr_en_model = KoreanTranslator(model_name=os.getenv('KR_TRANS_MODEL'))
        
        if os.getenv('en_vi_weight'):
            self.tokenizer_en2vi = AutoTokenizer.from_pretrained(os.getenv('en_vi_token'))
            self.model_en2vi = AutoModelForSeq2SeqLM.from_pretrained(os.getenv('en_vi_weight'))
        else: 
            self.tokenizer_en2vi = None
            self.model_en2vi = None

    # def translate_ja2en(self, text: str, target_lang: str = "en") -> str:
    #     if not text.strip():
    #         return text
        
    #     if target_lang.lower() == "en":
    #         system_prompt = "Translate to English"
    #     elif target_lang.lower() == "ja":
    #         system_prompt = "Translate to Japanese."
    #     else:
    #         system_prompt = "Translate accurately."
            
    #     clean_text = re.sub(r'[\r\n]+', ' ', text)
    #     clean_text = unicodedata.normalize('NFKC', clean_text).strip()
    #     self.ja_llm.reset()
        
    #     response = self.ja_llm.create_chat_completion(
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": clean_text}
    #         ],
    #         temperature=0.1,
    #         max_tokens=128,
    #         repeat_penalty=1.1
    #     )

    #     translated_text = response['choices'][0]['message']['content'].strip()
    #     print("original text:", clean_text, " -> translated text:", translated_text)
    #     return translated_text

    def translate_ja2en(self, text: str, target_lang: str = "en") -> str:
        return self.ja_en_model.translate_ja2en( text=text)

    def translate_zh2en(self, text: str, translator: any, beam_size: int = 5) -> str:
        if not text.strip():
            return text
        try:
            translated_text = translator(
                text, 
                beam_size=beam_size,
                patience=1,
                length_penalty=0.8,
                coverage_penalty=1.0,
                repetition_penalty=1.2
            )
            if isinstance(translated_text, list):
                translated_text = translated_text[0]
            
            translated_text = str(translated_text).strip()
            print(f"original text: {text} -> translated text: {translated_text}")
            return translated_text
        except Exception as e:
            print(f"Lỗi khi dịch QuickMT: {e}")
            return text

    def translate_kr2en(self, text: str) -> str:
        if not text or not text.strip():
            return text
        try:
            translated_text = self.kr_en_model.translate(text=text)
            translated_text = str(translated_text).strip()
            print(f"original text: {text} -> translated text: {translated_text}")
            return translated_text
        except Exception as e:
            print(f"Lỗi khi dịch KR->EN: {e}")
            return text

    def translate_en2vi(self, en_text: str) -> str:
        if self.model_en2vi is None or self.tokenizer_en2vi is None:
            return en_text
        input_ids = self.tokenizer_en2vi(en_text, return_tensors="pt").input_ids
        output_ids = self.model_en2vi.generate(
            input_ids,
            decoder_start_token_id=self.tokenizer_en2vi.lang_code_to_id["vi_VN"],
            num_return_sequences=1,
            num_beams=5,
            early_stopping=True
        )
        vi_text = self.tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
        return " ".join(vi_text)

    def draw_text(self, font, text, x, y, x_max, y_max):
        """Hàm dựng ảnh và vẽ văn bản tiếng Việt/Anh đã dịch vào khung."""
        if not text:
            return Image.new('RGB', (x_max - x, y_max - y), (255, 255, 255))

        frame_width = round((x_max - x) * 1.1)
        frame_height = round((y_max - y) * 1.1)

        padding_x = int(frame_width * 0.1)
        padding_y = int(frame_height * 0.1)

        usable_width = frame_width - (2 * padding_x)
        usable_height = frame_height - (2 * padding_y)

        current_font = font
        MIN_FONT_SIZE = 10

        while True:
            lines = text_wrap(text, current_font, usable_width)
            if not lines:
                break

            line_height = int(current_font.getbbox("Ay")[3])
            total_text_height = len(lines) * line_height

            if total_text_height <= usable_height:
                break

            current_size = None
            try:
                current_size = current_font.size
            except AttributeError:
                pass
            if current_size is None:
                try:
                    current_size = current_font.font.size
                except AttributeError:
                    break

            if current_size <= MIN_FONT_SIZE:
                break

            new_size = max(current_size - 1, MIN_FONT_SIZE)
            try:
                current_font = ImageFont.truetype(current_font.path, new_size)
            except (AttributeError, OSError):
                break

        lines = text_wrap(text, current_font, usable_width)
        if not lines:
            return Image.new('RGB', (1, 1), (255, 255, 255))

        max_line_width = max(current_font.getbbox(line)[2] for line in lines)

        if max_line_width > usable_width:
            usable_width = max_line_width
            padding_x = int(usable_width * 0.1)
            frame_width = usable_width + (2 * padding_x)

        line_height = int(current_font.getbbox("Ay")[3])
        total_text_height = len(lines) * line_height

        if total_text_height > usable_height:
            frame_height = total_text_height + (2 * padding_y)

        add_text_img = Image.new('RGB', (frame_width, frame_height), (255, 255, 255))
        draw = ImageDraw.Draw(add_text_img)

        start_y = max(padding_y, (frame_height - total_text_height) // 2)
        current_y = start_y

        for line in lines:
            draw.text((padding_x, current_y), line, font=current_font, fill=(0, 0, 0))
            current_y += line_height

        return add_text_img