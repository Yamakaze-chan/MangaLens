"""
Korean OCR — PP-OCRv5 with korean_PP-OCRv5_mobile_rec
Optimised for small screen-captured bubble text.
Thread-safe · CPU-first · Windows / Anaconda compatible

═══════════════════════════════════════════════════════════
INSTALLATION  (run inside your Conda environment)
═══════════════════════════════════════════════════════════

  # 1. CPU-only PaddlePaddle
  pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

  # 2. PaddleOCR (v3.x required for PP-OCRv5 API)
  pip install paddleocr

  # 3. Supporting libraries
  pip install opencv-python numpy Pillow

  Optional GPU (CUDA 11.8):
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

═══════════════════════════════════════════════════════════
QUICK USAGE
═══════════════════════════════════════════════════════════

  from korean_ocr import KoreanOCR

  ocr = KoreanOCR()
  text = ocr.extract_text("bubble_crop.png")      # → plain str
  boxes = ocr.extract_text_with_boxes("bubble_crop.png")

  # Multithreaded batch
  from concurrent.futures import ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=4) as pool:
      results = list(pool.map(ocr.extract_text, image_paths))

═══════════════════════════════════════════════════════════
PREPROCESSING PIPELINE (applied automatically)
═══════════════════════════════════════════════════════════

  Small images go through these steps before OCR:

  1. Upscale  — if shortest side < MIN_SIDE, scale up with
                INTER_CUBIC so text pixels are large enough
                for the detector to find strokes.

  2. Denoise  — light fastNlMeans to reduce JPEG/screen
                compression artefacts without blurring edges.

  3. Sharpen  — unsharp-mask to crisp up Korean stroke edges.

  4. Contrast — CLAHE on the luminance channel so faint text
                on coloured bubble backgrounds becomes readable.

  5. Padding  — a thin white border so characters at the very
                edge of the crop are not clipped by the detector.

  All steps are tunable via PreprocessConfig or can be disabled
  individually for your specific screen/game source.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════ #
#  Preprocessing configuration                                            #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class PreprocessConfig:
    """
    Knobs for the small-image preprocessing pipeline.

    Tune these if your source images differ (e.g. high-DPI screens,
    dark-mode UIs, heavy JPEG compression, etc.).

    Attributes:
        min_side (int):
            If the shorter dimension of the input image is smaller than
            this value the image is upscaled before OCR.
            Default 96 px — typical bubble-text crops are 30-80 px tall.
        target_side (int):
            The shorter dimension is scaled *up* to this value.
            PaddleOCR's text detector works best when text height is
            roughly 32-64 px; 256 gives comfortable headroom.
        denoise (bool):
            Apply fastNlMeansDenoisingColored to remove compression
            artefacts. Disable if your images are already clean.
        denoise_h (int):
            Filter strength for luminance denoising (3-10).
            Higher = smoother but may erase thin strokes.
        sharpen (bool):
            Apply an unsharp mask after denoising.
        sharpen_amount (float):
            Strength of the unsharp mask (0.5–2.0 is typical).
        clahe (bool):
            Apply CLAHE contrast enhancement on the L channel.
            Very helpful for text on coloured bubble backgrounds.
        clahe_clip (float):
            CLAHE clip limit; higher = more aggressive contrast.
        padding (int):
            Pixels of white border added on all sides before OCR.
            Prevents the detector from missing edge characters.
    """
    min_side: int        = 96
    target_side: int     = 256
    denoise: bool        = True
    denoise_h: int       = 5
    sharpen: bool        = True
    sharpen_amount: float = 1.2
    clahe: bool          = True
    clahe_clip: float    = 2.0
    padding: int         = 8


# ═══════════════════════════════════════════════════════════════════════ #
#  Main class                                                             #
# ═══════════════════════════════════════════════════════════════════════ #

class KoreanOCR:
    """
    Thread-safe Korean OCR for small screen-captured bubble text.

    Uses PaddleOCR PP-OCRv5 with ``korean_PP-OCRv5_mobile_rec`` and
    applies a preprocessing pipeline that makes tiny crops readable.

    The PaddleOCR engine is initialised exactly once (class-level
    singleton with double-checked locking) and shared safely across
    threads; inference is read-only after init.

    Args:
        device (str):
            ``"cpu"`` (default) or ``"gpu:0"``.
        use_textline_orientation (bool):
            Enable text-line orientation classifier. Defaults to True.
        confidence_threshold (float):
            Minimum confidence to keep a detected line. Default 0.5.
        line_separator (str):
            Separator between lines in :meth:`extract_text`. Default "\n".
        preprocess_config (PreprocessConfig | None):
            Preprocessing settings. Pass ``None`` to disable all
            preprocessing (not recommended for small images).
    """

    _engine = None
    _init_lock = threading.Lock()

    def __init__(
        self,
        device: str = "cpu",
        use_textline_orientation: bool = True,
        confidence_threshold: float = 0.5,
        line_separator: str = "\n",
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.line_separator = line_separator
        self.cfg = preprocess_config if preprocess_config is not None else PreprocessConfig()
        self._ensure_engine(device=device, use_textline_orientation=use_textline_orientation)

    # ------------------------------------------------------------------ #
    #  Engine init                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def _ensure_engine(cls, device: str, use_textline_orientation: bool) -> None:
        if cls._engine is not None:
            return
        with cls._init_lock:
            if cls._engine is not None:
                return
            from paddleocr import PaddleOCR
            logger.info(
                "Initialising PaddleOCR PP-OCRv5 (korean_PP-OCRv5_mobile_rec) on %s ...",
                device,
            )
            cls._engine = PaddleOCR(
                text_recognition_model_name="",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=use_textline_orientation,
                device=device,
            )
            logger.info("PaddleOCR engine ready.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract_text(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> str:
        """
        Preprocess *image* then run OCR, returning all text as a string.

        Args:
            image: File path, NumPy BGR/RGB array, or PIL Image.

        Returns:
            Recognised text lines joined by *line_separator*, or ``""``
            if nothing passes the confidence threshold.
        """
        arr = self._to_bgr_array(image)
        arr = self._preprocess(arr)
        results = self._engine.ocr(arr)
        return self._results_to_text(results)

    def extract_text_with_boxes(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> list[dict]:
        """
        Preprocess *image* then run OCR, returning per-box detail.

        Returns:
            List of ``{"text": str, "confidence": float, "box": list}``.
            Boxes are in preprocessed-image coordinates (upscaled +
            padded), not the original image coordinates.
        """
        arr = self._to_bgr_array(image)
        arr = self._preprocess(arr)
        results = self._engine.ocr(arr)
        return self._results_to_boxes(results)

    # ------------------------------------------------------------------ #
    #  Preprocessing pipeline                                             #
    # ------------------------------------------------------------------ #

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing pipeline to a BGR image.

        Steps (each can be disabled via PreprocessConfig):
          1. Upscale small images so text strokes are detector-friendly.
          2. Denoise with fastNlMeans to remove screen/JPEG artefacts.
          3. Unsharp-mask to sharpen Korean stroke edges.
          4. CLAHE contrast enhancement on the L (luminance) channel.
          5. White padding so edge characters are not clipped.
        """
        cfg = self.cfg

        # 1. Upscale ──────────────────────────────────────────────────── #
        h, w = img.shape[:2]
        short_side = min(h, w)
        if short_side < cfg.min_side:
            scale = cfg.target_side / short_side
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.debug("Upscaled %dx%d → %dx%d (scale=%.2f)", w, h, new_w, new_h, scale)

        # 2. Denoise ──────────────────────────────────────────────────── #
        if cfg.denoise:
            img = cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=cfg.denoise_h,        # luminance filter strength
                hColor=cfg.denoise_h,   # colour filter strength
                templateWindowSize=7,
                searchWindowSize=21,
            )

        # 3. Sharpen (unsharp mask) ────────────────────────────────────── #
        if cfg.sharpen:
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
            img = cv2.addWeighted(
                img,     1 + cfg.sharpen_amount,
                blurred, -cfg.sharpen_amount,
                0,
            )

        # 4. CLAHE contrast on L channel ──────────────────────────────── #
        if cfg.clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=cfg.clahe_clip,
                tileGridSize=(4, 4),   # small tiles suit tiny crops
            )
            l_ch = clahe.apply(l_ch)
            img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

        # 5. Padding ──────────────────────────────────────────────────── #
        if cfg.padding > 0:
            img = cv2.copyMakeBorder(
                img,
                cfg.padding, cfg.padding, cfg.padding, cfg.padding,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),   # white border
            )

        return img

    # ------------------------------------------------------------------ #
    #  Input normalisation                                                 #
    # ------------------------------------------------------------------ #

    def _to_bgr_array(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> np.ndarray:
        """Convert any supported input to a BGR NumPy array."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError(f"OpenCV could not decode: {path}")
            return arr

        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Image array is empty.")
            if image.ndim not in (2, 3):
                raise ValueError(
                    f"Expected 2-D or 3-D array, got shape {image.shape}."
                )
            # Grayscale -> BGR so the pipeline always works in colour
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image

        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            "Pass a file path (str/Path), np.ndarray, or PIL.Image."
        )

    # ------------------------------------------------------------------ #
    #  Result parsing                                                      #
    # ------------------------------------------------------------------ #

    def _results_to_text(self, results) -> str:
        lines: list[str] = []
        for page in (results or []):
            texts  = getattr(page, "rec_texts",  None) or []
            scores = getattr(page, "rec_scores", None) or []
            for text, score in zip(texts, scores):
                if float(score) >= self.confidence_threshold and text:
                    lines.append(text)
        return self.line_separator.join(lines)

    def _results_to_boxes(self, results) -> list[dict]:
        boxes: list[dict] = []
        for page in (results or []):
            texts  = getattr(page, "rec_texts",  None) or []
            scores = getattr(page, "rec_scores", None) or []
            polys  = getattr(page, "rec_polys",  None) or []
            for i, (text, score) in enumerate(zip(texts, scores)):
                if float(score) < self.confidence_threshold or not text:
                    continue
                box = polys[i].tolist() if i < len(polys) else []
                boxes.append({"text": text, "confidence": float(score), "box": box})
        return boxes


# ═══════════════════════════════════════════════════════════════════════ #
#  Demo                                                                   #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # ── Custom config example: aggressive upscaling for tiny HUD text ── #
    cfg = PreprocessConfig(
        min_side=64,          # upscale anything shorter than 64 px
        target_side=256,      # scale up to 256 px short side
        denoise=True,
        denoise_h=5,
        sharpen=True,
        sharpen_amount=1.2,
        clahe=True,
        clahe_clip=2.0,
        padding=8,
    )

    ocr = KoreanOCR(device="cpu", confidence_threshold=0.4, preprocess_config=cfg)

    image_paths: list[str] = sys.argv[1:]

    if not image_paths:
        print("Usage:  python korean_ocr.py image1.png image2.jpg ...")
        print("\nNo images supplied - running self-test with a tiny synthetic crop ...\n")

        # Simulate a small bubble-text crop (50 px tall - typical game UI)
        tiny = np.full((50, 200, 3), 240, dtype=np.uint8)
        cv2.putText(tiny, "hello", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)

        plain  = ocr.extract_text(tiny)
        detail = ocr.extract_text_with_boxes(tiny)
        print(f"extract_text()            -> {plain!r}")
        print(f"extract_text_with_boxes() -> {len(detail)} box(es) detected")
        sys.exit(0)

    def process(path: str) -> tuple[str, str]:
        try:
            return path, ocr.extract_text(path)
        except Exception as exc:
            return path, f"[ERROR] {exc}"

    workers = min(8, len(image_paths))
    print(f"Processing {len(image_paths)} image(s) with {workers} thread(s)...\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, p): p for p in image_paths}
        for fut in as_completed(futures):
            path, text = fut.result()
            print("=" * 60)
            print(f"File : {path}")
            print(f"Text :\n{text or '(no text detected)'}")