# <center>MangaLens</center>
<text><center><strong>Read Manga directly on your screen</strong></center></text>

This project is real-time application that can OCR your screen and detect text in page of your manga.

## Demo


https://github.com/user-attachments/assets/3b693bc5-8470-48cb-8f22-1a31f5d85b84

## Key Combination Usage
- <code>`</code> (*Key Code: 192*) : detect and translate again
- `Shift` + <code>`</code> : Clear screen
- `Ctrl` + `Alt` : Quit
- `Shift` + `W`: Get specific area of your screen to translate

## How to use
Wait until your terminal show `OCR ready` then press <code>`</code> to start (Re-press if it not detect and show translation)

## Installation
You can install and run (and modify) with these steps:
- Step 1: Clone this repository
```
git clone https://github.com/Yamakaze-chan/MangaLens.git
```
- Step 2: Change working directory
```
cd MangaLens
```
- Step 3: Install require packages
```
pip install -r requirements.txt
```
- Step 4: Download YOLO weight
```
git clone https://huggingface.co/Yamakaze/MangaLens  
```
- Step 5: create `.env` file
```
cd . > .env
```
- Step 6: Open `.env` and add this
```
#Yolo weight
YOLO_weight = MangaLens/best2.pt

#MangaOCR weight
MangaOCR_weight = ""
MANGA_LANGUAGE_DETECTION_MODEL = MangaLens/manga_effnet_b0.onnx

#Translation pretrained models
JA_TRANS_MODEL = LiquidAI/LFM2-350M-ENJP-MT-GGUF
ZH_TRANS_MODEL = quickmt/quickmt-zh-en
ZH_READ_MODEL = jzhang533/PaddleOCR-VL-For-Manga
KR_TRANS_MODEL = Helsinki-NLP/opus-mt-ko-en
en_vi_weight = vinai/vinai-translate-en2vi-v2
en_vi_token = vinai/vinai-translate-en2vi-v2
```
- Step 7: Run and enjoy
```
python app.py
```
## Models
If you see this project is useful, don't forget to give star to these projects
- [YOLO](https://github.com/ultralytics/ultralytics): I am using v12 (because It's new and I want to try it :smiley: )
- [manga-ocr](https://github.com/kha-white/manga-ocr): To read text from image
- [PaddleOCR-VL-For-Manga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga): This model OCR Chinese is really good
- [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT): translate offline and fast
- [LFM2-350M-ENJP-MT](https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT): translate Japanese fast with the good result
- [QuickMT-ZH-EN](https://huggingface.co/quickmt/quickmt-zh-en): translate Chinese good and fast
- [EfficientNet](https://github.com/lukemelas/efficientnet-pytorch): Detect text language to pick translator
- **This project**: Why not? :smiley:
## Contact
For any inquiries, please feel free to contact me at nhatvipmason@gmail.com
## Citation
```
@article{tiedemann2023democratizing,
  title={Democratizing neural machine translation with {OPUS-MT}},
  author={Tiedemann, J{\"o}rg and Aulamo, Mikko and Bakshandaeva, Daria and Boggia, Michele and Gr{\"o}nroos, Stig-Arne and Nieminen, Tommi and Raganato\
, Alessandro and Scherrer, Yves and Vazquez, Raul and Virpioja, Sami},
  journal={Language Resources and Evaluation},
  number={58},
  pages={713--755},
  year={2023},
  publisher={Springer Nature},
  issn={1574-0218},
  doi={10.1007/s10579-023-09704-w}
}

@InProceedings{TiedemannThottingal:EAMT2020,
  author = {J{\"o}rg Tiedemann and Santhosh Thottingal},
  title = {{OPUS-MT} — {B}uilding open translation services for the {W}orld},
  booktitle = {Proceedings of the 22nd Annual Conferenec of the European Association for Machine Translation (EAMT)},
  year = {2020},
  address = {Lisbon, Portugal}
 }
```
```
@inproceedings{vinaitranslate,
title     = {{A Vietnamese-English Neural Machine Translation System}},
author    = {Thien Hai Nguyen and 
             Tuan-Duy H. Nguyen and 
             Duy Phung and 
             Duy Tran-Cong Nguyen and 
             Hieu Minh Tran and 
             Manh Luong and 
             Tin Duy Vo and 
             Hung Hai Bui and 
             Dinh Phung and 
             Dat Quoc Nguyen},
booktitle = {Proceedings of the 23rd Annual Conference of the International Speech Communication Association: Show and Tell (INTERSPEECH)},
year      = {2022}
}

```
## Convert to desktop app
You can convert to desktop app with **PyInstaller**   
- [Document](https://pyinstaller.org/en/stable/)   
- [Install](https://pypi.org/project/pyinstaller/)   
In case you want to work with GUI instead of command line, [Auto PY to EXE](https://pypi.org/project/pyinstaller/) is my suggestion.
