# <center>MangaLens</center>
<text><center><strong>Read Manga directly on your screen</strong></center></text>

This project is real-time application that can OCR your screen and detect text in page of your manga.

## Demo
<iframe src="https://drive.google.com/file/d/1FJm20oze3H05ZFhITgM3CqcUnPHi-F9G/preview" width="640" height="480" allow="autoplay"></iframe>
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
YOLO_weight = MangaLens/best1.pt

#MangaOCR weight
MangaOCR_weight = ""

#Translation pretrained models (OPUS-MT)
ja_en_weight = Helsinki-NLP/opus-mt-ja-en
ja_en_token = Helsinki-NLP/opus-mt-ja-en
zh_en_weight = Helsinki-NLP/opus-mt-zh-en
zh_en_token = Helsinki-NLP/opus-mt-zh-en
```
- Step 7: Run and enjoy
```
python app.py
```
## Models
If you see this project is useful, don't forget to give star to these projects
- [YOLO](https://github.com/ultralytics/ultralytics): I am using v12 (because It's new and I want to try it :smiley: )
- [manga-ocr](https://github.com/kha-white/manga-ocr): To read text from image
- [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT): offline and fast
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
## Convert to desktop app
You can convert to desktop app with **PyInstaller**   
- [Document](https://pyinstaller.org/en/stable/)   
- [Install](https://pypi.org/project/pyinstaller/)   
In case you want to work with GUI instead of command line, [Auto PY to EXE](https://pypi.org/project/pyinstaller/) is my suggestion.
