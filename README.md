# Comic Translator

Comic translator is an application that can convert spanish or french comics into English or vice versa.
It uses Tesseract OCR([Github](https://github.com/tesseract-ocr/)) to detect text over image and GNMT([Paper](https://smerity.com/articles/2016/google_nmt_arch.html)) to translate text between languages.

## Installation

You need to install Python 3.x . 

Install virtual environment thorugh pip in python 3.x.
```bash
pip3 install virtualenv 
```
and create a viertual environment :
```bash
python3 -m venv /path/to/new/virtual/environment 
```

and run requirements.txt file which is present in root directory to install all dependencies:

```bash
pip install -r requirements.txt
```

## Running Application

You can find test images in images/WORK/{language folder}  folder. 

To run application, run :

```bash
python main.py
```
Application should be run in sequential manner, i.e run first part, the second and so on. Translation may not work if you miss any part.

Application will return output as output.png in root directory.
