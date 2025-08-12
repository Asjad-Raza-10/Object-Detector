@echo off
CALL "C:\ProgramData\miniconda3\Scripts\activate.bat" tf_latest
E:
cd "E:\Asjad Raza\Github\Object-Detector"
streamlit run webapp.py
pause
