@echo off
title 19 Sistem Kuantum Bilgisayari & Studio App
echo [1/2] Gerekli Python kütüphaneleri kontrol ediliyor...
pip install -r requirements.txt
echo.
echo [2/2] Stüdyo Arayüzü Başlatılıyor...
python studio_app.py
pause
