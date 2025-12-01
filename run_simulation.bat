@echo off
chcp 65001 >nul
cd /d %~dp0
python -u run_rough_surface_simulation.py
pause

