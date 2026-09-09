@echo off
setlocal enableextensions

echo [FastDL] Legacy alias: Stage 1 - TinyStories quick preview
echo [FastDL] Delegating to run-tiny-quick.bat
call "%~dp0run-tiny-quick.bat"
exit /b %errorlevel%
