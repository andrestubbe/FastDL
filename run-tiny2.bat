@echo off
setlocal enableextensions

echo [FastDL] Legacy alias: Stage 2 - TinyStories transformer smoke run
echo [FastDL] Delegating to run-tiny-transformer.bat
call "%~dp0run-tiny-transformer.bat"
exit /b %errorlevel%
