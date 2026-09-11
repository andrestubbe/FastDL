@echo off
setlocal enableextensions
pushd "%~dp0"

echo [FastDL] Stage 3 - TinyStories big-data run
echo [FastDL] Using dataset: data\TinyStories-train.txt
echo [FastDL] This is the long-running dataset path, not the quick smoke test.

if exist "C:\Users\andre\tools\apache-maven-3.9.9\bin\mvn.cmd" (
    set "M2_HOME=C:\Users\andre\tools\apache-maven-3.9.9"
    set "PATH=%M2_HOME%\bin;%PATH%"
)

echo [FastDL] Starting TinyStories big-data transformer demo...
cd /d "%~dp0examples\TinyStoriesDemo"
call mvn -q exec:java -Dexec.mainClass=fastdl.demo.TinyStoriesBigDemo
if errorlevel 1 (
    echo [FastDL] TinyStories big-data demo failed.
    pause
    exit /b %errorlevel%
)

popd
pause
