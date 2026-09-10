@echo off
setlocal enableextensions

echo [FastDL] Stage 3 - TinyStories big-data run
echo [FastDL] This is the long-running dataset path, not the quick smoke test.

if exist "C:\Users\andre\tools\apache-maven-3.9.9\bin\mvn.cmd" (
    set "M2_HOME=C:\Users\andre\tools\apache-maven-3.9.9"
    set "PATH=%M2_HOME%\bin;%PATH%"
)

call mvn clean install -DskipTests -q
if errorlevel 1 (
    echo [FastDL] Root build failed.
    pause
    exit /b %errorlevel%
)

echo [FastDL] Starting TinyStories big-data transformer demo...
cd examples\TinyStoriesDemo
call mvn -q compile
if errorlevel 1 (
    echo [FastDL] TinyStories big-data compile failed.
    pause
    exit /b %errorlevel%
)

java -cp "..\..\target\classes;target\classes" fastdl.demo.TinyStoriesBigDemo
if errorlevel 1 (
    echo [FastDL] TinyStories big-data demo failed.
    pause
    exit /b %errorlevel%
)

cd ..\..
pause
