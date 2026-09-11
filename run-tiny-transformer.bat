@echo off
setlocal enableextensions

if exist "C:\Users\andre\tools\apache-maven-3.9.9\bin\mvn.cmd" (
    set "M2_HOME=C:\Users\andre\tools\apache-maven-3.9.9"
    set "PATH=%M2_HOME%\bin;%PATH%"
)

echo [FastDL] TinyStories Stage 2 - transformer smoke run...
call mvn clean install -DskipTests -q
if errorlevel 1 (
    echo [FastDL] Root build failed.
    pause
    exit /b %errorlevel%
)

echo [FastDL] Starting TinyStories transformer demo...
cd examples\TinyStoriesDemo
call mvn -q compile
if errorlevel 1 (
    echo [FastDL] TinyStories transformer compile failed.
    pause
    exit /b %errorlevel%
)

java -cp "..\..\target\classes;target\classes" fastdl.demo.TinyStoriesTransformerDemo --mode medium
if errorlevel 1 (
    echo [FastDL] TinyStories transformer demo failed.
    pause
    exit /b %errorlevel%
)

cd ..\..
pause
