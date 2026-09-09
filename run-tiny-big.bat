@echo off
setlocal enableextensions

if exist "C:\Users\andre\tools\apache-maven-3.9.9\bin\mvn.cmd" (
    set "M2_HOME=C:\Users\andre\tools\apache-maven-3.9.9"
    set "PATH=%M2_HOME%\bin;%PATH%"
)

echo [FastDL] Installing local FastDL dependency...
call mvn clean install -DskipTests -q
if errorlevel 1 (
    echo [FastDL] Root build failed.
    pause
    exit /b %errorlevel%
)

echo [FastDL] Starting TinyStories big-data transformer demo...
cd examples\TinyStoriesDemo
call mvn compile exec:java -Dexec.mainClass=fastdl.demo.TinyStoriesTransformerDemo -q
if errorlevel 1 (
    echo [FastDL] TinyStories big-data demo failed.
    pause
    exit /b %errorlevel%
)

cd ..\..
pause
