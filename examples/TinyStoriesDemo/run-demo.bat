@echo off
setlocal enableextensions

if exist "C:\Users\andre\tools\apache-maven-3.9.9\bin\mvn.cmd" (
    set "M2_HOME=C:\Users\andre\tools\apache-maven-3.9.9"
    set "PATH=%M2_HOME%\bin;%PATH%"
)

echo [FastDL] Installing local FastDL dependency...
call mvn clean install -DskipTests -q
if errorlevel 1 (
    echo [FastDL] Build failed.
    pause
    exit /b %errorlevel%
)

echo [FastDL] Starting TinyStories demo...
call mvn compile exec:java -Dexec.mainClass=fastdl.demo.TinyStoriesMiniDemo -q
if errorlevel 1 (
    echo [FastDL] Demo failed.
    pause
    exit /b %errorlevel%
)

pause
