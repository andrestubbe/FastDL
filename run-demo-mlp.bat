@echo off
setlocal
cd /d "%~dp0"

echo [FastDL] Compiling and starting MLPBoundaryDemo...
call mvn clean compile exec:java -Dexec.mainClass="fastdl.demo.MLPBoundaryDemo" -q
