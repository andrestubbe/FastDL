@echo off
echo [FastDL] Building root module...
call C:\Users\andre\tools\apache-maven-3.9.9\bin\mvn.cmd clean install -DskipTests -q -f pom.xml
if errorlevel 1 ( echo BUILD FAILED ^& pause ^& exit /b 1 )
echo [FastDL] Launching TinyStories Explorer...
call C:\Users\andre\tools\apache-maven-3.9.9\bin\mvn.cmd compile exec:java -Dexec.mainClass=fastdl.demo.TinyStoriesExploreDemo -f examples\TinyStoriesDemo\pom.xml
pause
