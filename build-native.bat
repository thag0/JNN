@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "JAVA_HOME="

for %%K in (
	"HKLM\SOFTWARE\JavaSoft\JDK"
	"HKLM\SOFTWARE\WOW6432Node\JavaSoft\JDK"
) do (
	for /f "tokens=2,*" %%A in ('reg query %%K /v CurrentVersion 2^>nul') do (
		set "JDK_VERSION=%%B"
		for /f "tokens=2,*" %%C in ('reg query %%K\!JDK_VERSION! /v JavaHome 2^>nul') do (
			set "JAVA_HOME=%%D"
			goto :jdk_achado
		)
	)
)

:jdk_achado

if not defined JAVA_HOME (
  echo ERRO: Nenhum JDK encontrado no sistema.
  exit /b 1
)

set "JAVA_HOME=%JAVA_HOME:"=%"

if not exist "%JAVA_HOME%\include\jni.h" (
	echo ERRO: JDK encontrado, mas jni.h nao existe:
	echo %JAVA_HOME%
	exit /b 1
)

@rem build jni

set OUT_DIR=bin\nativo\cpu\win64
set DLL=jnn_native.dll

set SRC=^
 jnn\nativo\jni\jnn_jni.c ^
 jnn\nativo\dispatch\dispatcher.c ^
 jnn\nativo\cpu\matmul.c ^
 jnn\nativo\cpu\conv2d.c ^
 jnn\nativo\cpu\maxpool.c

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

gcc ^
 -O3 ^
 -shared ^
 -march=x86-64 ^
 -ffast-math ^
 -mfma ^
 -fopenmp ^
 -I"%JAVA_HOME%\include" ^
 -I"%JAVA_HOME%\include\win32" ^
 -I"jnn\nativo" ^
 -I"jnn\nativo\cpu" ^
 -I"jnn\nativo\dispatch" ^
 -I"jnn\nativo\jni" ^
 %SRC% ^
 -o "%OUT_DIR%\%DLL%"

if errorlevel 1 (
  echo ERRO na compilacao JNI
  exit /b 1
)

copy C:\mingw64\bin\libgomp-1.dll "%OUT_DIR%" > nul
copy C:\mingw64\bin\libgcc_s_seh-1.dll "%OUT_DIR%" > nul
copy C:\mingw64\bin\libwinpthread-1.dll "%OUT_DIR%" > nul

echo Build nativo OK