@echo off
REM Run a Python command inside the MSVC + venv environment needed to build
REM PyTorch CUDA extensions on Windows. cl.exe and the Windows SDK headers/libs
REM only exist after vcvars64.bat runs, so JIT extension builds fail without this.
REM
REM Usage (from anywhere):  kernels\winbuild.bat -m kernels.build_smoke
REM Equivalent to running the command from an "x64 Native Tools Command Prompt".
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0.."
set "PATH=%CD%\.venv\Scripts;%PATH%"
".venv\Scripts\python.exe" %*
