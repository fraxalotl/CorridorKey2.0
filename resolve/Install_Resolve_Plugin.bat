@echo off
rem Installs the CorridorKey Fusion node into DaVinci Resolve.
rem Uses the venv created during CorridorKey setup if it exists.
setlocal
set "ROOT=%~dp0.."
set "PY=python"
if exist "%ROOT%\venv\Scripts\python.exe" set "PY=%ROOT%\venv\Scripts\python.exe"
"%PY%" "%~dp0install_resolve_plugin.py" %*
echo.
pause
