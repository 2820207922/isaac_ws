
set WSL_IP=%1
set WSL_FRP_PATH=%2
set WORKING_PATH=%3/.vscode

%WORKING_PATH%/sudo %WORKING_PATH%/frps.cmd %WSL_IP% %WSL_FRP_PATH%