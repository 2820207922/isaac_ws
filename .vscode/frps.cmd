
set WSL_IP=%1
set FRP_PATH=%2

net stop iphlpsvc
net start iphlpsvc
netsh interface portproxy add v4tov4 listenport=1234 listenaddress=0.0.0.0 connectport=1234 connectaddress=%WSL_IP%
netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=%WSL_IP%
netsh interface portproxy add v4tov4 listenport=7000 listenaddress=0.0.0.0 connectport=7000 connectaddress=%WSL_IP%
wsl -e %FRP_PATH%/frps 