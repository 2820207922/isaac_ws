
HOST_IP=$1
TEMPLATE_CONFIG=$2
FRPC_PATH=$3

tmp_config=/tmp/frpc.toml
rm -f $tmp_config
echo serverAddr = "\"$HOST_IP\"" >> $tmp_config
cat $TEMPLATE_CONFIG >> $tmp_config
/$FRPC_PATH/frpc -c $tmp_config
