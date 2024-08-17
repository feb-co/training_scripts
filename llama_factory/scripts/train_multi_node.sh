#!/bin/sh


# 定义环境变量
WORK_DIR=$1
CONFIG_FILE=$2
NUM_NODES=$3
HOST_FILE=$4
ENV_FILE=$5
CONDA_BIN=$6
CONDA_ENV=$7

# 检查参数数量
if [ $# -ne 7 ]; then
    echo "Usage: $0 <WORK_DIR> <CONFIG_FILE> <NUM_NODES> <HOST_FILE> <ENV_FILE> <CONDA_BIN> <CONDA_ENV>"
    exit 1
fi


# 检查主机文件和环境变量文件是否存在
for file in "$HOST_FILE" "$ENV_FILE"; do
    if [ ! -f "$file" ]; then
        echo "Error: File '$file' does not exist"
        exit 1
    fi
done

# 读取主机列表
mapfile -t HOSTS < $HOST_FILE


# 检查节点数是否有效
if [ $NUM_NODES -lt 1 ] || [ $NUM_NODES -gt ${#HOSTS[@]} ]; then
    echo "Error: Number of nodes must be between 1 and ${#HOSTS[@]}"
    exit 1
fi


# 定义主节点地址（假设第一个主机为主节点）
MASTER_ADDR=$(echo ${HOSTS[0]} | cut -d'@' -f2)


# 读取环境变量文件内容
ENV_VARS=$(cat "$ENV_FILE" | sed 's/^/export /')


# 函数：在后台执行命令
run_command() {
    index=$1
    host=$2

    command="FORCE_TORCHRUN=1 NNODES=$NUM_NODES RANK=$index MASTER_ADDR=$MASTER_ADDR MASTER_PORT=2223 llamafactory-cli train $CONFIG_FILE"

    if [ $index -eq 0 ]; then
        echo "Running main node (RANK=0) locally"
        (
            eval "$ENV_VARS"
            echo "Starting main node command execution"
            eval "$command"
        ) &
    else
        echo "Running node RANK=$index on $host"
        ssh $host "
            cd $WORK_DIR
            $ENV_VARS
            echo \"$(date '+%Y-%m-%d %H:%M:%S') - Starting command execution on node $index\"
            $CONDA_BIN run -n $CONDA_ENV $command
        " &
    fi
}

# 执行命令
for ((i=0; i<$NUM_NODES; i++)); do
    run_command $i "${HOSTS[$i]}"
done

# 等待所有后台任务完成
echo "Waiting for all nodes to complete"
wait
echo "All commands completed"

# 检查每个节点的状态
for ((i=0; i<$NUM_NODES; i++)); do
    if [ $i -eq 0 ]; then
        echo "Checking status of main node (RANK=0)"
    else
        echo "Checking status of node RANK=$i on ${HOSTS[$i]}"
        ssh ${HOSTS[$i]} "ps aux | grep llamafactory-cli" || echo "Node $i process not found"
    fi
done
