#!/bin/bash

# 设置固定的参数
# 将通用参数定义为变量，方便修改
ALG_NAME="ROME"
MODEL_NAME="./CurrKE/gpt2-xl"
HPARAMS_FNAME="gpt2-xl.json"
DS_NAME="zsre_relation"
DATASET_SIZE_LIMIT="100"
NUM_EDITS="100"
DOWNSTREAM_EVAL_STEPS="5"
BASE_DATA_PATH="./data/relation_edit/Split_IDK"

# 循环从 1 到 700
for (( i=1; i<=2; i+=1 ))
do
    # 使用 printf 格式化数字，使其为三位数（例如：1 -> 001, 12 -> 012）
    formatted_num=$(printf "%03d" $i)

    # 构建当前批次的文件路径
    file_path="${BASE_DATA_PATH}/batch_${formatted_num}.json"

    # 打印将要执行的命令，方便调试和跟踪
    echo "==========================================================="
    echo "===> Processing Batch: ${formatted_num}"
    echo "===> File: ${file_path}"
    echo "==========================================================="

    # 检查文件是否存在，如果不存在则跳过
    if [ ! -f "$file_path" ]; then
        echo "Warning: File ${file_path} not found, skipping."
        continue
    fi

    # 执行Python脚本
    # 使用变量来构建命令
    CUDA_VISIBLE_DEVICES=4 \
    python3 -m experiments.evaluate_relation    \
        --alg_name="${ALG_NAME}" \
        --model_name="${MODEL_NAME}" \
        --hparams_fname="${HPARAMS_FNAME}" \
        --ds_name="${DS_NAME}" \
        --dataset_size_limit="${DATASET_SIZE_LIMIT}" \
        --num_edits="${NUM_EDITS}" \
        --downstream_eval_steps="${DOWNSTREAM_EVAL_STEPS}" \
        --file="${file_path}"

    # 检查上一个命令的退出状态，如果失败则退出脚本
    if [ $? -ne 0 ]; then
        echo "Error: Command failed for batch ${formatted_num}. Aborting."
        exit 1
    fi
done

echo "All batches processed successfully!"