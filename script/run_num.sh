#!/bin/bash

# --- 在这里配置你的参数 ---

# GPU设备ID
export CUDA_VISIBLE_DEVICES=1

# 模型和数据集参数
MODEL_NAME="./CurrKE/EleutherAI_gpt-j-6B"
HPARAMS_FNAME="EleutherAI_gpt-j-6B.json"
DS_NAME="zsre_twostep"
DATASET_SIZE_LIMIT="2000"
NUM_EDITS="100"
DOWNSTREAM_EVAL_STEPS="5"

# 需要依次执行的算法名称列表
# 在引号内添加或删除算法名称，用空格隔开
ALG_NAMES=("MEMIT" "MEMIT_rect" "MEMIT_prune" "NSE" "ROME" "FT")

# --- 脚本主程序 ---

# 遍历算法名称列表
for alg_name in "${ALG_NAMES[@]}"; do
    echo "============================================================"
    echo "正在为算法 [$alg_name] 执行评估..."
    echo "============================================================"

    # 构建并执行命令
    # 如果命令执行失败 (||)，则打印错误信息并继续下一个循环
    python3 -m experiments.evaluate_relation \
        --alg_name="$alg_name" \
        --model_name="$MODEL_NAME" \
        --hparams_fname="$HPARAMS_FNAME" \
        --ds_name="$DS_NAME" \
        --dataset_size_limit="$DATASET_SIZE_LIMIT" \
        --num_edits="$NUM_EDITS" \
        --downstream_eval_steps="$DOWNSTREAM_EVAL_STEPS" || echo "
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! 算法 [$alg_name] 执行失败，已跳过。
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    "
done

echo "============================================================"
echo "所有评估任务已执行完毕。"
echo "============================================================"