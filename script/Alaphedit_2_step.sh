CUDA_VISIBLE_DEVICES=4 \
python3 -m experiments.evaluate_relation    \
    --alg_name="ROME"  \
    --model_name="./model/Llama3-8B"   \
    --hparams_fname="Llama3-8B.json" \
    --ds_name="zsre_twostep"\
    --dataset_size_limit="100"  \
    --num_edits="100" \
    --downstream_eval_steps="5"