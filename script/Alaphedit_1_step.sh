CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.evaluate_relation   \
    --alg_name="AlphaEdit"  \
    --model_name="./CurrKE/EleutherAI_gpt-j-6B"   \
    --hparams_fname="EleutherAI_gpt-j-6B.json" \
    --ds_name="zsre_twostep"\
    --dataset_size_limit="2000"  \
    --num_edits="100" \
    --downstream_eval_steps="5"