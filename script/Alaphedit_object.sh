CUDA_VISIBLE_DEVICES=0 \
python3 -m experiments.evaluate_relationstp1    \
    --alg_name="AlphaEdit"  \
    --model_name="./model/Llama3-8B"   \
    --hparams_fname="Llama3-8B.json" \
    --ds_name="zsre_ob"\
    --dataset_size_limit="2000"  \
    --num_edits="100" \
    --downstream_eval_steps="5" \

