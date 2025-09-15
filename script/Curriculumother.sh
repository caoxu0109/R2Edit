CUDA_VISIBLE_DEVICES=4 \
python3 -m experiments.curriculumKE    \
    --alg_name="CurriculumEdit"  \
    --model_name="./model/gpt2-xl"   \
    --hparams_fname="gpt2-xl.json" \
    --ds_name="zsre_twostep"\
    --dataset_size_limit="100"  \
    --num_edits="100" \
    --downstream_eval_steps="5" \