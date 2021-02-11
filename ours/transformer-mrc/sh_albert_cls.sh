#sketchy module
export DATA_DIR=/share/jpl/sentence_paraphrasing/AwesomeMRC/transformer-mrc/self_data/11_05_squad
export TASK_NAME=squad
CUDA_VISIBLE_DEVICES=0 python ./examples/run_cls.py \
    --model_type electra \
    --model_name_or_path /share/willXu/keypoint_experiment/pretrain_model/pretrain/electra-large \
    --task_name $TASK_NAME \
    --do_lower_case \
    --version_2_with_negative \
    --data_dir $DATA_DIR \
    --train_file train.json \
    --predict_file test.json \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_gpu_train_batch_size=6   \
    --per_gpu_eval_batch_size=8   \
    --warmup_steps=814 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --output_dir /share/jpl/sentence_paraphrasing/AwesomeMRC/transformer-mrc/models/squad/cls_no_squad_self_electra-large-v2 \
    --save_steps 2500 \
    --n_best_size=20 \
    --do_eval \
    --do_train \
#    --eval_all_checkpoints \
#    --fp16

