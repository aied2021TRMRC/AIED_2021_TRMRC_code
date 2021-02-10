export SQUAD_DIR=../data/sed
export MODEL_NAME=../base_models/albert-base-squad2
export OUT_MODEL_DIR=./models

CUDA_VISIBLE_DEVICES=4 python ../transformers-3.3.1/examples/question-answering/run_squad.py \
  --model_type bart \
  --model_name_or_path $MODEL_NAME \
  --config_name $MODEL_NAME \
  --tokenizer_name $MODEL_NAME \
  --do_eval \
  --do_train \
  --version_2_with_negative \
  --data_dir $SQUAD_DIR \
  --train_file train.json \
  --predict_file valid.json \
  --per_gpu_train_batch_size 4\
  --learning_rate 3e-5 \
  --do_lower_case \
  --num_train_epochs 4 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUT_MODEL_DIR \
  --overwrite_output_dir 