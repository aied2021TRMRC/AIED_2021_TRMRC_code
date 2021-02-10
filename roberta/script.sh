export SQUAD_DIR=../data/sed
export MODEL_NAME=../base_models/roberta-base-squad2
export OUTPUT_MODEL_DIR=./models

@ python ../transformers-3.3.1/examples/question-answering/run_squad.py \
  --model_type roberta \
  --model_name_or_path $MODEL_NAME \
  --config_name $MODEL_NAME \
  --tokenizer_name $MODEL_NAME \
  --do_train \
  --do_eval \
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
  --output_dir $OUTPUT_MODEL_DIR \
  --overwrite_output_dir \