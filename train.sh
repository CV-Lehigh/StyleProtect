export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"   
export INSTANCE_DIR="CLEAN_DATA_PATH"
export OUTPUT_DIR="PROTECTED_DATA_PATH"
export SAVE_MODEL_DIR="SAVE_MODEL_PATH"
export INFER_DIR="INFER_PATH"



for folder in $INSTANCE_DIR/*; do
  accelerate launch style_protect.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$folder \
  --output_dir=$OUTPUT_DIR/${folder##*/} \
  --instance_prompt="a photo of an artwork" \
  --resolution=512 \
  --noaug \
  --learning_rate=1e-5 \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --mixed_precision bf16  \
  --alpha=5e-3  \
  --eps=0.1
done

for folder in $OUTPUT_DIR/*; do
  accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision bf16 \
  --snr_gamma=5.0 \
  --instance_data_dir=$folder \
  --output_dir=$SAVE_MODEL_DIR/${folder##*/} \
  --instance_prompt="an artwork in sks style" \
  --use_8bit_adam \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
done

python infer.py \
  --path=$SAVE_MODEL_DIR \
  --output_path=$INFER_DIR

