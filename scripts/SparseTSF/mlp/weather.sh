if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SparseTSF

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

seq_len=720
for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 4 \
    --model_type 'mlp' \
    --d_model 128 \
    --enc_in 21 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 256 --learning_rate 0.05
done
