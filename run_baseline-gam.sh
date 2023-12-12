dataset="HCPActivity"
batch_size="16"
model="GAM"
hidden="64"
main="main.py"
python $main --dataset $dataset --model $model --device 'cuda' --batch_size $batch_size --runs 10
