dataset="HCPActivity"
model=AirGC
main="main.py"
lr=0.01
lambda_amp=0.5
runs=5
K=5
epochs=100

python $main  --dataset $dataset --model $model --lr $lr --lambda_amp $lambda_amp --runs $runs --epochs $epochs --K 5
