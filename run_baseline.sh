dataset="HCPActivity"
batch_size="16"
model="GCNConv"
hidden="64"
main="main.py"
epochs=100
runs=10

python $main --dataset $dataset --model $model --device 'cuda' --batch_size $batch_size --runs runs --epochs $epochs
