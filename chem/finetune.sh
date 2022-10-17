#### GIN fine-tuning
split=scaffold
dataset=$1
input_model=$2

[ -z "${input_model}" ] && input_model="init_weights/pretrained.pth"


for runseed in 0 1 2 3 4 5 6 7 8 9
do
model_file=${unsup}
python finetune.py --input_model_file $input_model --split $split --runseed $runseed --gnn_type gin --dataset $dataset
done

# --lr 1e-3 --epochs 100