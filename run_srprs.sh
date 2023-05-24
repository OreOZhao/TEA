cd src

version="TEA"
gpus='1'

# en_de
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset en_de_15k_V1"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --neighbor"
paras="$paras --nsp"
paras="$paras --template_id 4"
paras="$paras --version ${version}"
echo $paras

python -u TEAPreprocess.py $paras
paras="$paras --fold 1"
paras="$paras --gpus ${gpus}"
python -u TEATrain.py $paras


# en_fr
paras=""
paras="$paras --datasets_root ../data"
paras="$paras --dataset en_de_15k_V1"
paras="$paras --result_root ../outputs"
paras="$paras --pretrain_bert_path ../pre_trained_models/bert-base-multilingual-uncased"
paras="$paras --log"
paras="$paras --relation"
paras="$paras --neighbor"
paras="$paras --nsp"
paras="$paras --template_id 4"
paras="$paras --version ${version}"
echo $paras

python -u TEAPreprocess.py $paras
paras="$paras --fold 1"
paras="$paras --gpus ${gpus}"
python -u TEATrain.py $paras
