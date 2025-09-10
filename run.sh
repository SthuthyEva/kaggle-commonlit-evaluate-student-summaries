eval "$(conda shell.bash hook)"
conda activate kaggle

# python train.py -config config_s.yaml -fold 0
# python train.py -config config_s.yaml -fold 1
# python train.py -config config_s.yaml -fold 2
# python train.py -config config_s.yaml -fold 3

python train.py -config config_b.yaml -fold 0
python train.py -config config_b.yaml -fold 1
python train.py -config config_b.yaml -fold 2
python train.py -config config_b.yaml -fold 3

# python train.py -config config_l.yaml -fold 0
# python train.py -config config_l.yaml -fold 1
# python train.py -config config_l.yaml -fold 2
# python train.py -config config_l.yaml -fold 3