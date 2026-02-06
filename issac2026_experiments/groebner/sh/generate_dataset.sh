#!/usr/bin/env bash
# Generate Groebner (F -> G) datasets.
# - QQ:   configs/data.yaml
# - ZZ:   configs/data_ZZ.yaml
# - GF7:  configs/data_GF7.yaml (任意)

# QQ
python3 generate_dataset.py --config_path configs/data.yaml

# # ZZ
# python3 generate_dataset.py --config_path configs/data_ZZ.yaml

# # GF7 (必要なら)
# python3 generate_dataset.py --config_path configs/data_GF7.yaml

