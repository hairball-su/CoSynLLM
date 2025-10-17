# CoSynLLM
CoSynLLM: an LLM-assisted predictive framework for predicting drug combination synergy

# File
- Data/: the two dataset used in the experiments
- model/: save the trained model weights
- pretrained_model: pretrained models used in the experiment
- dataset.py: process the dataset
- early_stop.py: early stop mechanism
- fp_main.py: main tex
- gen_fp.py: calculate fingerprint
- gen_prompts.py: convert description to pkl
- metrics.py: set the metrics used in the experiment
- model.py: CoSynLLM model
- _drug_smiles.xlsx: generated drug description
  
# Requirements
- python 3.9
- pytorch 1.12.1
- numpy 1.26.4
- pandas 1.4.3
- scikit-learn 1.6.0
- tqdm 4.67.1
- rdkit 2024.3.2
  
# Run the code
下载预训练模型
 ```
Please download it yourself from https://huggingface.co/sentence-transformers/all-mpnet-base-v2. The path name can be used as a reference 'pretrained_model/sentence-transformers/all-mpnet-base-v2'.
 ```
选择数据集，生成输入特征向量
 ```
dataset = 'ALMANAC'  # ONEIL or ALMANAC
python gen_fp.py
python gen_prompts.py
```
运行主文件
 ```
python fp_main.py -d ALMANAC -g 0  #d: dataset  g: gpu id
