# TEA
Code for "From Alignment to Entailment: A Unified Textual Entailment Framework for Entity Alignment", Findings of ACL 2023.

## Dependencies
- python 3.9
- pytorch 1.12.1
- transformers 4.24.0
- tqdm

## Dataset

You can download the DBP15K and SRPRS dataset from [JAPE](https://github.com/nju-websoft/JAPE), [RSN](https://github.com/nju-websoft/RSN), or [SDEA](https://github.com/zhongziyue/SDEA).

1. Unzip the datasets in `TEA/data`.
2. Preprocess the datasets.

```bash
cd src
python DBPPreprocess.py
python SRPRSPreprocess.py
```

## Pre-trained Language Model

You can download the pre-trained language model `bert-base-multilingual-uncased` from [huggingface](https://huggingface.co/bert-base-multilingual-uncased) and put the model in `TEA/pre_trained_models`

## Project Structure

```
TEA/
├── src/: The soruce code. 
├── data/: The datasets. 
│   ├── DBP15k/: The downloaded DBP15K benchmark. 
│   │   ├── fr_en/
│   │   ├── ja_en/
│   │   ├── zh_en/
│   ├── entity-alignment-full-data/: The downloaded SRPRS benchmark. 
│   │   ├── en_de_15k_V1/
│   │   ├── en_fr_15k_V1/
├── pre_trained_models/: The pre-trained transformer-based models. 
│   ├── bert-base-multilingual-uncased: The model used in our experiments.
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   ├── ......
```

## How to run

To run TEA, use the example script `run_dbp15k.sh` or `run_srprs.sh`. You could customize the following parameters:

- --nsp: trains with TEA-NSP. If not specified, the default is TEA-MLM.
- --neighbor: adds neighbor sequences in training. If not specified, the model is only trained with attribute sequences.
- --template_id: changes the template of prompt according to the `template_list` in `src/KBConfig.py`.

You could also run FT-EA with `src/FTEATrain.py`.

## Acknowledgements
Our codes are modified based on [SDEA](https://github.com/zhongziyue/SDEA). We would like to appreciate their open-sourced work.

## Citation
Please cite the following paper as reference if you find our work useful.

```bibtex
@inproceedings{zhao-etal-2023-alignment,
    title = "From Alignment to Entailment: A Unified Textual Entailment Framework for Entity Alignment",
    author = "Zhao, Yu  and
      Wu, Yike  and
      Cai, Xiangrui  and
      Zhang, Ying  and
      Zhang, Haiwei  and
      Yuan, Xiaojie",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.559",
    pages = "8795--8806",
}
```
