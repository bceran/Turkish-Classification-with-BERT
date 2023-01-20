# Turkish Text Classification with BERT

![huggingface](https://github.com/bceran/Turkish-Classification-with-BERT/blob/main/photos/huggingface.png)

This project includes a Turkish classifier written in PyTorch. In the project, a pre-trained BertModel was taken from the transformer library and fine-tuned. 

## Libraries Used

    torch==1.8.1
    pandas
    tensorboard
    transformers

## Requirements

- train.csv, validation.csv and test.csv dataset in data folder.
- pytorch_model.bin file from huggingface(bert-base-turkish-cased)
- Python environment with libraries in requirements.txt
- And please organize config_file.py
