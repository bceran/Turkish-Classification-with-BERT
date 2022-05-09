import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

import pandas as pd


class CommentDataset(Dataset):
    def __init__(self,
                 annotations_file_path: str,  # csv file path
                 device: str,  # cpu or cuda
                 tokenizer: str,
                 max_length: int = 64,
                 ):
        self.annotations = pd.read_csv(annotations_file_path, index_col=0)
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer,
                                                       do_lower_case=True)
        self.max_len = max_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        comment = self.annotations.iloc[item, 0]
        label = self.annotations.iloc[item, 1]
        comment_ids, attention_mask = self._tokenize_comment(comment)
        label = self._return_label_as_tensor(label)

        return comment_ids, attention_mask, label

    def _tokenize_comment(self, comment):
        encoding = self.tokenizer.encode_plus(comment,
                                              max_length=self.max_len,
                                              add_special_tokens=True,  # Add '[CLS]' and '[SEP]' token
                                              return_token_type_ids=False,
                                              padding='max_length',  # add [PAD] token
                                              return_attention_mask=True,
                                              truncation=True,
                                              return_tensors='pt')  # return pytorch tensor

        return encoding['input_ids'], encoding['attention_mask']

    @staticmethod
    def _return_label_as_tensor(label):
        if label == 'positive':
            return torch.tensor(2, dtype=torch.long)
        elif label == 'neutral':
            return torch.tensor(1, dtype=torch.long)
        else:
            return torch.tensor(0, dtype=torch.long)


def main():
    dataset = CommentDataset('../data/short_csv.csv',
                             'cpu',
                             '../tokenizer/')

    comment_ids, attention_mask, label = dataset[8]

    print(f"comment: {comment_ids}")
    print(f"attention mask: {attention_mask}")
    print(f"label: {label}")


if __name__ == "__main__":
    main()
