import torch
import torch.nn as nn

from transformers import BertModel


class BertModelClassifier(nn.Module):
    def __init__(self,
                 model: str,
                 n_classes: int,
                 dropout_rate: float = 0.4):
        super(BertModelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model)
        self.Linear1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.Linear2 = nn.Linear(1024, n_classes)
        self.dropout = nn.Dropout(p=dropout_rate)

        # freeze first 11 Layer (12th Layer is trainable)
        for name, param in list(self.bert.named_parameters())[:-18]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        hidden_layers = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
        mean_pooled = torch.sum(hidden_layers*attention_mask.unsqueeze(-1),
                                dim=1)/torch.sum(attention_mask.unsqueeze(-1), dim=1)
        # mean of last hidden embeddings except padding tokens
        output = self.Linear1(mean_pooled)
        output = nn.functional.relu(output)
        output = self.dropout(output)
        output = self.Linear2(output)
        output = nn.functional.softmax(output, dim=1)

        return output


def main():
    model = BertModelClassifier("../model",
                                3)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


if __name__ == "__main__":
    main()
