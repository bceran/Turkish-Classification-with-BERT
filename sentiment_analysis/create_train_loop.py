import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from create_dataset import CommentDataset
from create_model import BertModelClassifier

from config_file import *

DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_dataloader(data: Dataset,
                      batch_size: int) -> DataLoader:
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return dataloader


def train_single_epoch(model: nn.Module,
                       data_loader: DataLoader,
                       loss_fn: nn.modules.loss,
                       optimizer: optim,
                       device: str) -> float:
    model.train()
    total_loss = 0
    len_data = 0
    for data, attention_mask, labels in data_loader:
        data, attention_mask, labels = data.to(device), attention_mask.to(device), labels.to(device)
        output = model(data, attention_mask)
        loss = loss_fn(output, labels)

        total_loss += loss.item()
        len_data += len(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len_data


def evaluate_single_epoch(model: nn.Module,
                          data_loader: DataLoader,
                          loss_fn: nn.modules.loss,
                          device: str) -> float:
    model.eval()
    with torch.no_grad():
        total_loss = 0
        len_data = 0
        for data, attention_mask, labels in data_loader:
            data, attention_mask, labels = data.to(device), attention_mask.to(device), labels.to(device)
            output = model(data, attention_mask)
            loss = loss_fn(output, labels)

            total_loss += loss.item()
            len_data += len(data)

    return total_loss / len_data


def train_loop(model: nn.Module,
               train_data_loader: DataLoader,
               val_data_loader: DataLoader,
               loss_fn: nn.modules.loss,
               optimizer: optim,
               device: str,
               epochs: int,
               writer: SummaryWriter = None) -> None:
    loss_note_for_txt = ''
    for epoch in range(epochs):
        t0 = time.time()
        print(f"Epoch {epoch + 1}")
        train_loss = train_single_epoch(model, train_data_loader, loss_fn, optimizer, device)
        val_loss = evaluate_single_epoch(model, val_data_loader, loss_fn, device)
        t1 = time.time() - t0
        epoch_note = (f"=========== Epoch: {epoch}, --- loss: {train_loss:.4f}, val_loss: {val_loss:.4f},"
                      f" --- time: {t1 / 60:.2f} dk ===========")
        print(epoch_note)
        loss_note_for_txt = loss_note_for_txt + '\n' + epoch_note

        if writer is not None:
            writer.add_scalar('For Every Epoch/Training Loss', train_loss, epoch)
            writer.add_scalar('For Every Epoch/Validation Loss', val_loss, epoch)

    with open('../results/loss_note.txt', 'w') as f:
        f.write(loss_note_for_txt)
    print('\nTraining is over!\n')
    print('Train loop logging is saved in results/loss_note.txt\n')


def main():
    print(f"Using device : {DEVICE}")

    train_dataset = CommentDataset(annotations_file_path=TRAIN_ANNOTATIONS_FILE,
                                   device=DEVICE,
                                   tokenizer=TOKENIZER)

    val_dataset = CommentDataset(annotations_file_path=VAL_ANNOTATIONS_FILE,
                                 device=DEVICE,
                                 tokenizer=TOKENIZER)

    batch_size = BATCH_SIZE

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = BertModelClassifier(model=MODEL,
                                n_classes=N_CLASSES)

    loss_fn = nn.NLLLoss()  # softmax in model
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           eps=1e-8,
                           weight_decay=0.01)

    writer = SummaryWriter('../runs/train_loop_writer')

    train_loop(model=model,
               train_data_loader=train_dataloader,
               val_data_loader=val_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               device=DEVICE,
               epochs=EPOCHS,
               writer=writer)

    torch.save(model.state_dict(), "../models/son_model_kayit.pth")
    print("Trained feed forward net saved at son_model_kayit.pth in models folder.")


if __name__ == "__main__":
    main()
