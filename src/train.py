import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import pickle
import spacy  # for tokenizer
from PIL import Image  # Load img
from tqdm import tqdm
from dataset import get_loader
from utils import load_checkpoint, save_checkpoint
from model import CNNtoRNN
from config import config


transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_loader, dataset = get_loader(
        root_folder=config["root_folder"],
        annotation_file=config["annotation_file"],
        transform=transform,
        num_workers=2,
)
with open('vocab.pkl', 'wb') as file:
    pickle.dump(dataset.vocab, file, pickle.HIGHEST_PROTOCOL)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True
train_CNN = False

# Hyperparameters
embed_size = config["embed_size"]
hidden_size = config["hidden_size"]
vocab_size = len(dataset.vocab)
num_layers = config["num_layers"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]

# for tensorboard
writer = SummaryWriter("runs/flickr")
step = 0

# initialize model, loss etc
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Only finetune the CNN
for name, param in model.encoderCNN.inception.named_parameters():
    if "fc.weight" in name or "fc.bias" in name:
        param.requires_grad = True
    else:
        param.requires_grad = train_CNN

if load_model:
    step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

for epoch in range(num_epochs):
    # Uncomment the line below to see a couple of test cases
    # print_examples(model, device, dataset)
    print("epoch number: ", epoch)

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        }
        save_checkpoint(checkpoint)

    for idx, (imgs, captions) in tqdm(
        enumerate(train_loader), total=len(train_loader), leave=True
    ):
        imgs = imgs.to(device)
        captions = captions.to(device)

        outputs = model(imgs, captions[:-1])
        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
        )

        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1

        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()
