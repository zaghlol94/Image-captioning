import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import argparse
from utils import load_checkpoint
import torchvision.transforms as transforms
from PIL import Image  # Load img
from model import CNNtoRNN
from config import config


def inference(model, device, vocab, image):
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    img = Image.open(image).convert("RGB")
    test_img1 = transform(img).unsqueeze(
        0
    )
    img.show()
    print(
        "OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), vocab))
    )


parser = argparse.ArgumentParser(description="translate string from german to english")
parser.add_argument("-i", "--image", type=str, required=True, help="image that you want to translate")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('vocab.pkl', 'rb') as inp:
    vocab = pickle.load(inp)

embed_size = config["embed_size"]
hidden_size = config["hidden_size"]
vocab_size = len(vocab)
num_layers = config["num_layers"]
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
inference(model, device, vocab, args.image)
