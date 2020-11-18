#!/usr/bin/env python
#training
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import konlpy
import pickle
import models
import utils
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str,
                help="input image name")

    args = parser.parse_args()
    img_name = args.img

    with open('./train_loader.pkl', 'rb') as f:
        train_loader = pickle.load(f)

    with open('./dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100
    train_CNN = False
    
    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = models.CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    step = utils.load_checkpoint(torch.load("my_checkpoint_Adam_GRU_seq2seq1.pth.tar"), model, optimizer)

    model.train()
    utils.print_caption(model, device, dataset,img_name)
    
if __name__=="__main__":
    main()
