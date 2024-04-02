from dataset import *
from model import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import argparse
import wandb

def train_epoch(dataloader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        optimizer.zero_grad()

        decoder_outputs, _, _ = model(input_tensor, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(train_dataloader, model, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, n_epochs + 1)):
        loss = train_epoch(train_dataloader, model, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            tqdm.write('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments for training')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for the training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the training')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden unit size for RNN')
    parser.add_argument('--num-epochs', type=int, default=100, help='Total number of training epochs')
    parser.add_argument('--print-every', type=int, default=5, help='Number of epochs between each print of training log')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Ratio of data in database used for training')

    args = parser.parse_args()
    
    wandb.init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_lang, output_lang, train_dataloader, val_dataloader = get_dataloader(args.batch_size, device, args.train_ratio)

    model = RNN(input_lang.n_words, args.hidden_size, output_lang.n_words, device).to(device)

    train(train_dataloader, model, args.num_epochs, print_every=args.print_every)

    evaluateRandomly(val_dataloader, model, input_lang, output_lang)