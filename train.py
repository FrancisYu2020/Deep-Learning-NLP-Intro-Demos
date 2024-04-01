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

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

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

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, input_tensor, input_lang, output_lang):
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = tensorToSentence(decoded_ids, output_lang)
    return decoded_words, decoder_attn
    
def evaluateRandomly(val_dataloader, encoder, decoder, input_lang, output_lang):
    count = 0
    total_score = 0
    for data in val_dataloader:
        count += 1
        p = np.random.uniform()
        source_tensor, target_tensor = data
        target_words = tensorToSentence(target_tensor.squeeze(), output_lang)
        source_sentence = ' '.join(tensorToSentence(source_tensor.squeeze(), input_lang))
        target_sentence = ' '.join(target_words)
        if p < 0.1:
            print('>', source_sentence)
            print('=', target_sentence)
        output_words, _ = evaluate(encoder, decoder, source_tensor, input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        if p < 0.1:
            print('<', output_sentence)
            print('')
        total_score += sentence_bleu([target_words], output_words)
    print(f'The BLEU score on validation data is: {total_score / count: .4f}')

hidden_size = 128
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_lang, output_lang, train_dataloader, val_dataloader = get_dataloader(batch_size, device)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words, device).to(device)

train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

encoder.eval()
decoder.eval()
evaluateRandomly(val_dataloader, encoder, decoder, input_lang, output_lang)