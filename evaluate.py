import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(model, input_tensor, input_lang, output_lang):
    model.eval()
    with torch.no_grad():
        decoder_outputs, decoder_hidden, decoder_attn = model(input_tensor)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = tensorToSentence(decoded_ids, output_lang)
    return decoded_words, decoder_attn
    
def evaluateRandomly(val_dataloader, model, input_lang, output_lang):
    count = 0
    total_score = 0
    for data in tqdm(val_dataloader):
        count += 1
        p = np.random.uniform()
        source_tensor, target_tensor = data
        target_words = tensorToSentence(target_tensor.squeeze(), output_lang)
        source_sentence = ' '.join(tensorToSentence(source_tensor.squeeze(), input_lang))
        target_sentence = ' '.join(target_words)
        if p < 0.1:
            tqdm.write('>', source_sentence)
            tqdm.write('=', target_sentence)
        output_words, _ = evaluate(model, source_tensor, input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        if p < 0.1:
            tqdm.write('<', output_sentence)
            tqdm.write('')
        total_score += sentence_bleu([target_words], output_words)
    print(f'The BLEU score on validation data is: {total_score / count: .4f}')