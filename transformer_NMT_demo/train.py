import torch
import torch.nn as nn
from dataset import *
import wandb
from transformer import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_model = False
save_model = False

# hyperparameter
num_epochs = 10
learning_rate = 1e-4
batch_size = 32

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
embedding_size = 128
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1
max_length = 100
forward_expansion = 4
src_pad_idx = src_vocab["<PAD>"]

# wandb.init(project='transformer4NMT', entity='hangy6-rls')
BATCH_SIZE = 128
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Hyperparameters
model = Transformer(
    src_vocab_size,
    tgt_vocab_size,
    embedding_size,
    num_heads,
    num_encoder_layers,
    num_encoder_layers,
    max_length,
    forward_expansion,
    dropout,
    src_pad_idx
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    for src, tgt in train_dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, src_pad_idx)

        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask,
                       src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, TGT_VOCAB_SIZE), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_dataloader)}")


# sentence = "ein pfred geht unter einer brucke neben einem boot."

# for epoch in range(num_epochs):
#     print(f"[Epoch {epoch} / {num_epochs}]")

#     if save_model:
#         pass

#     model.eval()
#     translated_sentence = translated_sentence(
#         model, sentence, german, english, device, max_length=100
#     )
#     print(f"Translated example sentence\n {translated_sentence}")
#     model.train()

#     for batch_idx, batch in enumerate(train_iterator):
#         inp_data = batch.src.to(device)
#         target = batch.tgt.to(device)

#         output = model(inp_data, target[:-1])
#         print(output.size())
#         output = output.reshape(-1, output.shape[2])
#         target = target[1:].reshape[-1]
#         optimizer.zero_grad()
#         loss = criterion(output, target)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
#         optimizer.step()
# wandb.watch(model, log='all')
# wandb.finish()