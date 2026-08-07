#--
import pandas as pd 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import nltk 
from nltk.tokenize import word_tokenize

# %%
with open('train.txt','r',encoding='utf-8') as f :
    text  = f.read()
print(text)

# %%
import re
text = re.sub(r'^[IVXLCDM]+\.\s*', '', text, flags=re.MULTILINE)
text = re.sub(r'\s+', ' ', text).strip()
text = text.lower()
text = text.replace("_the_",'the')

# %%
print(text)

# %%
nltk.download("punkt_tab")

# %%
# tokenise 
tokens = word_tokenize(text)
tokens

# %%
vocab = sorted(set(tokens))

# %%
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for word, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda c: " ".join(itos[i] for i in c )

# %%
encoded_data = encode(tokens)

# %%
print(encoded_data[:20])
print(decode(encoded_data[:20]))

# %%
n = int(0.8*(len(encoded_data)))
train_data = encoded_data[:n]
val_data = encoded_data[n:]


# %%
from torch.utils.data import Dataset,DataLoader

# %%
encoded_data = torch.tensor(encoded_data, dtype=torch.long)

# %%
class TextDataset(Dataset):
    def __init__(self,data,seq_length):
        super().__init__()
        self.data = data
        self.seq_length=seq_length
    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]

        return x, y

# %%
dataset = TextDataset(encoded_data, seq_length=30)

# %%
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# %%
X, Y = next(iter(loader))
X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)
print(X.shape)
print(Y.shape)

# %%
class LSTM_Model(nn.Module):
    def __init__(self,vocab_size,embd,hidden_size,num_layer):
        super().__init__()
        self.emb = nn.Embedding(vocab_size,embd)
        self.lstm = nn.LSTM(embd,hidden_size=hidden_size,num_layers=num_layer, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size,vocab_size)
    def forward(self,x):
        x = self.emb(x)
        out,(hiden,cell) = self.lstm(x)
        logits = self.linear(out)
        return(logits)
    

# %%
model = LSTM_Model(
    vocab_size=len(vocab),
    embd=128,
    hidden_size=256,
    num_layer=2,
)
model = model.to(device)

# %%
# training loop 
epochs  = 100
learning_rate = 0.001
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
device = 'cuda' if torch.cuda.is_available() else "cpu"
for epoch in range(epochs):
  total_loss = 0

  for batch_x, batch_y in loader:

    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    optimizer.zero_grad()

    output = model(batch_x)
    output = output.transpose(1,2)
    loss = criteria(output, batch_y)

    loss.backward()

    optimizer.step()

    total_loss = total_loss + loss.item()

  print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")

# %%
avg_loss = total_loss/len(vocab)

# %%
avg_loss

# %%
torch.save(model.state_dict(), "lstm_text_generator.pth")

# %%

def generate(model, seed_text, max_new_tokens=50, temperature=1.0):
    model.eval()
    text = seed_text
    text = re.sub(r'^[IVXLCDM]+\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = text.replace("_the_",'the')
    tokens = word_tokenize(seed_text.lower())
    context = encode(tokens)  # list of ints
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

    generated = context

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)              # (1, seq_len, vocab_size)
            last_logits = logits[:, -1, :] / temperature   # (1, vocab_size) - only last timestep
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1) - sample, not argmax

            generated = torch.cat([generated, next_token], dim=1)

    output_ids = generated[0].tolist()
    return decode(output_ids)

print(generate(model, seed_text="To Sherlock Holmes she is always ", max_new_tokens=30, temperature=0.8))


# %%


# %%


# %%



