import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
from keras.preprocessing import sequence
from keras.datasets import imdb
import math
import numpy as np

print("The environment is set up.")

# 加载IMDB词汇表
word_index = imdb.get_word_index()
# 反转词汇表（从索引到单词的映射）
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = "[PAD]"  # 填充符
index_to_word[1] = "[START]"  # 起始符
index_to_word[2] = "[UNK]"  # 未知词
index_to_word[3] = "[UNUSED]"

# 将整数序列转换为自然语言句子
def decode_review(encoded_sequence):
    return ' '.join([index_to_word.get(i, "[UNK]") for i in encoded_sequence])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        output = self.fc(enc_output[:, 0, :])  # Use only the [CLS] token
        return output

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            loss = criterion(output, tgt)
            total_loss += loss.item()
            predictions = output.argmax(dim=1)
            correct += (predictions == tgt).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


# Parameters
max_features = 20000
maxlen = 64
batch_size = 512

# Load Datasets
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

train_data = TensorDataset(torch.tensor(x_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
test_data = TensorDataset(torch.tensor(x_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'log/imdb.pt'
loaded_model = Transformer(20000, 2, 128, 4, 2, 512, 64, 0.1).to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()  # Set the model to evaluation mode

def predict(model, sample, device):
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        sample = torch.tensor(sample, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
        output = model(sample)  # Forward pass
        # Check the shape of the output
        # print(f"Model output shape: {output.shape}")

        # Adjusting argmax to get the predicted label
        prediction = output.argmax(dim=-1).item()  # Get predicted label (adjust dim to -1)
        return prediction

# Read a sample from the test dataset (e.g., the first one)
sample, true_label = x_test[5], y_test[5]
decoded_sample = decode_review(sample)

print(f"Decoded Sample: {decoded_sample}")
# Make a prediction
predicted_label = predict(loaded_model, sample, device)
print(f"Predicted Label: {'Positive' if predicted_label else 'Negative'}")

# Optional: Compare prediction with true label
if predicted_label == true_label:
    print("correct!")
else:
    print("incorrect.")