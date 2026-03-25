import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from collections import Counter

VOCAB_SIZE = 30000
EMBED_DIM = 64
HIDDEN_DIM = 256  
SEQ_LEN = 31      
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
DATA_PATH = './data/text.txt'
MODEL_SAVE_PATH = './ckpt/lstm_lm.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextDataset(Dataset):
    def __init__(self, file_path, seq_len, max_vocab_size):
        self.seq_len = seq_len
        with open(file_path, 'r', errors='ignore', encoding='gb2312') as f:
            text = f.read().replace('\n', ' ').split() # 按空格分词示例

        word_counts = Counter(text)
        top_words = [w for w, _ in word_counts.most_common(max_vocab_size - 1)]
        self.word2id = {w: i+1 for i, w in enumerate(top_words)}
        self.word2id['<UNK>'] = 0
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.data = [self.word2id.get(w, 0) for w in text]
        
        # 实际可用的样本数量
        self.num_samples = len(self.data) - seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=1,         # 单层 LSTM
                            batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x, hidden):
        emb = self.embedding(x) 
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def train():
    # 准备数据
    dataset = TextDataset(DATA_PATH, SEQ_LEN, VOCAB_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 初始化模型、损失函数、优化器
    model = LSTMLanguageModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"开始训练... 模型已加载至 {device}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            h0 = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM).to(device)
            c0 = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM).to(device)
            hidden = (h0, c0)
            
            # 前向传播
            optimizer.zero_grad()
            logits, hidden = model(inputs, hidden)
            
            loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"===> Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")
        
        if avg_loss < 3.0:
            print(f"目标达成！Average Loss ({avg_loss:.4f}) < 3.0。正在保存模型...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"模型已保存至 {MODEL_SAVE_PATH}")
            break 

if __name__ == '__main__':
    if not os.path.exists('./data'):
        os.makedirs('./data')
        print("请在 ./data/ 目录下放入 text.txt 文件后再运行。")
    elif not os.path.exists(DATA_PATH):
        print("未找到 ./data/text.txt 文件，请检查路径。")
    else:
        train()