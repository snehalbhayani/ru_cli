# import sqlite3
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# # Connect to SQLite database
# DB_PATH = "../my_cli/history.db"  # Change this to your database path
# conn = sqlite3.connect(DB_PATH)
# cursor = conn.cursor()

# # Load commands from the database
# cursor.execute("SELECT command FROM history")  # Change 'your_table' to the actual table name
# commands = [row[0] for row in cursor.fetchall()]
# conn.close()

# # Step 1: Build Vocabulary
# def build_vocab(commands):
#     chars = sorted(set("".join(commands)))  # Unique characters
#     char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # Start from 1 (0 is for padding)
#     char_to_idx["<PAD>"] = 0
#     idx_to_char = {i: c for c, i in char_to_idx.items()}
#     return char_to_idx, idx_to_char

# char_to_idx, idx_to_char = build_vocab(commands)
# vocab_size = len(char_to_idx)
# # Step 2: Prepare Training Data (Seq length = 3)
# SEQ_LEN = 3

# def create_sequences(commands, seq_len, char_to_idx):
#     input_data, target_data = [], []

#     for command in commands:
#         encoded = [char_to_idx[c] for c in command]

#         if len(encoded) < seq_len + 1:
#             continue  # Ignore short commands

#         for i in range(len(encoded) - seq_len):
#             input_data.append(encoded[i:i + seq_len])
#             target_data.append(encoded[i + 1:i + seq_len + 1])

#     return torch.tensor(input_data, dtype=torch.long), torch.tensor(target_data, dtype=torch.long)

# train_data, target_data = create_sequences(commands, SEQ_LEN, char_to_idx)
# num_epochs = 2000
# batch_size, seq_len = train_data.shape
# print(train_data.shape)
# hidden_size = 34
# num_layers = 1

# # import pdb
# # pdb.set_trace()

# # Step 3: Define LSTM Model
# class LSTMModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim=16, hidden_size=128, num_layers=1):
#         super(LSTMModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, x, h, c):
#         x = self.embedding(x)
#         out, (h, c) = self.lstm(x, (h, c))
#         logits = self.fc(out)
#         return logits, h, c

# # Initialize Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = LSTMModel(vocab_size, 16, hidden_size, num_layers).to(device)

# # Step 4: Training
# criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding in loss
# optimizer = optim.Adam(model.parameters(), lr=0.001)



# for epoch in range(num_epochs):
#     h = torch.zeros(num_layers, batch_size, hidden_size).to(device)
#     c = torch.zeros(num_layers, batch_size, hidden_size).to(device)

#     train_data, target_data = train_data.to(device), target_data.to(device)

#     optimizer.zero_grad()
#     logits, _, _ = model(train_data, h, c)

#     loss = criterion(logits.view(-1, vocab_size), target_data.view(-1))
#     loss.backward()
#     optimizer.step()

#     if (epoch + 1) % 100 == 0:
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# # Step 5: Generate Text
# def generate_text(model, start_seq, char_to_idx, idx_to_char, length=10):
#     model.eval()
#     input_seq = [char_to_idx[c] for c in start_seq]
#     input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

#     h = torch.zeros(num_layers, 1, hidden_size).to(device)
#     c = torch.zeros(num_layers, 1, hidden_size).to(device)

#     generated_text = start_seq
#     for _ in range(length):
#         logits, h, c = model(input_tensor, h, c)
#         probs = torch.softmax(logits[:, -1, :], dim=-1)
#         next_char_idx = torch.multinomial(probs, 1).item()
#         if next_char_idx == 0:
#             break
#         generated_text += idx_to_char[next_char_idx]
#         input_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

#     return generated_text

# # Generate text from a starting sequence
# print("Generated:", generate_text(model, "aa1", char_to_idx, idx_to_char, length=1))

import sqlite3
import nltk
import requests
from nltk.tokenize import sent_tokenize
import datetime
from pathlib import Path

# Download the book text and sentence tokenizer
nltk.download("punkt", quiet=True)

# Download book text from Project Gutenberg
book_url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"  # Pride and Prejudice
response = requests.get(book_url)
BOOK_TEXT = response.text if response.status_code == 200 else ""

# Tokenize into sentences
sentences = sent_tokenize(BOOK_TEXT)

# Database file path
db_path = Path("commands.db")

# Connect to SQLite and create the table
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS commands (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        command TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        frequency INTEGER DEFAULT 0
    )
""")

# Insert each sentence into a new row
current_timestamp = datetime.datetime.now().isoformat()
for sentence in sentences:
    cursor.execute("""
        INSERT INTO commands (command, timestamp, frequency)
        VALUES (?, ?, 0)
    """, (sentence.strip(), current_timestamp))

# Commit and close
conn.commit()
conn.close()
print("Database created and populated successfully!")
