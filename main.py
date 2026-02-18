import torch
import torch.nn as nn
import torch.optim as optim



with open("data.txt","r") as f:
    text=f.read().lower()

print(text)

words=text.split()
vocab = sorted(set(words))
vocab = ["<PAD>", "<UNK>"] + vocab

print("Vocab size:", len(vocab))
print(vocab)

words_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_words = {i: word for i, word in enumerate(vocab)}

PAD_IDX = words_to_idx["<PAD>"]
UNK_IDX = words_to_idx["<UNK>"]


s="i love ai"

tokens=[words_to_idx[word] for word in s.split()]
print(tokens)


# create training sequences
sentences = text.split("\n")

inputs = []
targets = []

for line in sentences:
    words = line.split()
    
    for i in range(1, len(words)):
        input_seq = words[:i]      # words before
        target_word = words[i]     # next word
        
        inputs.append([words_to_idx.get(w, UNK_IDX) for w in input_seq])
        targets.append(words_to_idx.get(target_word, UNK_IDX))

print("\nSample training pairs:\n")

for i in range(len(inputs)):
    print(inputs[i], "->", targets[i])


# find max input length
max_len = max(len(seq) for seq in inputs)

# pad sequences so all same length
padded_inputs = []
for seq in inputs:
    padded = seq + [PAD_IDX] * (max_len - len(seq))
    padded_inputs.append(padded)

X = torch.tensor(padded_inputs)
y = torch.tensor(targets)

print("\nTensor shapes:")
print("X:", X.shape)
print("y:", y.shape)


# model parameters
vocab_size = len(vocab)
embed_dim = 32

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (x.size(-1) ** 0.5)
        
        weights = self.softmax(scores)
        
        out = torch.matmul(weights, V)
        return out


class NextWordModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = SelfAttention(embed_dim)
        
        # Removed dependency on max_len
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)          
        x = self.attention(x)          
        
        # Aggregate sequence to handle variable lengths
        x = x.mean(dim=1)              
        x = self.fc(x)
        return x

model = NextWordModel(vocab_size, embed_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 500

for epoch in range(epochs):
    optimizer.zero_grad()
    
    output = model(X)
    loss = criterion(output, y)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


def predict_next(text_input):
    model.eval()
    
    words = text_input.split()
    # Handle unknown words
    tokens = [words_to_idx.get(w, UNK_IDX) for w in words]
    input_tensor = torch.tensor([tokens])
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    print(f"Input: '{text_input}' -> Next word prediction:", idx_to_words[pred])

print("\nTesting predictions:")
predict_next("i love")
predict_next("ai is")
predict_next("coding is")


def generate_text(start_text, max_words=6):
    model.eval()
    
    current_text = start_text
    
    for _ in range(max_words):
        words = current_text.split()
        
        # keep only last X words if we wanted a fixed window
        
        tokens = [words_to_idx.get(w, UNK_IDX) for w in words]
        
        input_tensor = torch.tensor([tokens])
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
        
        next_word = idx_to_words[pred]
        current_text += " " + next_word
    
    print("Generated:", current_text)


print("\nGenerated sentences:\n")
generate_text("i")
generate_text("i love")
generate_text("ai")
generate_text("coding")


