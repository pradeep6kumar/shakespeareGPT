import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
import math

# Adjusted hyperparameters
BATCH_SIZE = 32
BLOCK_SIZE = 128
LEARNING_RATE = 3e-4
N_EMBD = 512  # Reduced from 768
N_HEAD = 8    # Reduced from 12
N_LAYER = 8   # Reduced from 12
DROPOUT = 0.2 # Increased from 0.1
WEIGHT_DECAY = 0.01

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        
        # Added dropout to attention
        self.attention = nn.MultiheadAttention(
            N_EMBD, 
            N_HEAD, 
            dropout=DROPOUT, 
            batch_first=True,
            bias=False  # Reduce parameters
        )
        
        # Modified feed forward with additional dropout
        self.feed_forward = nn.Sequential(
            nn.Linear(N_EMBD, 3 * N_EMBD, bias=False),  # Reduced multiplier from 4 to 3
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(3 * N_EMBD, N_EMBD, bias=False),
            nn.Dropout(DROPOUT)
        )
        
        # Layer normalization with improved epsilon
        self.ln1 = nn.LayerNorm(N_EMBD, eps=1e-5)
        self.ln2 = nn.LayerNorm(N_EMBD, eps=1e-5)
        
    def forward(self, x, mask=None):
        # Pre-norm architecture for better training stability
        attn_out = self.attention(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)[0]
        x = x + attn_out
        x = x + self.feed_forward(self.ln2(x))
        return x

class ShakespeareModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD, eps=1e-5)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)
        
        # Improved weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * N_LAYER))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * N_LAYER))
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
                
    def forward(self, idx):
        B, T = idx.shape
        
        # Get embeddings and apply dropout
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # Apply dropout to combined embeddings
        x = self.drop(tok_emb + pos_emb)
        
        # Create causal mask
        mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(idx.device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        
        # Create overlapping sequences for better context
        self.examples = []
        stride = block_size // 2  # Add stride for overlapping sequences
        for i in range(0, len(data) - block_size, stride):
            x = data[i:i + block_size]
            y = data[i + 1:i + block_size + 1]
            self.examples.append((x, y))
            
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_model(model, train_dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        
        # Forward pass with mixed precision
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_dataloader)

def main():
    # Load and preprocess data
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create full dataset
    full_dataset = TextDataset(text, BLOCK_SIZE)
    
    # Split into train and validation sets (90-10 split)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=True
    )
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShakespeareModel(full_dataset.vocab_size).to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Load checkpoint if exists
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists('shakespeare_model_best.pth'):
        print("Loading checkpoint 'shakespeare_model_best.pth'")
        checkpoint = torch.load('shakespeare_model_best.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_loss']
        print(f"Loaded checkpoint (epoch {start_epoch})")
    
    # Training loop
    n_epochs = 1000 
    patience = 5
    patience_counter = 0
    
    for epoch in range(start_epoch, n_epochs):
        train_loss = train_model(model, train_dataloader, optimizer, scheduler, device)
        val_loss = evaluate_model(model, val_dataloader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Validation loss improved to {val_loss:.6f}. Saving checkpoint...')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': val_loss,
            }, 'shakespeare_model_best.pth')
        
        # Only keep target loss check
        if train_loss < 0.0999999:
            print(f'Target loss achieved! Training completed at epoch {epoch+1}')
            break

if __name__ == '__main__':
    main()