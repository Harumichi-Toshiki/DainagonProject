import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import math
import shogi
from tqdm import tqdm
import matplotlib.pyplot as plt # グラフ描画用

# ==========================================
# ⚙️ 設定 (Dainagon v11: Radeon Special)
# ==========================================
DEVICE = torch_directml.device()
DATA_FILE = "data/train_data_suisho.txt"
VOCAB_FILE = "data/vocab_v2.json"
MODEL_SAVE_PATH = "model/chunagon_v11.pt"
GRAPH_SAVE_PATH = "model/learning_curve_v11.png"

# ★モデル構成: 10ブロック × 128チャンネル (ガチ構成)
BLOCKS = 10
CHANNELS = 128
INPUT_CHANNELS = 42

# ★学習設定
EPOCHS = 5
LEARNING_RATE = 0.0005
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 0.8

# ★メモリ対策 (勾配蓄積 + FP16)
# 論理バッチ(学習したいサイズ): 大きくする (2048 ~ 4096)
TARGET_BATCH_SIZE = 2048 
# 物理バッチ(VRAMに乗るサイズ): 小さくする (RX 9070 XTなら 256~512 はいけるはず)
# ※ エラーが出たら 128 に下げてください
PHYSICAL_BATCH_SIZE = 1024 
ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE

print(f"⚡ Settings: {BLOCKS} Blocks, {CHANNELS} Channels")
print(f"⚡ Batch: {PHYSICAL_BATCH_SIZE} (Physical) x {ACCUM_STEPS} steps = {TARGET_BATCH_SIZE} (Target)")

# DirectMLでのFP16 (AMP) は環境によって不安定なことがあります。
# Trueにしてエラーが出る場合は False にしてください。
USE_AMP = True

# ==========================================
# 共通関数・データセット
# ==========================================
TBL_NUM = str.maketrans("123456789", "987654321")
TBL_ALP = str.maketrans("abcdefghi", "ihgfedcba")

def rotate_move(move_usi):
    if "*" in move_usi:
        return move_usi[0] + "*" + move_usi[2:].translate(TBL_NUM).translate(TBL_ALP)
    return move_usi[:2].translate(TBL_NUM).translate(TBL_ALP) + \
           move_usi[2:4].translate(TBL_NUM).translate(TBL_ALP) + move_usi[4:]

def make_features(board):
    features = np.zeros((INPUT_CHANNELS, 9, 9), dtype=np.float32)
    is_white = (board.turn == shogi.WHITE)
    for sq in range(81):
        p = board.piece_at(sq)
        if p:
            if p.color == board.turn: idx = p.piece_type - 1
            else: idx = p.piece_type - 1 + 14
            target_sq = (80 - sq) if is_white else sq
            features[idx, target_sq // 9, target_sq % 9] = 1.0
    piece_order = [shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER, shogi.GOLD, shogi.BISHOP, shogi.ROOK]
    colors = [(board.turn, 28), (not board.turn, 35)]
    for color, offset in colors:
        hand = board.pieces_in_hand[color]
        for i, p_type in enumerate(piece_order):
            count = hand[p_type]
            if count > 0: features[offset + i, :, :] = count / 10.0
    return features

class DistillationDataset(Dataset):
    def __init__(self, txt_path, vocab):
        self.vocab = vocab
        self.samples = []
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                self.samples = [line.strip() for line in f]
        print(f"📖 Loaded {len(self.samples)} distilled samples.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        line = self.samples[idx]
        parts = line.split()
        sfen = " ".join(parts[:-2])
        move_usi = parts[-2]
        score_cp = int(parts[-1])
        
        board = shogi.Board(sfen)
        x = make_features(board)
        
        if board.turn == shogi.WHITE: move_label = rotate_move(move_usi)
        else: move_label = move_usi
        move_id = self.vocab.get(move_label, -100)
        
        # 評価値を勝率(0~1)に変換
        win_rate = 1.0 / (1.0 + math.exp(-score_cp / 600.0))
        
        return (torch.from_numpy(x), torch.tensor(move_id, dtype=torch.long), torch.tensor(win_rate, dtype=torch.float32))

# ==========================================
# モデル定義 (SE-ResNet / 128ch)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) # SE Block採用

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)

class DainagonNet(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS), nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResBlock(CHANNELS) for _ in range(BLOCKS)])
        self.fc_input_dim = CHANNELS * 9 * 9
        self.dropout = nn.Dropout(p=0.3)
        self.policy_head = nn.Linear(self.fc_input_dim, num_moves)
        self.value_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1), 
            nn.Sigmoid()
        )
    def forward(self, x):
        h = self.conv_input(x)
        for block in self.res_blocks: h = block(h)
        h = h.view(-1, self.fc_input_dim)
        h = self.dropout(h)
        return self.policy_head(h), self.value_head(h)

# ==========================================
# グラフ描画関数
# ==========================================
def plot_learning_curve(loss_history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Dainagon Learning Curve')
    plt.xlabel('Steps (x Accumulation)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"📊 Graph saved to {save_path}")
    plt.close()

# ==========================================
# 学習ループ (FP16 + GradAccum)
# ==========================================
def train():
    os.makedirs("model", exist_ok=True)
    with open(VOCAB_FILE, 'r') as f: vocab = json.load(f)
    
    print("📖 データセットを読み込んでいます...")
    dataset = DistillationDataset(DATA_FILE, vocab)
    
    loader = DataLoader(
        dataset, 
        batch_size=PHYSICAL_BATCH_SIZE, # 物理バッチを使用
        shuffle=True, 
        num_workers=4,
        persistent_workers=True,
        pin_memory=True
    )
    
    model = DainagonNet(len(vocab)).to(DEVICE)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 続きから学習を開始します: {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    else:
        print("✨ 新規学習を開始します")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # AMP Scaler (CUDA用だがDirectMLでも動く場合がある)
    # 動かない場合は通常のFP32で動作するようフォールバックする記述
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    criterion_p = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_v = nn.MSELoss()
    
    print("=== Training v11 Started ===")
    
    loss_history = [] # グラフ用データ

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (x, t_m, t_v) in enumerate(pbar):
            x, t_m, t_v = x.to(DEVICE), t_m.to(DEVICE), t_v.to(DEVICE)
            
            # ★FP16 (Mixed Precision) コンテキスト
            # DirectMLでautocastが効かない場合は自動でFP32になります
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                p, v = model(x)
                
                loss = (POLICY_WEIGHT * criterion_p(p, t_m)) + \
                       (VALUE_WEIGHT * criterion_v(v.squeeze(), t_v))
                
                # ★勾配蓄積: Lossを割る
                loss = loss / ACCUM_STEPS

            # Backward
            scaler.scale(loss).backward()
            
            # ★蓄積回数分溜まったら更新
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # グラフ用に記録 (更新のタイミングで)
                current_loss = loss.item() * ACCUM_STEPS
                loss_history.append(current_loss)
                pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
                
                # こまめに値を保存して学習曲線を更新しても良いかも
            else:
                # 蓄積中は表示だけ更新
                pbar.set_postfix({"Loss": f"{loss.item() * ACCUM_STEPS:.4f} (Accum)"})
            
            total_loss += loss.item() * ACCUM_STEPS # 平均計算用
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Finished. Average Loss: {avg_loss:.4f}")
        
        # モデル保存
        torch.save({'model': model.state_dict(), 'vocab': vocab}, MODEL_SAVE_PATH)
        print(f"💾 Model saved to {MODEL_SAVE_PATH}")
        
        # 毎エポック終了時にグラフを更新
        plot_learning_curve(loss_history, GRAPH_SAVE_PATH)

if __name__ == "__main__":
    train()