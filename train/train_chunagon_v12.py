import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim
import torch_directml
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import math
import shogi
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ 設定 (Dainagon v12: Fixed for DirectML)
# ==========================================
DEVICE = torch_directml.device()
DATA_FILE = "data/train_data_suisho.txt"
VOCAB_FILE = "data/vocab_v2.json"
MODEL_SAVE_PATH = "model/chunagon_v12.5.pt" 
GRAPH_SAVE_PATH = "model/learning_curve_v12.png"

# ★モデル構成
BLOCKS = 10
CHANNELS = 128
INPUT_CHANNELS = 46

# ★学習設定
EPOCHS = 5
LEARNING_RATE = 0.0005
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 0.8

# ★メモリ対策
TARGET_BATCH_SIZE = 4096 
PHYSICAL_BATCH_SIZE = 512 
ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE

print(f"⚡ Settings: {BLOCKS} Blocks, {CHANNELS} Channels")
print(f"⚡ Batch: {PHYSICAL_BATCH_SIZE} (Physical) x {ACCUM_STEPS} steps = {TARGET_BATCH_SIZE} (Target)")

# ★DirectMLではAMPが不安定、かつ警告が出るため False に固定
USE_AMP = False

# ==========================================
# 共通関数・データセット (変更なし)
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
    
    # 1. 盤上の駒
    for sq in range(81):
        p = board.piece_at(sq)
        if p:
            if p.color == board.turn: idx = p.piece_type - 1
            else: idx = p.piece_type - 1 + 14
            target_sq = (80 - sq) if is_white else sq
            features[idx, target_sq // 9, target_sq % 9] = 1.0
            
    # 2. 持ち駒
    piece_order = [shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER, shogi.GOLD, shogi.BISHOP, shogi.ROOK]
    colors = [(board.turn, 28), (not board.turn, 35)]
    for color, offset in colors:
        hand = board.pieces_in_hand[color]
        for i, p_type in enumerate(piece_order):
            count = hand[p_type]
            if count > 0: features[offset + i, :, :] = count / 10.0

    # 3. 利きのヒートマップ
    for sq in range(81):
        target_sq = (80 - sq) if is_white else sq
        row, col = target_sq // 9, target_sq % 9
        
        my_attackers = board.attackers(board.turn, sq)
        cnt_my = len(my_attackers)
        if cnt_my > 0: features[42, row, col] = min(1.0, cnt_my / 5.0) 

        opp_attackers = board.attackers(not board.turn, sq)
        cnt_opp = len(opp_attackers)
        if cnt_opp > 0: features[43, row, col] = min(1.0, cnt_opp / 5.0)
        
    # 4. 手数
    move_count_val = min(board.move_number / 300.0, 1.0)
    features[44, :, :] = move_count_val

    # 5. 王手フラグ
    if board.is_check(): features[45, :, :] = 1.0

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

        # 手数の取得 (SFENの4番目の要素)
        try:
            game_move_count = int(parts[3])
        except:
            game_move_count = 0
        
        board = shogi.Board(sfen)
        x = make_features(board)
        
        if board.turn == shogi.WHITE: move_label = rotate_move(move_usi)
        else: move_label = move_usi
        move_id = self.vocab.get(move_label, -100)
        win_rate = 1.0 / (1.0 + math.exp(-score_cp / 800.0))
        
        return (torch.from_numpy(x), torch.tensor(move_id, dtype=torch.long), torch.tensor(win_rate, dtype=torch.float32), torch.tensor(game_move_count, dtype=torch.float32))

# ==========================================
# 🐯 Plan A: CBAM (そのままでOK)
# ==========================================
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True), # ここはReLUでOK
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_s, max_s], dim=1)
        spatial_att = self.sigmoid(self.conv_spatial(spatial))
        
        return x * spatial_att

# ==========================================
# 🐯 Plan B: DainagonBlock (Mish撤廃 -> ReLU)
# ==========================================
class DainagonBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Pre-Activation配置: BN -> ReLU -> Conv
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        
        self.cbam = CBAMBlock(channels)
        self.relu = nn.ReLU(inplace=True) # ★Mishの代わりにReLUを使用

    def forward(self, x):
        residual = x
        
        # Pre-Activation 1 (ReLU版)
        out = self.relu(self.bn1(x)) # F.mish -> self.relu
        out = self.conv1(out)
        
        # Pre-Activation 2 (ReLU版)
        out = self.relu(self.bn2(out)) # F.mish -> self.relu
        out = self.conv2(out)
        
        out = self.cbam(out)
        
        return out + residual

# ==========================================
# 🐯 Plan C: Interactive Head (Mish撤廃 -> ReLU)
# ==========================================
class DainagonNet(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, CHANNELS, 3, padding=1, bias=False),
            nn.BatchNorm2d(CHANNELS),
            nn.ReLU(inplace=True) # ★Mish -> ReLU
        )
        
        self.res_blocks = nn.ModuleList([DainagonBlock(CHANNELS) for _ in range(BLOCKS)])
        
        self.fc_input_dim = CHANNELS * 9 * 9
        self.dropout = nn.Dropout(p=0.3)
        
        # Head部分
        self.policy_fc = nn.Linear(self.fc_input_dim, 1024)
        self.policy_act = nn.ReLU(inplace=True) # ★Mish -> ReLU
        self.policy_head = nn.Linear(1024, num_moves)
        
        self.value_head = nn.Sequential(
            nn.Linear(self.fc_input_dim + 1024, 256), 
            nn.ReLU(inplace=True), # ★Mish -> ReLU
            nn.Dropout(p=0.3),
            nn.Linear(256, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.conv_input(x)
        for block in self.res_blocks: h = block(h)
        h = h.view(-1, self.fc_input_dim)
        h = self.dropout(h)
        
        p_feat = self.policy_act(self.policy_fc(h))
        policy_out = self.policy_head(p_feat)
        
        v_input = torch.cat([h, p_feat], dim=1)
        value_out = self.value_head(v_input)
        
        return policy_out, value_out

# ==========================================
# グラフ描画・学習ループ (AMP無効化対応)
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
    plt.close()

def train():
    os.makedirs("model", exist_ok=True)
    with open(VOCAB_FILE, 'r') as f: vocab = json.load(f)
    
    print("📖 データセットを読み込んでいます...")
    dataset = DistillationDataset(DATA_FILE, vocab)
    
    loader = DataLoader(
        dataset, 
        batch_size=PHYSICAL_BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        persistent_workers=False,
        pin_memory=True
    )
    
    model = DainagonNet(len(vocab)).to(DEVICE)

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 続きから学習を開始します: {MODEL_SAVE_PATH}")
        # DirectML用: CPUにロードしてからデバイスへ送る
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu') 
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        model.to(DEVICE)
    else:
        print("✨ 新規学習を開始します")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_p = nn.CrossEntropyLoss(ignore_index=-100)
    # criterion_v は手動計算するため使いません
    
    print("=== Training v12.5 Started (Endgame Weighted) ===")
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # ★修正点1: move_count を正しく受け取る
        for i, (x, t_m, t_v, move_count) in enumerate(pbar):
            # ★修正点2: 変数名を統一して転送
            x = x.to(DEVICE)
            t_m = t_m.to(DEVICE)
            t_v = t_v.to(DEVICE)
            move_count = move_count.to(DEVICE)
            
            # 順伝播
            p, v = model(x)

            # --- 損失関数の計算 (ここが心臓部！) ---
            
            # 1. Policy Loss (指し手の一致率)
            loss_p = criterion_p(p, t_m)
            
            # 2. Value Loss (終盤の重み付け)
            # 手動で二乗誤差を計算
            loss_v_raw = (v.squeeze() - t_v) ** 2
            
            # ★ 80手目以降なら重みを2.0倍、それ以外は1.0倍
            weight = torch.where(move_count > 80, 2.0, 1.0)
            
            # 重みをかけて平均を取る
            loss_v = (loss_v_raw * weight).mean()
            
            # 合算
            loss = (POLICY_WEIGHT * loss_p) + (VALUE_WEIGHT * loss_v)
            # -------------------------------------
            
            loss = loss / ACCUM_STEPS
            loss.backward() 
            
            if (i + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                current_loss = loss.item() * ACCUM_STEPS
                loss_history.append(current_loss)
                pbar.set_postfix({"Loss": f"{current_loss:.4f}"})
            
            total_loss += loss.item() * ACCUM_STEPS
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Finished. Average Loss: {avg_loss:.4f}")
        torch.save({'model': model.state_dict(), 'vocab': vocab}, MODEL_SAVE_PATH)
        plot_learning_curve(loss_history, GRAPH_SAVE_PATH)

if __name__ == "__main__":
    train()