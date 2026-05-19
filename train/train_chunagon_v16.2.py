import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_directml
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import os
import json
import math
import shogi
import mmap
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# ==========================================
# 👑 Dainagon 15B/192ch Final (5 Epochs Limit Break)
# ==========================================
DEVICE = torch_directml.device()
DATA_FILE = "data/train_data_suisho10Mn_shuffled.txt" 
VOCAB_FILE = "data/vocab_v2.json"
MODEL_SAVE_PATH = "model/chunagon_16.2.pt" 
GRAPH_SAVE_PATH = "model/learning_curve_16.2.png" 

INPUT_CHANNELS = 55  # 55のままだが、54番(一手前)は常に0.0
BLOCKS = 15
CHANNELS = 192

EPOCHS = 5           # ★ 5エポックで限界突破
LEARNING_RATE = 0.00001
TARGET_BATCH_SIZE = 2048 
PHYSICAL_BATCH_SIZE = 1024
ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE
TOTAL_STEPS_PER_EPOCH = 100000000 // PHYSICAL_BATCH_SIZE 
PLOT_INTERVAL_SEC = 1800 

def plot_learning_curve(loss_history, save_path):
    if not loss_history: return
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Dainagon Final (15B/192ch) Learning Curve')
    plt.xlabel('Steps (x Accumulation)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def bb_to_squares(bb):
    while bb > 0:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r

def make_features_v16(board, is_white):
    features = np.zeros((INPUT_CHANNELS, 9, 9), dtype=np.float32)
    
    for sq in range(81):
        p = board.piece_at(sq)
        if p:
            idx = p.piece_type - 1 if p.color == board.turn else p.piece_type - 1 + 14
            target_sq = (80 - sq) if is_white else sq
            y, x = divmod(target_sq, 9)
            features[idx, y, x] = 1.0

    piece_order = [shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER, shogi.GOLD, shogi.BISHOP, shogi.ROOK]
    colors = [(board.turn, 28), (not board.turn, 35)]
    for color, offset in colors:
        hand = board.pieces_in_hand[color]
        for i, p_type in enumerate(piece_order):
            count = hand[p_type]
            if count > 0: features[offset + i, :, :] = count / 10.0

    for sq in range(81):
        p = board.piece_at(sq)
        if not p: continue
        pt = p.piece_type
        if pt not in [shogi.ROOK, shogi.PROM_ROOK, shogi.BISHOP, shogi.PROM_BISHOP]: continue
        is_self = (p.color == board.turn)
        is_rook = (pt == shogi.ROOK or pt == shogi.PROM_ROOK)
        ch = (42 if is_rook else 43) if is_self else (44 if is_rook else 45)
        attacks_bb = board.attacks_from(pt, sq, board.occupied, p.color)
        for t_sq in bb_to_squares(attacks_bb):
            target_sq = (80 - t_sq) if is_white else t_sq
            y, x = divmod(target_sq, 9)
            features[ch, y, x] = 1.0

    k_sq_self = board.king_squares[board.turn]
    k_sq_opp = board.king_squares[not board.turn]
    for k_sq, base_ch in [(k_sq_self, 46), (k_sq_opp, 48)]:
        if k_sq is None: continue
        ky, kx = divmod(k_sq, 9)
        for r, r_ch in [(3, 1), (2, 0)]:
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    ny, nx = ky + dy, kx + dx
                    if 0 <= ny < 9 and 0 <= nx < 9:
                        orig_sq = ny * 9 + nx
                        target_sq = (80 - orig_sq) if is_white else orig_sq
                        ty, tx = divmod(target_sq, 9)
                        features[base_ch + r_ch, ty, tx] = 1.0

    if board.is_check(): features[50, :, :] = 1.0
    features[51, :, :] = min(board.move_number / 200.0, 1.0)
    for i in range(9):
        features[52, :, i] = i / 8.0 
        features[53, i, :] = i / 8.0 
        
    # ★ 諸悪の根源だった 54番チャンネル(一手前) の代入処理を完全削除！常に0.0になる。

    return features

class ChunkedMmapDataset(IterableDataset):
    def __init__(self, data_file, vocab):
        self.data_file, self.vocab = data_file, vocab
        self.TBL_NUM = str.maketrans("123456789", "987654321")
        self.TBL_ALP = str.maketrans("abcdefghi", "ihgfedcba")
    def _rotate_move(self, move_usi):
        if "*" in move_usi: return move_usi[0] + "*" + move_usi[2:].translate(self.TBL_NUM).translate(self.TBL_ALP)
        return move_usi[:2].translate(self.TBL_NUM).translate(self.TBL_ALP) + move_usi[2:4].translate(self.TBL_NUM).translate(self.TBL_ALP) + move_usi[4:]
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        file_size = os.path.getsize(self.data_file)
        if worker_info is None:
            start, end = 0, file_size
        else:
            per_worker = file_size // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else file_size
        with open(self.data_file, "r", encoding="utf-8") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mm.seek(start)
            if start > 0: mm.readline()
            while mm.tell() < end:
                line_bytes = mm.readline()
                if not line_bytes: break
                line = line_bytes.decode('utf-8')
                parts = line.split()
                if len(parts) < 6: continue
                
                sfen = " ".join(parts[:4])
                move_usi = parts[4]
                score_cp = int(parts[5])
                try: move_count = int(parts[3])
                except: move_count = 0
                
                is_white = (parts[1] == 'w')
                board = shogi.Board(sfen)
                x = make_features_v16(board, is_white)
                label = self._rotate_move(move_usi) if is_white else move_usi
                move_id = self.vocab.get(label, -100)
                win_rate = 1.0 / (1.0 + math.exp(-score_cp / 800.0))
                
                yield (torch.from_numpy(x), torch.tensor(move_id, dtype=torch.long), 
                       torch.tensor(win_rate, dtype=torch.float32), torch.tensor(move_count, dtype=torch.float32))
            mm.close()

class ValidationDataset(IterableDataset):
    def __init__(self, data_file, vocab, num_samples=10000):
        self.data_file, self.vocab, self.num_samples = data_file, vocab, num_samples
    def __iter__(self):
        with open(self.data_file, "r", encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - self.num_samples * 400))
            f.readline() 
            count = 0
            for line in f:
                if count >= self.num_samples: break
                parts = line.split()
                if len(parts) < 6: continue
                sfen = " ".join(parts[:4])
                move_usi, score_cp = parts[4], int(parts[5])
                is_white = (parts[1] == 'w')
                board = shogi.Board(sfen)
                x = make_features_v16(board, is_white)
                if is_white:
                    tbl_n, tbl_a = str.maketrans("123456789", "987654321"), str.maketrans("abcdefghi", "ihgfedcba")
                    move_usi = move_usi[0] + "*" + move_usi[2:].translate(tbl_n).translate(tbl_a) if "*" in move_usi else move_usi[:2].translate(tbl_n).translate(tbl_a) + move_usi[2:4].translate(tbl_n).translate(tbl_a) + move_usi[4:]
                move_id = self.vocab.get(move_usi, -100)
                win_rate = 1.0 / (1.0 + math.exp(-score_cp / 800.0))
                yield (torch.from_numpy(x), torch.tensor(move_id, dtype=torch.long), torch.tensor(win_rate, dtype=torch.float32))
                count += 1

def evaluate(model, val_loader):
    model.eval()
    correct, total, val_loss_v, batch_count = 0, 0, 0, 0
    with torch.no_grad():
        for x, t_m, t_v in val_loader:
            x, t_m, t_v = x.to(DEVICE), t_m.to(DEVICE), t_v.to(DEVICE)
            p, v = model(x)
            pred_move = p.argmax(1)
            correct += (pred_move == t_m).sum().item()
            total += t_m.size(0)
            val_loss_v += ((v.squeeze() - t_v)**2).mean().item()
            batch_count += 1  
    return correct / total if total > 0 else 0, val_loss_v / batch_count if batch_count > 0 else 0

# ==========================================
# 🧠 モデル定義 (SiLU採用・モダンCBAM)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool, self.max_pool = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.silu = nn.SiLU() # ★ ReLU -> SiLU
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.silu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.silu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.silu = nn.SiLU() # ★ ReLU -> SiLU
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca, self.sa = ChannelAttention(channels), SpatialAttention()
    def forward(self, x):
        residual = x
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return self.silu(out + residual)

class DainagonNet(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv_input = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, CHANNELS, 3, padding=1, bias=False), nn.BatchNorm2d(CHANNELS), nn.SiLU())
        self.res_blocks = nn.ModuleList([CBAMBlock(CHANNELS) for _ in range(BLOCKS)])
        self.fc_input_dim = CHANNELS * 9 * 9
        
        self.dropout = nn.Dropout(p=0.15)
        self.policy_head = nn.Linear(self.fc_input_dim, num_moves)
        self.value_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256), nn.SiLU(), # ★
            nn.Dropout(p=0.15), nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        h = self.conv_input(x)
        for block in self.res_blocks: h = block(h)
        h = self.dropout(h.view(-1, self.fc_input_dim))
        return self.policy_head(h), self.value_head(h)

def train():
    os.makedirs("model", exist_ok=True)
    with open(VOCAB_FILE, 'r') as f: vocab = json.load(f)
    
    dataset = ChunkedMmapDataset(DATA_FILE, vocab)
    loader = DataLoader(dataset, batch_size=PHYSICAL_BATCH_SIZE, num_workers=6)
    val_dataset = ValidationDataset(DATA_FILE, vocab)
    val_loader = DataLoader(val_dataset, batch_size=PHYSICAL_BATCH_SIZE)

    model = DainagonNet(len(vocab)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ★ Cosine Annealing スケジューラ (5エポックかけて学習率をゼロに近づける)
    total_steps = TOTAL_STEPS_PER_EPOCH * EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    print("=== Training v16.2 Started (15B/192ch + Cosine Annealing + Focal Loss) ===")
    
    loss_history = []
    last_plot_time = time.time() 

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader, total=TOTAL_STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_running_loss = 0.0
        step_count = 0

        for i, (x, t_m, t_v, move_count) in enumerate(pbar):
            x, t_m, t_v, move_count = x.to(DEVICE), t_m.to(DEVICE), t_v.to(DEVICE), move_count.to(DEVICE)
            p, v = model(x)
            
            # Policy & Value Loss (要素単位)
            loss_p_base = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')(p, t_m)
            loss_v_base = (v.squeeze() - t_v)**2
            
            # 👑 Focal & Late-Game Weighting
            importance_weight = torch.where((t_v < 0.1) | (t_v > 0.9), 2.5, 1.0)
            move_weight = torch.where(move_count > 80, 2.0, 1.0)
            final_weight = importance_weight * move_weight
            
            loss_p = (loss_p_base * final_weight).mean()
            loss_v = (loss_v_base * final_weight).mean()
            
            loss = (loss_p + loss_v) / ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                scheduler.step() # ★ バッチごとにブレーキを踏む
                optimizer.zero_grad()
                
                current_loss = loss.item() * ACCUM_STEPS
                loss_history.append(current_loss)
                epoch_running_loss += current_loss
                step_count += 1
                
                # 現在の学習率を取得して表示
                current_lr = scheduler.get_last_lr()[0]
                
                pbar.set_postfix({
                    "Loss": f"{current_loss:.4f}", 
                    "L_v": f"{loss_v.item():.4f}",
                    "LR": f"{current_lr:.2e}" # 学習率の低下を確認できる
                })

                current_time = time.time()
                if current_time - last_plot_time > PLOT_INTERVAL_SEC:
                    acc, v_err = evaluate(model, val_loader)
                    print(f"\n📈 【テスト結果】 Policy精度: {acc:.2%}, Value誤差: {v_err:.5f}")
                    model.train() 
                    plot_learning_curve(loss_history, GRAPH_SAVE_PATH)
                    torch.save({'model': model.state_dict(), 'vocab': vocab}, MODEL_SAVE_PATH)
                    last_plot_time = current_time

        torch.save({'model': model.state_dict(), 'vocab': vocab}, f"model/chunagon_ep{epoch+1}.pt")
        plot_learning_curve(loss_history, GRAPH_SAVE_PATH)

if __name__ == "__main__":
    train()