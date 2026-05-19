import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
# ⚙️ 設定 (Dainagon v16.2: Automated ResNet)
# ==========================================
DEVICE = torch_directml.device()
DATA_FILE = "data/train_data_suisho10Mn_shuffled.txt" 
VOCAB_FILE = "data/vocab_v2.json"
MODEL_SAVE_PATH = "model/chunagon_v16_2.pt" 
BEST_MODEL_PATH = "model/chunagon_v16_2_best.pt" 
GRAPH_SAVE_PATH = "model/learning_curve_v16_2.png" 

INPUT_CHANNELS = 55 
BLOCKS = 16
CHANNELS = 192

# 学習設定
EPOCHS = 5
LEARNING_RATE = 0.0005 # CosineAnnealingで下がるため、初期値は少し高めでOK
TARGET_BATCH_SIZE = 2048 
PHYSICAL_BATCH_SIZE = 1024
ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE
TOTAL_STEPS_PER_EPOCH = 100000000 // PHYSICAL_BATCH_SIZE 
PLOT_INTERVAL_SEC = 1800 

def plot_learning_curve(loss_history, save_path):
    if not loss_history: return
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Dainagon v16.2 Learning Curve')
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
        if worker_info is None: start, end = 0, file_size
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
                parts = line_bytes.decode('utf-8').split()
                if len(parts) < 6: continue
                sfen, move_usi, score_cp = " ".join(parts[:4]), parts[4], int(parts[5])
                try: move_count = int(parts[3])
                except: move_count = 0
                is_white = (parts[1] == 'w')
                board = shogi.Board(sfen)
                x = make_features_v16(board, is_white)
                label = self._rotate_move(move_usi) if is_white else move_usi
                move_id = self.vocab.get(label, -100)
                win_rate = 1.0 / (1.0 + math.exp(-score_cp / 800.0))
                yield (torch.from_numpy(x), torch.tensor(move_id, dtype=torch.long), torch.tensor(win_rate, dtype=torch.float32), torch.tensor(move_count, dtype=torch.float32))
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
                sfen, move_usi, score_cp = " ".join(parts[:4]), parts[4], int(parts[5])
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
            
            # ★ 評価時も BCE(Logits) で計算
            val_loss_v += F.binary_cross_entropy_with_logits(v.squeeze(), t_v).item()
            batch_count += 1  
    return correct / total if total > 0 else 0, val_loss_v / batch_count if batch_count > 0 else 0

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
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
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return self.relu(out + residual)

class DainagonNet(nn.Module):
    def __init__(self, num_moves):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, CHANNELS, 3, padding=1, bias=False), 
            nn.BatchNorm2d(CHANNELS), nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([CBAMBlock(CHANNELS) for _ in range(BLOCKS)])
        self.fc_input_dim = CHANNELS * 9 * 9
        self.dropout = nn.Dropout(p=0.3)
        self.policy_head = nn.Linear(self.fc_input_dim, num_moves)
        
        # ★補正1: Value Head から Sigmoid を完全削除（BCE対応）
        self.value_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256), nn.ReLU(),
            nn.Dropout(p=0.3), nn.Linear(256, 1) 
        )
        
        # ★補正2: ResNetのポテンシャルを引き出す He(Kaiming) 初期化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.conv_input(x)
        for block in self.res_blocks: h = block(h)
        h = h.view(-1, self.fc_input_dim)
        h = self.dropout(h)
        return self.policy_head(h), self.value_head(h)

def train():
    os.makedirs("model", exist_ok=True)
    with open(VOCAB_FILE, 'r') as f: vocab = json.load(f)
    
    dataset = ChunkedMmapDataset(DATA_FILE, vocab)
    loader = DataLoader(dataset, batch_size=PHYSICAL_BATCH_SIZE, num_workers=6)
    val_dataset = ValidationDataset(DATA_FILE, vocab)
    val_loader = DataLoader(val_dataset, batch_size=PHYSICAL_BATCH_SIZE)

    model = DainagonNet(len(vocab)).to(DEVICE)
    
    # ★補正3: 丸暗記を防ぐ AdamW (Weight Decay付き)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # ★補正4: 学習率の自動減衰 (CosineAnnealingLR)
    total_scheduler_steps = (TOTAL_STEPS_PER_EPOCH // ACCUM_STEPS) * EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_scheduler_steps)
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 Loading checkpoint: {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)

    print("=== Training v16.2 Started (AdamW + CosineLR + BCE + He Init) ===")
    
    loss_history = []
    last_plot_time = time.time() 
    best_v_err = float('inf') # ★ Bestモデル判定用

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader, total=TOTAL_STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}")
        
        epoch_running_loss = 0.0
        step_count = 0

        for i, (x, t_m, t_v, move_count) in enumerate(pbar):
            x, t_m, t_v, move_count = x.to(DEVICE), t_m.to(DEVICE), t_v.to(DEVICE), move_count.to(DEVICE)
            p, v = model(x)
            
            loss_p = nn.CrossEntropyLoss(ignore_index=-100)(p, t_m)
            
            # ★補正5: BCEWithLogitsLoss によるValue解像度の爆上げ
            loss_v_base = F.binary_cross_entropy_with_logits(v.squeeze(), t_v, reduction='none')
            loss_v = (loss_v_base * torch.where(move_count > 80, 2.0, 1.0)).mean()
            
            loss = (loss_p + loss_v) / ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step() # ★毎ステップ学習率を自動調整
                optimizer.zero_grad()
                
                current_loss = loss.item() * ACCUM_STEPS
                loss_history.append(current_loss)
                
                epoch_running_loss += current_loss
                step_count += 1
                avg_loss = epoch_running_loss / step_count
                
                pbar.set_postfix({
                    "Loss": f"{current_loss:.4f}", 
                    "L_v(BCE)": f"{loss_v.item():.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.6f}"
                })

                current_time = time.time()
                if current_time - last_plot_time > PLOT_INTERVAL_SEC:
                    acc, v_err = evaluate(model, val_loader)
                    print(f"\n📈 【テスト結果】 Policy精度: {acc:.2%}, Value誤差(BCE): {v_err:.5f}")
                    
                    # ★補正6: 過学習前に「一番賢い状態」を自動保存
                    if v_err < best_v_err:
                        best_v_err = v_err
                        torch.save({'model': model.state_dict(), 'vocab': vocab}, BEST_MODEL_PATH)
                        print(f"🌟 最高精度更新！ Bestモデルを保存しました: {BEST_MODEL_PATH}")

                    model.train() 
                    plot_learning_curve(loss_history, GRAPH_SAVE_PATH)
                    torch.save({'model': model.state_dict(), 'vocab': vocab}, MODEL_SAVE_PATH)
                    print(f"⏰ {time.strftime('%H:%M:%S')} - Periodic save & plot updated.")
                    last_plot_time = current_time

        torch.save({'model': model.state_dict(), 'vocab': vocab}, MODEL_SAVE_PATH)
        plot_learning_curve(loss_history, GRAPH_SAVE_PATH)

if __name__ == "__main__":
    train()