import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import math
import cshogi
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ 設定 (Dainagon v16: 1億局面・RAM全乗せ爆速モード)
# ==========================================
DEVICE = torch_directml.device()
DATA_FILE = "data/Suisho10Mn_psv/Suisho10Mn_psv.bin"
VOCAB_FILE = "data/vocab_v2.json"
MODEL_SAVE_PATH = "model/chunagon_v16.pt" 
GRAPH_SAVE_PATH = "model/learning_curve_v16.png" 

INPUT_CHANNELS = 55 
BLOCKS = 20      
CHANNELS = 256   

EPOCHS = 1       
LEARNING_RATE = 0.0005
TARGET_BATCH_SIZE = 2048 
PHYSICAL_BATCH_SIZE = 512 # VRAMに余裕があれば 1024 でも可
ACCUM_STEPS = TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE
PLOT_INTERVAL_SEC = 1800 

def plot_learning_curve(loss_history, save_path):
    if not loss_history: return
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.title('Dainagon v16 Learning Curve')
    plt.xlabel('Steps (x Accumulation)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# ==========================================
# ⚡ 特徴量生成 (C++完全同期・安全版)
# ==========================================
def make_features_cshogi(board: cshogi.Board):
    is_white = (board.turn == cshogi.WHITE)
    features = np.zeros((INPUT_CHANNELS, 9, 9), dtype=np.float32)
    
    PT_TO_CH = {
        1: 0, 2: 1, 3: 2, 4: 3, 7: 4, 5: 5, 6: 6, 8: 7, 
        9: 8, 10: 9, 11: 10, 12: 11, 13: 12, 14: 13
    }
    
    k_sq_self = -1
    k_sq_opp = -1
    pieces = board.pieces
    
    # 1. 盤上の駒
    for sq in range(81):
        p = pieces[sq]
        if p == 0: continue
        
        pt = p & 15
        color = p >> 4
        
        if pt == 8: # KING
            if color == board.turn: k_sq_self = sq
            else: k_sq_opp = sq

        idx = PT_TO_CH[pt]
        if color != board.turn: idx += 14
            
        target_sq = (80 - sq) if is_white else sq
        y, x = divmod(target_sq, 9)
        features[idx, y, x] = 1.0
        
    # 2. 持ち駒
    cshogi_hand_idx = [0, 1, 2, 3, 6, 4, 5] 
    colors = [(board.turn, 28), (board.turn ^ 1, 35)]
    for color, offset in colors:
        for i, idx in enumerate(cshogi_hand_idx):
            count = board.pieces_in_hand[color][idx]
            if count > 0: features[offset + i, :, :] = count / 10.0

    # 3. 大駒の利き
    DIRECTIONS_ROOK = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    DIRECTIONS_BISHOP = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for sq in range(81):
        p = pieces[sq]
        if p == 0: continue
        pt = p & 15
        if pt not in [5, 6, 13, 14]: continue
            
        color = p >> 4
        is_self = (color == board.turn)
        is_rook = (pt == 6 or pt == 14)
        ch = (42 if is_rook else 43) if is_self else (44 if is_rook else 45)
        
        dirs_long = DIRECTIONS_ROOK if is_rook else DIRECTIONS_BISHOP
        dirs_short = DIRECTIONS_BISHOP if is_rook else DIRECTIONS_ROOK
        
        y_start, x_start = divmod(sq, 9)
        
        for dy, dx in dirs_long:
            cy, cx = y_start + dy, x_start + dx
            while 0 <= cy < 9 and 0 <= cx < 9:
                c_sq = cy * 9 + cx
                target_sq = (80 - c_sq) if is_white else c_sq
                ty, tx = divmod(target_sq, 9)
                features[ch, ty, tx] = 1.0
                if pieces[c_sq] != 0: break
                cy += dy; cx += dx
                
        if pt == 13 or pt == 14:
            for dy, dx in dirs_short:
                cy, cx = y_start + dy, x_start + dx
                if 0 <= cy < 9 and 0 <= cx < 9:
                    c_sq = cy * 9 + cx
                    target_sq = (80 - c_sq) if is_white else c_sq
                    ty, tx = divmod(target_sq, 9)
                    features[ch, ty, tx] = 1.0

    # 4. 玉の周辺
    for k_sq, base_ch in [(k_sq_self, 46), (k_sq_opp, 48)]:
        if k_sq == -1: continue 
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

    # 5. その他
    if board.is_check(): features[50, :, :] = 1.0
    features[51, :, :] = min(board.move_number / 200.0, 1.0)
    for i in range(9):
        features[52, :, i] = i / 8.0 
        features[53, i, :] = i / 8.0 
        
    return features

# ==========================================
# 🌊 PSVデータローダー (memmap安全・高速版)
# ==========================================
class PSVDataset(Dataset):
    def __init__(self, psv_file, vocab):
        self.vocab = vocab
        self.psv_file = psv_file 
        
        # 1レコード40バイトなので、ファイルサイズから局面数を計算
        file_size = os.path.getsize(psv_file)
        actual_total = file_size // 40
        
        # ★ここを 15,000,000 に制限します
        self.length = min(15000000, actual_total)
        
        print(f"📥 試行モード: 15,000,000 局面をロードします（全データ中から先頭15Mを使用）")
        self.records = None 
        
        self.TBL_NUM = str.maketrans("123456789", "987654321")
        self.TBL_ALP = str.maketrans("abcdefghi", "ihgfedcba")

    def _rotate_move(self, move_usi):
        if "*" in move_usi: 
            return move_usi[0] + "*" + move_usi[2:].translate(self.TBL_NUM).translate(self.TBL_ALP)
        return move_usi[:2].translate(self.TBL_NUM).translate(self.TBL_ALP) + \
               move_usi[2:4].translate(self.TBL_NUM).translate(self.TBL_ALP) + move_usi[4:]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.records is None:
            # memmapを使えば、ワーカーごとに開かれてもメモリを共有するので爆発しない
            self.records = np.memmap(self.psv_file, dtype=cshogi.PackedSfenValue, mode='r')
            
        while True:
            try:
                record = self.records[idx]
                
                # ★修正1: メモリレイアウト崩れを防ぐため、確実にコピーして独立したバイト列にする
                sfen_bytes = np.array(record['sfen'], copy=True)
                
                board = cshogi.Board()
                # ★ここでハフマンコードの解凍が行われる
                board.set_hcp(sfen_bytes)
                is_white = (board.turn == cshogi.WHITE)
                
                x = make_features_cshogi(board)
                
                move_usi = cshogi.move_to_usi(record['move'])
                label = self._rotate_move(move_usi) if is_white else move_usi
                move_id = self.vocab.get(label, -100)
                
                # スコアの反転とスケーリング
                score_cp = record['score']
                if is_white:
                    score_cp = -score_cp
                    
                score_cp = max(min(score_cp, 30000), -30000) 
                win_rate = 1.0 / (1.0 + math.exp(-score_cp / 600.0))
                
                # ★修正2: gamePly (Pが大文字)
                try: move_count = float(record['gamePly'])
                except: move_count = 100.0
                
                return (torch.from_numpy(x), 
                        torch.tensor(move_id, dtype=torch.long), 
                        torch.tensor(win_rate, dtype=torch.float32), 
                        torch.tensor(move_count, dtype=torch.float32))

            except Exception as e:
                # ★修正3: 万が一データが壊れていて incorrect Huffman code が出ても、
                # クラッシュせずに次の局面を読みに行く無敵の防御壁
                idx = (idx + 1) % self.length

# ==========================================
# 🧠 モデル定義 (20 Blocks / 256 Ch)
# ==========================================
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
        self.value_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256), nn.ReLU(),
            nn.Dropout(p=0.3), nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        h = self.conv_input(x)
        for block in self.res_blocks: h = block(h)
        h = h.view(-1, self.fc_input_dim)
        h = self.dropout(h)
        return self.policy_head(h), self.value_head(h)

def train():
    os.makedirs("model", exist_ok=True)
    with open(VOCAB_FILE, 'r') as f: vocab = json.load(f)
    
    dataset = PSVDataset(DATA_FILE, vocab)
    
    # RAM全乗せなので num_workers は 0〜4 程度で十分高速です。
    loader = DataLoader(dataset, batch_size=PHYSICAL_BATCH_SIZE, num_workers=4, shuffle=True)

    model = DainagonNet(len(vocab)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"🔄 Checkpointを発見: {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        
    model.to(DEVICE)

    print("=== Training v16 Started (RAM-Loaded / Fully Fixed) ===")
    
    loss_history = []
    last_plot_time = time.time() 

    for epoch in range(EPOCHS):
        model.train()
        
        total_steps = len(dataset) // PHYSICAL_BATCH_SIZE
        pbar = tqdm(loader, total=total_steps, desc=f"Epoch {epoch+1}")
        
        epoch_running_loss = 0.0
        step_count = 0

        for i, (x, t_m, t_v, move_count) in enumerate(pbar):
            x = x.to(DEVICE)
            t_m = t_m.to(DEVICE)
            t_v = t_v.to(DEVICE)
            move_count = move_count.to(DEVICE)
            
            p, v = model(x)
            
            loss_p = nn.CrossEntropyLoss(ignore_index=-100)(p, t_m)
            
            # ★修正3: squeeze(1)でバッチ次元のバグを防止 ＆ 終盤(80手以降)の重み付けを復活
            loss_v = ((v.squeeze(1) - t_v)**2 * torch.where(move_count > 80, 2.0, 1.0)).mean()
            
            loss = (loss_p + loss_v) / ACCUM_STEPS
            loss.backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                current_loss = loss.item() * ACCUM_STEPS
                loss_history.append(current_loss)
                
                epoch_running_loss += current_loss
                step_count += 1
                avg_loss = epoch_running_loss / step_count
                
                pbar.set_postfix({
                    "Loss": f"{current_loss:.4f}", 
                    "Avg": f"{avg_loss:.4f}" 
                })

                current_time = time.time()
                if current_time - last_plot_time > PLOT_INTERVAL_SEC:
                    plot_learning_curve(loss_history, GRAPH_SAVE_PATH)
                    torch.save({'model': model.state_dict(), 'vocab': vocab}, MODEL_SAVE_PATH)
                    print(f"\n⏰ {time.strftime('%H:%M:%S')} - Periodic save & plot updated.")
                    last_plot_time = current_time

        torch.save({'model': model.state_dict(), 'vocab': vocab}, MODEL_SAVE_PATH)
        plot_learning_curve(loss_history, GRAPH_SAVE_PATH)
        print(f"✅ Epoch {epoch+1} 完了！")

if __name__ == "__main__":
    train()