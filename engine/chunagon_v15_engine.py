import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import shogi # python-shogi (v15の特徴量はこれに依存しています)
import torch.optim as optim
import os
import json
import math
import time
import traceback
import copy
import threading
from queue import Queue

# ==========================================
# ⚙️ 設定 (DirectMLチェック)
# ==========================================
try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

# ★修正: モデルパスをv15に変更
MODEL_PATH = "model/chunagon_v15.pt"
VOCAB_PATH = "data/vocab_v2.json"

# GPU設定
if HAS_DIRECTML:
    try:
        DEVICE = torch_directml.device()
        sys.stderr.write(f"Using DirectML Device: {DEVICE}\n")
    except Exception as e:
        sys.stderr.write(f"DirectML error: {e}, falling back to CPU\n")
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cpu")

sys.stderr.write(f"Final Selected Device: {DEVICE}\n")

# ★ バッチ処理・探索設定
TIME_LIMIT = 10.0      # 最大思考時間
BATCH_SIZE = 256       # GPUバッチサイズ
NUM_THREADS = 4        # 探索スレッド数
VIRTUAL_LOSS = 1.0     # 競合ペナルティ
C_PUCT = 1.0           # 探索の幅

# ★ AlphaZero式のノイズ設定 (対局時はノイズを切る)
DIRICHLET_ALPHA = 0.3  
EPSILON = 0.0          # ★修正: 0.0 (ガチ対局モード)

# 定跡設定
BOOK_PATH = "data/dainagon_kakugawaribook.bin"

# ==========================================
# 🧠 モデル定義 (v15: CBAM + 55ch)
# ==========================================
BLOCKS = 10
CHANNELS = 128
INPUT_CHANNELS = 55 # ★55ch

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


# ==========================================
# 🧠 特徴量 & ユーティリティ (v15仕様)
# ==========================================
TBL_NUM = str.maketrans("123456789", "987654321")
TBL_ALP = str.maketrans("abcdefghi", "ihgfedcba")

def rotate_move(move_usi):
    if "*" in move_usi:
        return move_usi[0] + "*" + move_usi[2:].translate(TBL_NUM).translate(TBL_ALP)
    return move_usi[:2].translate(TBL_NUM).translate(TBL_ALP) + \
           move_usi[2:4].translate(TBL_NUM).translate(TBL_ALP) + move_usi[4:]

# ★修正: 欠落していたビットボード展開関数を追加
def bb_to_squares(bb):
    """整数のビットボードから、立っているビットの座標を返すジェネレータ"""
    while bb > 0:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r

# ★修正: 関数名を統一し、is_whiteを内部で判定するように変更
def make_features(board):
    features = np.zeros((INPUT_CHANNELS, 9, 9), dtype=np.float32)
    is_white = (board.turn == shogi.WHITE) # ここで判定
    
    # 1. 盤上の駒 (0-27)
    for sq in range(81):
        p = board.piece_at(sq)
        if p:
            if p.color == board.turn: # 手番側
                idx = p.piece_type - 1
            else: # 相手側
                idx = p.piece_type - 1 + 14
                
            # 視点変換
            target_sq = (80 - sq) if is_white else sq
            y, x = divmod(target_sq, 9)
            features[idx, y, x] = 1.0

    # 2. 持ち駒 (28-41)
    piece_order = [shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER, shogi.GOLD, shogi.BISHOP, shogi.ROOK]
    colors = [(board.turn, 28), (not board.turn, 35)]
    for color, offset in colors:
        hand = board.pieces_in_hand[color]
        for i, p_type in enumerate(piece_order):
            count = hand[p_type]
            if count > 0:
                features[offset + i, :, :] = count / 10.0

    # 3. 大駒の利き (42-45)
    for sq in range(81):
        p = board.piece_at(sq)
        if not p: continue
        pt = p.piece_type
        if pt not in [shogi.ROOK, shogi.PROM_ROOK, shogi.BISHOP, shogi.PROM_BISHOP]:
            continue
            
        is_self = (p.color == board.turn)
        is_rook = (pt == shogi.ROOK or pt == shogi.PROM_ROOK)
        
        if is_self: ch = 42 if is_rook else 43
        else:       ch = 44 if is_rook else 45
        
        attacks_bb = board.attacks_from(pt, sq, board.occupied, p.color)
        
        # ★修正: bb_to_squares を利用
        for t_sq in bb_to_squares(attacks_bb):
            target_sq = (80 - t_sq) if is_white else t_sq
            y, x = divmod(target_sq, 9)
            features[ch, y, x] = 1.0

    # 4. 玉の周辺 (46-49)
    k_sq_self = board.king_squares[board.turn]
    k_sq_opp = board.king_squares[not board.turn]
    
    kings = [(k_sq_self, 46), (k_sq_opp, 48)]
    
    for k_sq, base_ch in kings:
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

    # 5. その他 (50-54)
    if board.is_check(): features[50, :, :] = 1.0
    features[51, :, :] = min(board.move_number / 200.0, 1.0)
    
    for i in range(9):
        features[52, :, i] = i / 8.0 # File
        features[53, i, :] = i / 8.0 # Rank
    
    if board.move_number > 0 and len(board.move_stack) > 0:
        last_move = board.move_stack[-1]
        to_sq = last_move.to_square
        target_sq = (80 - to_sq) if is_white else to_sq
        y, x = divmod(target_sq, 9)
        features[54, y, x] = 1.0

    return features

# ==========================================
# ⚡ GPUバッチ処理マネージャー
# ==========================================
class BatchEvaluator:
    def __init__(self, model):
        self.model = model
        self.queue = Queue()
        self.running = True
        self.worker_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.worker_thread.start()

    def _prediction_loop(self):
        while self.running:
            batch = []
            try:
                req = self.queue.get(timeout=0.005)
                batch.append(req)
            except:
                continue

            while len(batch) < BATCH_SIZE and not self.queue.empty():
                try:
                    batch.append(self.queue.get_nowait())
                except:
                    break

            if not batch: continue
            self._process_batch(batch)

    def _process_batch(self, batch):
        features_list = [req[0] for req in batch]
        x_np = np.stack(features_list)
        x_tensor = torch.from_numpy(x_np).to(DEVICE)
        
        with torch.no_grad():
            policy_logits, value_tensor = self.model(x_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = value_tensor.cpu().numpy()
        
        for i, req in enumerate(batch):
            _, event, result = req
            result['policy'] = policy_probs[i]
            result['value'] = values[i].item()
            event.set()

    def predict(self, features):
        event = threading.Event()
        result = {}
        self.queue.put((features, event, result))
        event.wait()
        return result['policy'], result['value']
    
class BasePlayer:
    def __init__(self):
        pass 
    def usi(self): pass
    def usinewgame(self): pass
    def setoption(self,args): pass
    def isready(self): pass
    def position(self,sfen,usi_moves): pass
    def set_limits(self,btime=None,wtime=None,byoyomi=None,binc=None,
                   winc=None,nodes=None,infinite=False,ponder=False): pass
    def go(self): pass
    def stop(self): pass
    def ponderhit(self,last_limits): pass
    def quit(self): pass
    
    def run(self):
        sys.stdout.reconfigure(line_buffering=True)
        while True:
            try:
                line = sys.stdin.readline()
                if not line: break
                line = line.strip()
                if not line: continue
                
                cmd = line.split(' ', 1)
                
                if cmd[0]=='usi':
                    self.usi()
                    print('usiok',flush=True)
                elif cmd[0]=='isready':
                    self.isready()
                    print('readyok',flush=True)
                elif cmd[0]=='usinewgame':
                    self.usinewgame()
                elif cmd[0]=='position':
                    args=cmd[1].split('moves')
                    self.position(args[0].strip(), args[1].split() if len(args)>1 else [])
                elif cmd[0]=='go':
                    kwargs={}
                    if len(cmd)>1:
                        args=cmd[1].split(' ')
                        if args[0]=='infinite': kwargs['infinite']=True
                        elif args[0]=='ponder': kwargs['ponder']=True
                        for i in range(len(args)-1):
                            if args[i] in ['btime','wtime','byoyomi','binc','winc','nodes']:
                                if args[i+1].isdigit():
                                    kwargs[args[i]]=int(args[i+1])
                    self.set_limits(**kwargs)
                    
                    try:
                        bestmove, _ = self.go()
                        if 'ponder' not in kwargs and 'infinite' not in kwargs:
                            print(f'bestmove {bestmove}', flush=True)
                    except Exception as e:
                        err_msg = traceback.format_exc().replace('\n', ' ')
                        print(f"info string Error in go: {e} {err_msg}", flush=True)
                        print(f"bestmove resign", flush=True)

                elif cmd[0]=='stop':
                    self.stop()
                elif cmd[0]=='quit':
                    self.quit()
                    break
            except Exception as e:
                print(f"info string Error in main loop: {e}", flush=True)

# ==========================================
# 🌳 MCTS用ノード (Noise対応)
# ==========================================
class UctNode:
    def __init__(self, p):
        self.n = 0
        self.w = 0.0
        self.p = p
        self.children = {}
        self.is_expanded = False
        self.lock = threading.Lock()

    def expand_with_moves(self, board, policy_probs, value, vocab_map):
        with self.lock:
            if self.is_expanded: return value
            self.is_expanded = True
            
            legal_moves = list(board.legal_moves)
            is_white = (board.turn == shogi.WHITE)
            
            for move in legal_moves:
                move_usi = move.usi()
                check_str = rotate_move(move_usi) if is_white else move_usi
                idx = vocab_map.get(check_str)
                prob = policy_probs[idx] if idx is not None else 0.00001
                self.children[move_usi] = UctNode(prob)
                
            return value

    def add_dirichlet_noise(self):
        with self.lock:
            child_keys = list(self.children.keys())
            if not child_keys: return
            
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(child_keys))
            
            for i, move_usi in enumerate(child_keys):
                node = self.children[move_usi]
                node.p = (1 - EPSILON) * node.p + EPSILON * noise[i]

    def select_child(self, c_puct=C_PUCT):
        with self.lock:
            best_score = -float('inf')
            best_move = None
            best_child = None
            sqrt_n = math.sqrt(self.n + 1)

            for move_usi, child in self.children.items():
                if child.n == 0: q = 0.5
                else: q = child.w / child.n
                
                u = c_puct * child.p * (sqrt_n / (1 + child.n))
                score = q + u

                if score > best_score:
                    best_score = score
                    best_move = move_usi
                    best_child = child
            
            if best_child:
                best_child.n += VIRTUAL_LOSS
                best_child.w -= VIRTUAL_LOSS
            
            return best_move, best_child

    def revert_virtual_loss(self, move_usi):
        if move_usi in self.children:
            child = self.children[move_usi]
            with self.lock:
                child.n -= VIRTUAL_LOSS
                child.w += VIRTUAL_LOSS

    def update(self, value):
        with self.lock:
            self.n += 1
            self.w += value

# ==========================================
# 🌲 木 (NodeTree)
# ==========================================
class NodeTree:
    def __init__(self):
        self.root = UctNode(1.0)
    def reset(self):
        self.root = UctNode(1.0)
    def advance(self, move_usi):
        if self.root.is_expanded and move_usi in self.root.children:
            self.root = self.root.children[move_usi]
        else:
            self.reset()

# ==========================================
# 🎮 エンジン本体
# ==========================================
class USIEngineV15(BasePlayer):
    def __init__(self):
       super().__init__()
       self.model = None
       self.evaluator = None
       self.vocab = {}
       self.vocab_map = {}
       self.board = shogi.Board()
       self.tree = NodeTree()
       self.history = []
       self.start_time = 0
       self.searching = False
       self.time_limits={}
       self.book={}
       self.book_loaded=False
       self.book_path=BOOK_PATH
       
    def usi(self):
        print("id name Chunagon_v15_Full") # ★名前を更新
        print("id author Youdo")
        print(f'option name BookPath type string default {self.book_path}')
        print('usiok')

    def isready(self):
        self.load_model()

    def load_model(self):
        if self.model is not None: return
        try:
            with open(VOCAB_PATH, 'r') as f:
                raw_vocab = json.load(f)
            
            processed_vocab = {}
            first_key = list(raw_vocab.keys())[0]
            if first_key.isdigit():
                processed_vocab = {int(k): v for k, v in raw_vocab.items()}
                self.vocab_map = {v: k for k, v in processed_vocab.items()}
            else:
                self.vocab_map = raw_vocab
                processed_vocab = {int(v): k for k, v in raw_vocab.items()}
            self.vocab = processed_vocab
            
            self.model = DainagonNet(len(self.vocab)).to(DEVICE)
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            if 'model' in checkpoint: self.model.load_state_dict(checkpoint['model'])
            else: self.model.load_state_dict(checkpoint)
            self.model.eval()

            self.evaluator = BatchEvaluator(self.model)

            if DEVICE.type != 'cpu':
                dummy = torch.zeros(1, INPUT_CHANNELS, 9, 9).to(DEVICE)
                self.model(dummy)
                print("info string GPU Warmup complete.")

            print(f"info string Model loaded (v15). Vocab: {len(self.vocab)}")
        
            if os.path.exists(self.book_path):
                try:
                    with open(self.book_path, "rb") as f:
                        self.book = pickle.load(f)
                    self.book_loaded = True
                    print(f"info string Book loaded from {self.book_path}: {len(self.book)} entries.", flush=True)
                except Exception as e:
                    print(f"info string Error loading book: {e}", flush=True)
            else:
                print(f"info string Book file not found at {self.book_path}", flush=True)

        except Exception as e:
            print(f"info string Error loading model: {e}")
        

    def usinewgame(self):
        self.tree.reset()
        self.history = []

    def position(self, sfen, usi_moves):
        self.board.reset()
        if sfen.startswith("sfen "):
            try: self.board.set_sfen(sfen[5:])
            except: pass
        
        if len(usi_moves) == len(self.history) + 1 and usi_moves[:-1] == self.history:
            self.tree.advance(usi_moves[-1])
        elif usi_moves == self.history:
            pass
        else:
            self.tree.reset()

        for m in usi_moves:
            self.board.push_usi(m)
        self.history = usi_moves
    
    def set_limits(self, **kwargs):
        self.time_limits = kwargs

    def stop(self):
        self.searching = False

    def solve_mate(self, board_original, max_depth=5, timeout=0.5):
        board = copy.deepcopy(board_original)
        start_time = time.time()
        
        def dfs_mate(current_depth):
            if (time.time() - start_time) > timeout: raise TimeoutError 
            if current_depth % 2 != 0: 
                moves = list(board.legal_moves)
                check_moves = []
                for m in moves:
                    board.push(m)
                    if board.is_check():
                        if board.is_checkmate():
                            board.pop(); return m
                        board.pop(); check_moves.append(m)
                    else: board.pop()
                for m in check_moves:
                    board.push(m)
                    if current_depth > 1:
                        if dfs_mate(current_depth - 1):
                            board.pop(); return m
                    board.pop()
                return None
            else:
                moves = list(board.legal_moves)
                if not moves: return True 
                for m in moves:
                    board.push(m)
                    if not dfs_mate(current_depth - 1):
                        board.pop(); return False
                    board.pop()
                return True

        try:
            for d in range(1, max_depth + 1, 2):
                result = dfs_mate(d)
                if result: return result.usi() if hasattr(result, 'usi') else str(result)
        except TimeoutError: return None
        return None
    
    def setoption(self, args):
        if len(args) >= 4 and args[1] == "BookPath" and args[2] == "value":
            self.book_path = args[3]
            print(f"info string BookPath changed to: {self.book_path}", flush=True)

    def go(self):
        if not self.model: return "resign", None

        # 0. Book
        if self.book_loaded:
            sfen_key = self.board.sfen()
            key = " ".join(sfen_key.split(" ")[:-1])
            if key in self.book:
                book_move = self.book[key]
                print(f"info string Book hit: {book_move}", flush=True)
                return book_move, None

        # 1. 1手詰め
        try:
            mate_1 = self.solve_mate(self.board, max_depth=1, timeout=0.05)
            if mate_1:
                print(f"info string Mate Found (1-move): {mate_1}", flush=True)
                return mate_1, None
        except: pass

        # 2. 評価値計算
        features = make_features(self.board)
        _, root_value = self.evaluator.predict(features)
        
        safe_value = max(0.0001, min(0.9999, root_value))
        current_cp = int(-600 * math.log(1.0 / safe_value - 1.0))
        
        # 3. 詰将棋
        if current_cp >= 2000:
            mate_move = self.solve_mate(self.board, max_depth=5, timeout=3.0)
            if mate_move: return mate_move, None
        elif current_cp >= 1000:
             mate_move = self.solve_mate(self.board, max_depth=3, timeout=0.5)
             if mate_move: return mate_move, None

        # 4. 時間管理
        self.start_time = time.time()
        if self.board.turn == shogi.BLACK:
            my_time_ms = self.time_limits.get('btime', 0)
            my_inc_ms = self.time_limits.get('binc', 0)
        else:
            my_time_ms = self.time_limits.get('wtime', 0)
            my_inc_ms = self.time_limits.get('winc', 0)
        byoyomi_ms = self.time_limits.get('byoyomi', 0)
        margin_ms = 500
        
        if byoyomi_ms > 0:
            base_time = (byoyomi_ms - margin_ms) / 1000.0
        else:
            safe_time = max(0, my_time_ms - margin_ms)
            base_time = (safe_time / 20 + my_inc_ms) / 1000.0
        
        base_time = max(0.2, base_time)
        if my_time_ms > 0:
            hard_limit = (my_time_ms - margin_ms) / 1000.0
            base_time = min(base_time, hard_limit)
        else:
            hard_limit = base_time
        max_extend_time = min(base_time * 2.0, hard_limit)
        
        print(f"info string Time: Base={base_time:.2f}s, MaxExtend={max_extend_time:.2f}s", flush=True)

        # 5. 探索
        self.searching = True
        threads = []
        for _ in range(NUM_THREADS):
            t = threading.Thread(target=self.search_worker)
            t.start()
            threads.append(t)
        
        while self.searching:
            elapsed = time.time() - self.start_time
            root = self.tree.root

            if elapsed >= hard_limit:
                self.searching = False
                break
            
            if elapsed >= base_time:
                should_extend = False
                if root.n > 100 and root.children:
                    children = list(root.children.values())
                    children.sort(key=lambda c: c.n, reverse=True)
                    if len(children) >= 2:
                        best_node = children[0]
                        second_node = children[1]
                        if second_node.n > best_node.n * 0.5:
                            if elapsed < max_extend_time:
                                should_extend = True
                if not should_extend:
                    self.searching = False
                    break
            
            if elapsed > base_time * 0.5 and root.n > 500:
                children = list(root.children.values())
                children.sort(key=lambda c: c.n, reverse=True)
                if len(children) >= 2:
                    if children[0].n > children[1].n * 3.0:
                        self.searching = False
                        break

            time.sleep(0.1)
            self.print_info()

        for t in threads:
            t.join()
        
        return self.get_best_move(), None
        
    def print_info(self):
        root = self.tree.root
        elapsed = time.time() - self.start_time
        if root.n == 0: return
        if not root.children: return

        pv_moves = []
        curr_node = root
        try:
            for _ in range(20):
                if not curr_node.children: break
                best_move, best_child = max(curr_node.children.items(), key=lambda item: item[1].n)
                if best_child.n == 0: break
                pv_moves.append(best_move)
                curr_node = best_child
        except Exception: pass
        
        if not pv_moves: return
        pv_str = " ".join(pv_moves)
        
        if pv_moves[0] in root.children:
            best_child = root.children[pv_moves[0]]
            win_rate = best_child.w / best_child.n if best_child.n > 0 else 0.5
        else: return

        win_rate = max(0.0001, min(0.9999, win_rate))
        cp = int(-600 * math.log(1.0 / win_rate - 1.0))
        nps = int(root.n / (elapsed + 0.001))
        
        print(f"info nodes {root.n} score cp {cp} pv {pv_str} nps {nps} time {int(elapsed*1000)}", flush=True)
        
    def get_best_move(self):
        root = self.tree.root
        if not root.children: return "resign"
        best_move = max(root.children.items(), key=lambda item: item[1].n)[0]
        return best_move

    def search_worker(self):
        while self.searching:
            root = self.tree.root
            if not root.is_expanded:
                features = make_features(self.board) # ★引数を修正
                p, v = self.evaluator.predict(features)
                root.expand_with_moves(self.board, p, v, self.vocab_map)
                if EPSILON > 0: root.add_dirichlet_noise() # ノイズは設定次第
            
            node = root
            try:
                if hasattr(self.board, 'copy'): sim_board = self.board.copy()
                else: sim_board = pickle.loads(pickle.dumps(self.board))
            except: sim_board = copy.deepcopy(self.board)
            
            path = [node]
            moves_to_leaf = []
            
            while node.is_expanded and node.children:
                move_str, next_node = node.select_child()
                if not next_node: break
                sim_board.push_usi(move_str)
                node = next_node
                path.append(node)
                moves_to_leaf.append(move_str)
                if sim_board.is_game_over(): break
            
            if sim_board.is_game_over():
                leaf_value = 0.0 if sim_board.is_checkmate() else 0.5
            else:
                features = make_features(sim_board) # ★引数を修正
                policy, value = self.evaluator.predict(features)
                leaf_value = node.expand_with_moves(sim_board, policy, value, self.vocab_map)

            curr_val = 1.0 - leaf_value
            node.update(curr_val)
            curr_val = 1.0 - curr_val
            
            for i in reversed(range(len(moves_to_leaf))):
                path[i].revert_virtual_loss(moves_to_leaf[i])
                path[i].update(curr_val)
                curr_val = 1.0 - curr_val

if __name__ == "__main__":
    engine = USIEngineV15()
    engine.run()