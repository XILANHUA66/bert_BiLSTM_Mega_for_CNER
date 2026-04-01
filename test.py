"""
本地 NER 推理脚本
用法:
    python ner_predict.py
    python ner_predict.py --model ./final_model --text "北京大学的李明今天去了上海。"

目录结构（从 Kaggle 下载后保持原样即可）:
    final_model/
        model.pt
        config.json
        vocab.txt
        tokenizer_config.json
        special_tokens_map.json
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast, BertConfig


# ─────────────────────────────────────────────────────────────────────────────
# 模型定义（与训练脚本保持完全一致）
# ─────────────────────────────────────────────────────────────────────────────

class EMALayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state
        self.alpha = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.delta = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        B, L, D = x.shape
        alpha = torch.sigmoid(self.alpha)
        delta = torch.sigmoid(self.delta)
        h = x.new_zeros(B, D, self.d_state)
        outputs = []
        for t in range(L):
            xt = x[:, t, :]
            h  = alpha * h + (1 - alpha) * xt.unsqueeze(-1)
            outputs.append((h * delta).sum(-1))
        return torch.stack(outputs, dim=1) * self.gamma + self.beta


class MEGABlock(nn.Module):
    def __init__(self, d_model, d_attn=256, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.ema    = EMALayer(d_model)
        self.q_proj = nn.Linear(d_model, d_attn, bias=False)
        self.k_proj = nn.Linear(d_model, d_attn, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate   = nn.Linear(d_model, d_model, bias=True)
        self.out    = nn.Linear(d_model, d_model, bias=False)
        self.scale  = math.sqrt(d_attn)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ffn, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.norm1(x)
        ema_out = self.ema(x)
        Q = self.q_proj(ema_out)
        K = self.k_proj(ema_out)
        V = self.v_proj(ema_out)
        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        if attention_mask is not None:
            mask = (1 - attention_mask).bool().unsqueeze(1)
            attn = attn.masked_fill(mask, float('-inf'))
        attn = self.drop(F.softmax(attn, dim=-1))
        ctx  = torch.bmm(attn, V) * torch.sigmoid(self.gate(ema_out))
        x    = self.drop(self.out(ctx)) + residual
        x    = x + self.drop(self.ffn(self.norm2(x)))
        return x


class BertBiLSTMMEGAForNER(nn.Module):
    def __init__(self, bert_model, num_labels, alpha_loss=0.5):
        super().__init__()
        self.num_labels  = num_labels
        self.alpha_loss  = alpha_loss
        self.bert        = bert_model
        H                = bert_model.config.hidden_size
        self.dropout     = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(
            input_size=H, hidden_size=H // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3,
        )
        self.classifier_lstm = nn.Linear(H, num_labels)
        self.mega            = MEGABlock(d_model=H, d_attn=256,
                                         d_ffn=H * 4, dropout=0.1)
        self.classifier_mega = nn.Linear(H, num_labels)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, **kwargs):
        seq = self.dropout(
            self.bert(input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).last_hidden_state
        )
        lstm_out, _ = self.bilstm(seq)
        logits1     = self.classifier_lstm(lstm_out)
        mega_out    = self.mega(seq, attention_mask=attention_mask)
        logits2     = self.classifier_mega(mega_out)
        return (logits1 + logits2) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# MSRA-NER 标签定义（与训练数据集顺序一致）
# ─────────────────────────────────────────────────────────────────────────────

LABEL_LIST = [
    "O",
    "B-LOC", "I-LOC",   # 人名
    "B-ORG", "I-ORG",   # 机构
    "B-PER", "I-PER",   # 地名
]

# 标签对应的中文说明（用于展示）
ENTITY_ZH = {"PER": "人名", "ORG": "机构", "LOC": "地名"}


# ─────────────────────────────────────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: str, device: torch.device):
    config     = BertConfig.from_pretrained(model_dir)
    bert_model = BertModel(config)                    # 空壳，权重从 model.pt 加载
    model      = BertBiLSTMMEGAForNER(bert_model, num_labels=len(LABEL_LIST))

    state_dict = torch.load(
        f"{model_dir}/model.pt",
        map_location=device,
        weights_only=True,          # 安全加载，避免 pickle 风险
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# BIO 标签序列 → 实体列表
# ─────────────────────────────────────────────────────────────────────────────

def decode_entities(tokens: list[str], labels: list[str]) -> list[dict]:
    """
    将 token 序列和对应 BIO 标签解析成实体列表。
    返回: [{"text": "李明", "type": "人名", "start": 5, "end": 7}, ...]
    """
    entities = []
    current_text  = []
    current_type  = None
    current_start = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith("B-"):
            # 先保存上一个实体
            if current_text:
                entities.append({
                    "text":  "".join(current_text),
                    "type":  ENTITY_ZH.get(current_type, current_type),
                    "start": current_start,
                    "end":   i,
                })
            current_text  = [token]
            current_type  = label[2:]
            current_start = i

        elif label.startswith("I-") and current_type == label[2:]:
            current_text.append(token)

        else:
            # O 标签或类型不连续，结束当前实体
            if current_text:
                entities.append({
                    "text":  "".join(current_text),
                    "type":  ENTITY_ZH.get(current_type, current_type),
                    "start": current_start,
                    "end":   i,
                })
            current_text  = []
            current_type  = None
            current_start = None

    # 处理末尾残留实体
    if current_text:
        entities.append({
            "text":  "".join(current_text),
            "type":  ENTITY_ZH.get(current_type, current_type),
            "start": current_start,
            "end":   len(tokens),
        })

    return entities


# ─────────────────────────────────────────────────────────────────────────────
# 推理主函数
# ─────────────────────────────────────────────────────────────────────────────

def predict(text: str, model, tokenizer, device: torch.device) -> list[dict]:
    # 分词（逐字切分，保留 offset 对齐）
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,   # 记录每个 token 对应原文位置
        truncation=True,
        max_length=512,
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)   # (1, L, C)

    pred_ids = logits[0].argmax(dim=-1).cpu().tolist()  # (L,)

    # 去掉 [CLS] 和 [SEP]，只保留实际 token
    tokens, pred_labels = [], []
    for idx, (token_id, label_id, offset) in enumerate(
            zip(input_ids[0].tolist(), pred_ids, offset_mapping)):
        start, end = offset
        if start == end:          # 特殊 token（[CLS]/[SEP]/[PAD]）
            continue
        tokens.append(text[start:end])
        pred_labels.append(LABEL_LIST[label_id])

    entities = decode_entities(tokens, pred_labels)
    return entities


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NER 命名实体识别推理")
    parser.add_argument("--model", default="./final_model",
                        help="final_model 目录路径（默认: ./final_model）")
    parser.add_argument("--text",  default=None,
                        help="要识别的文本；不传则进入交互模式")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"加载模型: {args.model} ...")

    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    model     = load_model(args.model, device)
    print("模型加载完成\n")

    def run_once(text: str):
        if not text.strip():
            return
        entities = predict(text, model, tokenizer, device)
        print(f"输入: {text}")
        if not entities:
            print("未识别到命名实体")
        else:
            print(f"识别到 {len(entities)} 个实体:")
            for e in entities:
                print(f"  [{e['type']}] {e['text']}"
                      f"  (字符位置 {e['start']}~{e['end']})")
        print()

    if args.text:
        run_once(args.text)
    else:
        print("进入交互模式，输入文本后回车，输入 q 退出\n")
        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("q", "quit", "exit"):
                break
            run_once(text)


if __name__ == "__main__":
    main()