"""CPU 全量 AISHELL-1 训练"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
import train, torch, json
from model.conformer_ctc_attn_transducer import ConformerCTCATTNTransducer

# 训练参数 (CPU 适配)
BATCH = 4; EPOCH = 30; DM = 64; BLOCKS = 2

print(f"=== CPU AISHELL-1 Training ===")
print(f"batch={BATCH} epoch={EPOCH} d_model={DM} blocks={BLOCKS}")

# 加载字典
with open('data/aishell/units.txt', encoding='utf-8') as f:
    vocab = {}
    for line in f:
        p = line.strip().split()
        if len(p) >= 2: vocab[p[0]] = int(p[1])
sos_id = len(vocab) - 1
print(f"Vocab: {len(vocab)} tokens")

# 加载 CMVN
with open('data/aishell/global_cmvn') as f:
    cmvn_raw = json.load(f)
fn = cmvn_raw['frame_num']
cmvn_mean = torch.tensor(cmvn_raw['mean_stat']) / fn
cmvn_var = torch.tensor(cmvn_raw['var_stat']) / fn - cmvn_mean * cmvn_mean
cmvn_std = cmvn_var.sqrt()
cmvn_data = {'mean_stat': cmvn_raw['mean_stat'], 'var_stat': cmvn_raw['var_stat'], 'frame_num': fn}

# 模型
model = ConformerCTCATTNTransducer(
    vocab_size=len(vocab), d_model=DM, enc_blocks=BLOCKS,
    ctc_weight=1.0, attn_weight=0.0, trans_weight=0.0
)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# 数据
ds = train.AsrDataset('data/aishell/train/data.list')
dev_ds = train.AsrDataset('data/aishell/dev/data.list')
print(f"Train: {len(ds.items)}, Dev: {len(dev_ds.items)}")

# WarmupLR
d_model = DM; warmup = 500
opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

import time, math
t0 = time.time()
best_cv = float('inf')
os.makedirs('exp/cpu_train', exist_ok=True)

for epoch in range(EPOCH):
    import random; random.shuffle(ds.items)
    el, n = 0, 0
    for i in range(0, len(ds.items), BATCH):
        # LR schedule
        step = epoch * len(ds.items) + i // BATCH + 1
        lr = math.pow(DM, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup, -1.5))
        for pg in opt.param_groups: pg['lr'] = lr

        bi = [ds[j] for j in range(i, min(i+BATCH, len(ds.items)))]
        if len(bi) < 2: continue
        b = train.collate_fn(bi, vocab, sos_id)
        feats, fl, ai, ao, ti, tl = [x.to('cpu') if isinstance(x, torch.Tensor) else x for x in b]
        opt.zero_grad()
        r = model(feats, fl, ai, ao, ti, tl)
        r['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        el += r['loss'].item(); n += 1

    # Validation
    model.eval()
    cv_el, cv_n = 0, 0
    with torch.no_grad():
        for i in range(0, len(dev_ds.items), BATCH):
            bi = [dev_ds[j] for j in range(i, min(i+BATCH, len(dev_ds.items)))]
            if len(bi) < 2: continue
            b = train.collate_fn(bi, vocab, sos_id)
            feats, fl, ai, ao, ti, tl = [x.to('cpu') if isinstance(x, torch.Tensor) else x for x in b]
            r = model(feats, fl, ai, ao, ti, tl)
            cv_el += r['loss'].item(); cv_n += 1
    model.train()

    t_av = el / max(1, n)
    c_av = cv_el / max(1, cv_n)
    elapsed = time.time() - t0
    status = "BEST" if c_av < best_cv else ""
    if c_av < best_cv:
        best_cv = c_av
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'config': {'d_model': DM, 'enc_blocks': BLOCKS}}, 'exp/cpu_train/best.pt')

    print(f'Epoch {epoch+1}/{EPOCH}: train={t_av:.2f} cv={c_av:.2f} | {elapsed/60:.0f}min {status}', flush=True)
    if epoch == 0:
        first_loss = t_av
    last_loss = t_av

drop = (first_loss - best_cv) / first_loss * 100
print(f"\nResult: {first_loss:.1f} -> {best_cv:.1f} (best CV) drop={drop:.0f}% | {time.time()-t0:.0f}s total")
print(f"Model: exp/cpu_train/best.pt")
