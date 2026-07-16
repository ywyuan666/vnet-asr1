"""调试 Transducer RNN-T loss"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torchaudio

# 模拟数据
B, T_enc, U, V = 2, 6, 5, 10
blank = 0

logits = torch.randn(B, T_enc, U, V, requires_grad=True)
targets = torch.randint(1, V, (B, U-1))
logit_lengths = torch.tensor([T_enc, T_enc], dtype=torch.int32)
target_lengths = torch.tensor([U-1, U-1], dtype=torch.int32)

print(f"logits shape: {logits.shape}")
print(f"targets shape: {targets.shape}")
print(f"logit_lengths: {logit_lengths}")
print(f"target_lengths: {target_lengths}")

try:
    loss = torchaudio.functional.rnnt_loss(
        logits=logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=blank,
        reduction="mean",
    )
    print(f"RNN-T loss: {loss.item()}")
    loss.backward()
    print("Backward: OK")
except Exception as e:
    print(f"RNN-T loss error: {e}")
    print(f"Error type: {type(e).__name__}")
    
    # 尝试不同的参数
    try:
        loss = torchaudio.functional.rnnt_loss(
            logits=logits.float(),
            targets=targets.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=blank,
            reduction="mean",
        )
        print(f"Second attempt loss: {loss.item()}")
    except Exception as e2:
        print(f"Second attempt error: {e2}")
        
        # 尝试 warp-transducer style
        try:
            import torch.nn.functional as F
            loss_fn = torchaudio.transforms.RNNTLoss(blank=blank, reduction="mean")
            loss = loss_fn(logits, targets, logit_lengths, target_lengths)
            print(f"Transform approach loss: {loss.item()}")
        except Exception as e3:
            print(f"Transform approach error: {e3}")
