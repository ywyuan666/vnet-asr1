# -*- coding: utf-8 -*-
"""
LM Rescoring for Attention Decoder
===================================
对 Attention 解码的 N-best 候选使用外部语言模型重打分。

用法：
    from tools.lm_rescore import LmRescorer
    rescorer = LmRescorer("data/lm/aishell_4gram.klm", lm_weight=0.3)
    best = rescorer.rescore(hypotheses, scores)
"""

import sys
from typing import List, Optional, Tuple


class LmRescorer:
    """
    基于 KenLM 的语言模型重打分器。

    对 Attention 解码器输出的 Top-K 候选进行 LM rescoring，
    然后根据 AM score + LM score 重新排序。

    Combined Score = AM_score + lm_weight * LM_score

    Args:
        model_path: KenLM binary 模型路径
        lm_weight: LM 权重 (0.0 ~ 1.0)
        blank_id: blank token ID
        sos_id: start-of-sentence token ID
        eos_id: end-of-sentence token ID
        id2token: token ID 到字符的映射列表（可选）
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        lm_weight: float = 0.3,
        blank_id: int = 0,
        sos_id: int = 2,
        eos_id: Optional[int] = None,
        id2token: Optional[List[str]] = None,
    ):
        self.lm_weight = lm_weight
        self.blank_id = blank_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.id2token = id2token

        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """加载 KenLM 模型。"""
        try:
            import kenlm
        except ImportError:
            print("[警告] KenLM 未安装，将使用简单的字符频率作为后备 LM")
            self.model = None
            return

        self.model = kenlm.Model(model_path)
        print(
            f"[LM Rescorer] 加载 KenLM 模型: {model_path} (order={self.model.order})"
        )

    def _tokens_to_text(self, token_ids: List[int]) -> str:
        """Token IDs to Chinese text string."""
        if self.id2token:
            chars = []
            for tid in token_ids:
                if tid in (self.blank_id, self.sos_id, self.eos_id):
                    continue
                if 0 <= tid < len(self.id2token):
                    chars.append(self.id2token[tid])
            return "".join(chars)
        return str(token_ids)

    def _lm_score(self, token_ids: List[int]) -> float:
        """Calculate LM score for a sequence of token IDs."""
        if self.model is None:
            # Fallback: length penalty only
            return -len(token_ids) * 0.1

        # Convert tokens to text and score
        text = self._tokens_to_text(token_ids)
        if not text.strip():
            return 0.0

        # KenLM: space-separated Chinese characters -> log10 score
        spaced = " ".join(text)
        return self.model.score(spaced, bos=True, eos=True)

    def rescore(
        self,
        hypotheses: List[List[int]],
        am_scores: List[float],
        lm_weight: Optional[float] = None,
    ) -> Tuple[List[int], float]:
        """
        对 N-best 候选用 LM 重打分。

        Args:
            hypotheses: N-best token ID lists
            am_scores: 对应的声学模型分数（log probability）
            lm_weight: LM 权重，None 则使用默认值

        Returns:
            (best_hypothesis, best_score)
        """
        if lm_weight is None:
            lm_weight = self.lm_weight

        best_score = -float("inf")
        best_hyp = hypotheses[0]

        for hyp, am_score in zip(hypotheses, am_scores):
            lm_score = self._lm_score(hyp)
            combined = am_score + lm_weight * lm_score

            if combined > best_score:
                best_score = combined
                best_hyp = hyp

        return best_hyp, best_score
