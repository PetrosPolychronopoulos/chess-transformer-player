import chess
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Deterministic transformer-based chess player.
    Uses legal-move log-probability ranking.
    """

    def __init__(
        self,
        name: str = "Student",
        model_id: str = "distilgpt2",
    ):
        super().__init__(name)

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None

        torch.set_num_threads(2)

    # -------------------------
    # Lazy loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        return f"Position: {fen}\nBest move:"

    # -------------------------
    # Log-prob scoring
    # -------------------------
    def _score_move(self, prompt: str, move_uci: str) -> float:
        full_text = prompt + move_uci

        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        move_tokens = self.tokenizer(move_uci, return_tensors="pt")["input_ids"][0]
        move_len = move_tokens.size(0)

        total_log_prob = 0.0

        for i in range(move_len):
            token_id = shift_labels[0, -move_len + i]
            total_log_prob += log_probs[0, -move_len + i, token_id].item()

        return total_log_prob

    # -------------------------
    # Candidate filtering
    # -------------------------
    def _select_candidates(self, board: chess.Board):
        legal_moves = list(board.legal_moves)

        captures = [m for m in legal_moves if board.is_capture(m)]
        checks = [m for m in legal_moves if board.gives_check(m)]

        candidates = captures + checks

        if len(candidates) < 8:
            others = [m for m in legal_moves if m not in candidates]
            candidates += others[: 8 - len(candidates)]

        if not candidates:
            candidates = legal_moves

        return candidates

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._load_model()
        except Exception:
            return None

        board = chess.Board(fen)
        candidates = self._select_candidates(board)

        if not candidates:
            return None

        prompt = self._build_prompt(fen)

        best_score = -float("inf")
        best_move = candidates[0]

        for move in candidates:
            try:
                score = self._score_move(prompt, move.uci())
                if score > best_score:
                    best_score = score
                    best_move = move
            except Exception:
                continue

        return best_move.uci()
