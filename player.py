import chess
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Fast deterministic transformer-based chess player.
    Uses next-token log-probability ranking over legal moves.
    Optimized for CPU.
    """

    def __init__(
        self,
        name: str = "Student",
        model_id: str = "distilgpt2",   # general LM (NOT chess-specific)
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
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        try:
            self._load_model()
        except Exception:
            return None

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        prompt = self._build_prompt(fen)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # logits for next token only
        logits = outputs.logits[:, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        best_score = -float("inf")
        best_move = legal_moves[0]

        for move in legal_moves:
            uci = move.uci()

            # take first token of move
            tokens = self.tokenizer(uci, add_special_tokens=False)["input_ids"]

            if not tokens:
                continue

            first_token = tokens[0]
            score = log_probs[0, first_token].item()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci()
