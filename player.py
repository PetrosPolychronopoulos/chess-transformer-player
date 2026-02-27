import chess
import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player


class TransformerPlayer(Player):

    def __init__(
        self,
        name: str = "Student",
        model_id: str = "1efd/chess-transformer",
    ):
        super().__init__(name)

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None

    # -------------------------
    # Safe lazy loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                raise RuntimeError(f"Model loading failed: {e}")

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        # Game over
        if not legal_moves:
            return None

        # Load model (only once)
        self._load_model()

        prompt = f"FEN: {fen}\nMove:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits[0, -1]

            best_move = None
            best_score = float("-inf")

            for move in legal_moves:
                uci = move.uci()

                tokens = self.tokenizer(
                    " " + uci,
                    add_special_tokens=False
                )["input_ids"]

                if not tokens:
                    continue

                token_id = tokens[0]
                score = logits[token_id].item()

                if score > best_score:
                    best_score = score
                    best_move = uci

            # Guaranteed legal (selected from legal_moves)
            if best_move is not None:
                return best_move

        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

        # This should never happen, but for absolute safety:
        # return first legal move (still deterministic, not random)
        return legal_moves[0].uci()
