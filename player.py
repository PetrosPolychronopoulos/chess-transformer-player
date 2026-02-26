import chess
import random
import re
import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player


class TransformerPlayer(Player):
    """
    Transformer-based chess player.
    Safe for championship execution.
    """

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "Student",
        model_id: str = "1efd/chess-transformer",
        temperature: float = 0.7,
        max_new_tokens: int = 8,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Lazy-loaded
        self.tokenizer = None
        self.model = None

    # -------------------------
    # Lazy model loading
    # -------------------------
    def _load_model(self):
        if self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

                self.model.to(self.device)
                self.model.eval()

            except Exception:
                # If loading fails, keep model None
                self.model = None
                self.tokenizer = None

    # -------------------------
    # Extract UCI move
    # -------------------------
    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    # -------------------------
    # Random fallback
    # -------------------------
    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves).uci() if legal_moves else None

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        if not legal_moves:
            return None

        # Try loading model if not loaded
        self._load_model()

        # If model failed to load → fallback immediately
        if self.model is None or self.tokenizer is None:
            return random.choice(legal_moves)

        prompt = f"FEN: {fen}\nMove:"

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            move = self._extract_move(decoded)

            if move and move in legal_moves:
                return move

        except Exception:
            pass

        # Safe fallback
        return random.choice(legal_moves)
