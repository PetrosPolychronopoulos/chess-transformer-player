import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "ChessTransformer",
        model_id: str = "2pp/chess-smollm-1000steps",
        max_new_tokens: int = 6,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load ONCE
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        # Avoid generation warning
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMove:"

    # -------------------------
    # Extract UCI move
    # -------------------------
    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    # -------------------------
    # Fallback (always legal)
    # -------------------------
    def _random_legal(self, board: chess.Board) -> Optional[str]:
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        board = chess.Board(fen)

        if board.is_game_over():
            return None

        prompt = self._build_prompt(fen)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # deterministic & fast
                )

            decoded = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Remove prompt part
            if decoded.startswith(prompt):
                decoded = decoded[len(prompt):]

            move = self._extract_move(decoded)

            if move:
                try:
                    chess_move = chess.Move.from_uci(move)
                    if chess_move in board.legal_moves:
                        return move
                except:
                    pass

        except Exception:
            pass

        # Fallback
        return self._random_legal(board)
