import chess
import torch
import re
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament.players import Player


class TransformerPlayer(Player):

    UCI_REGEX = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

    def __init__(
        self,
        name: str = "Student",
        model_id: str = "2pp/chess-smollm-1000steps",
    ):
        super().__init__(name)

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        return f"FEN: {fen}\nMove:"

    # -------------------------
    # Extract UCI
    # -------------------------
    def _extract_uci(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(0) if match else None

    # -------------------------
    # Material fallback
    # -------------------------
    def _material_fallback(self, board: chess.Board) -> str:
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        best_move = None
        best_score = -float("inf")

        for move in board.legal_moves:
            board.push(move)
            score = 0
            for piece_type, value in piece_values.items():
                score += len(board.pieces(piece_type, board.turn)) * value
                score -= len(board.pieces(piece_type, not board.turn)) * value
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci()

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        board = chess.Board(fen)

        if board.is_game_over():
            return None

        prompt = self._build_prompt(fen)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False
            )

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        move_uci = self._extract_uci(generated_text)

        # If LM move is legal → play it
        if move_uci:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    return move_uci
            except:
                pass

        # Fallback → material best move
        return self._material_fallback(board)
