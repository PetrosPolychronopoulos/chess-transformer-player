import chess
import random
import re
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


class TransformerPlayer(Player):

    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")

    def __init__(self, name: str):
        super().__init__(name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(1)

        self.model_id = "2pp/chess-smollm-1000steps"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.move_counter = 0  # IMPORTANT

    # -------------------------
    # Material evaluation
    # -------------------------
    def _material_score(self, board):
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        return score

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen):
        return f"FEN: {fen}\nMove:"

    # -------------------------
    # Extract UCI
    # -------------------------
    def _extract_move(self, text):
        match = self.UCI_REGEX.search(text)
        return match.group(1) if match else None

    # -------------------------
    # Main
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        board = chess.Board(fen)

        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        self.move_counter += 1

        # Only move
        if len(legal_moves) == 1:
            return legal_moves[0].uci()

        # Mate in 1
        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move.uci()
            board.pop()

        # Material gain
        current_material = self._material_score(board)
        scored = []

        for move in legal_moves:
            board.push(move)
            gain = self._material_score(board) - current_material
            board.pop()
            scored.append((move, gain))

        best_gain = max(g for _, g in scored)
        best_moves = [m for m, g in scored if g == best_gain]

        if best_gain > 0:
            return random.choice(best_moves).uci()

        # LM only every 4 moves
        if self.move_counter % 4 != 0:
            return random.choice(best_moves).uci()

        # LM call
        try:
            prompt = self._build_prompt(fen)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            move = self._extract_move(decoded)

            if move:
                chess_move = chess.Move.from_uci(move)
                if chess_move in legal_moves:
                    return move

        except Exception:
            pass

        return random.choice(best_moves).uci()
