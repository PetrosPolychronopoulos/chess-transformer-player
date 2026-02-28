import chess
import torch
import re
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

    UCI_REGEX = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

    def __init__(
        self,
        name: str = "Student",
        model_id: str = "2pp/chess-smollm-1000steps",
    ):
        super().__init__(name)

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None

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
    # Material evaluation (relative to side to move)
    # -------------------------
    def _material_score(self, board: chess.Board) -> int:
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        return score if board.turn == chess.WHITE else -score

    # -------------------------
    # Prompt (match training format)
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMove:"

    # -------------------------
    # LM scoring of full move
    # -------------------------
    def _score_move_with_lm(self, fen: str, move_uci: str) -> float:
        prompt = self._build_prompt(fen)
        full_text = prompt + " " + move_uci

        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        input_ids = inputs["input_ids"][0]

        # score only the move tokens
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        move_start = len(prompt_ids)

        score = 0.0
        for i in range(move_start, len(input_ids)):
            token_id = input_ids[i]
            score += log_probs[0, i - 1, token_id].item()

        return score

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        # Step 1: material filter
        current_material = self._material_score(board)

        move_scores = []
        for move in legal_moves:
            board.push(move)
            new_material = self._material_score(board)
            board.pop()

            gain = new_material - current_material
            move_scores.append((move, gain))

        best_gain = max(score for _, score in move_scores)

        candidate_moves = [
            move for move, score in move_scores if score == best_gain
        ]

        # If single best → play immediately
        if len(candidate_moves) == 1:
            return candidate_moves[0].uci()

        # Step 2: LM tie-break
        try:
            self._load_model()
        except Exception:
            return candidate_moves[0].uci()

        best_move = candidate_moves[0]
        best_score = -float("inf")

        for move in candidate_moves:
            lm_score = self._score_move_with_lm(fen, move.uci())

            if lm_score > best_score:
                best_score = lm_score
                best_move = move

        return best_move.uci()
