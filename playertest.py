import chess
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
    """
    Material-aware transformer chess player.
    1) Filters moves using material evaluation.
    2) Uses LM as tie-break.
    """

    def __init__(
        self,
        name: str = "Student",
        model_id: str = "sshleifer/tiny-gpt2",
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
    # Material evaluation
    # -------------------------
    def _material_score(self, board: chess.Board) -> int:
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        return score

    # -------------------------
    # Prompt
    # -------------------------
    def _build_prompt(self, fen: str) -> str:
        return f"Position: {fen}\nBest move:"

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        current_material = self._material_score(board)

        # Step 1: evaluate moves by material
        move_scores = []

        for move in legal_moves:
            board.push(move)

            new_material = self._material_score(board)
            material_gain = new_material - current_material

            mobility_bonus = 0.05 * len(list(board.legal_moves))

            check_bonus = 0.7 if board.is_check() else 0

            board.pop()

            capture_bonus = 0.3 if board.is_capture(move) else 0

            score = material_gain + capture_bonus + check_bonus + mobility_bonus

            move_scores.append((move, score))

        # Step 2: select best material gain
        best_gain = max(score for _, score in move_scores)

        best_material_moves = [
            move for move, score in move_scores if score == best_gain
        ]

        # If only one clearly best material move → play it
        if len(best_material_moves) == 1:
            return best_material_moves[0].uci()

        # Step 3: tie-break using transformer
        try:
            self._load_model()
        except Exception:
            return best_material_moves[0].uci()

        prompt = self._build_prompt(fen)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[:, -1, :]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        best_score = -float("inf")
        best_move = best_material_moves[0]

        for move in best_material_moves:
            tokens = self.tokenizer(move.uci(), add_special_tokens=False)["input_ids"]

            if not tokens:
                continue

            first_token = tokens[0]
            score = log_probs[0, first_token].item()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move.uci()
