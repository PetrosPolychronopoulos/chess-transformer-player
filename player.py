import chess
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player


class TransformerPlayer(Player):

    def __init__(
        self,
        name: str = "Student",
        model_id: str = "2pp/chess-transformer",
    ):
        super().__init__(name)

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self.model_loaded = False

    # -------------------------
    # Safe lazy loading
    # -------------------------
    def _load_model(self):
        if self.model_loaded:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=True  # allow offline usage
            )
        except Exception:
            # fallback to online if not cached
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                local_files_only=True
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True

    # -------------------------
    # Main API
    # -------------------------
    def get_move(self, fen: str) -> Optional[str]:

        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        try:
            self._load_model()
        except Exception:
            # deterministic safe fallback
            return legal_moves[0].uci()

        prompt = f"FEN: {fen}\nMove:"
        base_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        best_move = None
        best_score = float("-inf")

        with torch.no_grad():

            # Single forward pass for prompt
            base_outputs = self.model(**base_inputs, use_cache=True)
            base_logits = base_outputs.logits[:, -1, :]
            base_past = base_outputs.past_key_values

            for move in legal_moves:
                uci = move.uci()

                # IMPORTANT: match training distribution (leading space)
                move_tokens = self.tokenizer(
                    " " + uci,
                    add_special_tokens=False
                )["input_ids"]

                total_logprob = 0.0
                past = base_past
                logits = base_logits

                for token in move_tokens:

                    log_probs = torch.log_softmax(logits, dim=-1)
                    total_logprob += log_probs[0, token].item()

                    token_tensor = torch.tensor([[token]], device=self.device)

                    outputs = self.model(
                        input_ids=token_tensor,
                        past_key_values=past,
                        use_cache=True
                    )

                    logits = outputs.logits[:, -1, :]
                    past = outputs.past_key_values

                if total_logprob > best_score:
                    best_score = total_logprob
                    best_move = uci

        if best_move is not None:
            return best_move

        # Absolute deterministic safety
        return legal_moves[0].uci()
