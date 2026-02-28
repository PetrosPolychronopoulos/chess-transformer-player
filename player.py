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
        top_k: int = 3,  # refinement depth
    ):
        super().__init__(name)

        self.model_id = model_id
        self.top_k = top_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self.loaded = False

    # -------------------------
    # Safe lazy loading
    # -------------------------
    def _load_model(self):
        if self.loaded:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                local_files_only=True
            )
        except Exception:
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
        self.loaded = True

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
            return legal_moves[0].uci()

        prompt = f"FEN: {fen}\nMove:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():

            base_outputs = self.model(**inputs, use_cache=True)
            base_logits = base_outputs.logits[:, -1, :]
            base_past = base_outputs.past_key_values

            log_probs = torch.log_softmax(base_logits, dim=-1)

            # -------------------------
            # Stage 1: Fast ranking
            # -------------------------
            scored_moves = []

            for move in legal_moves:
                uci = move.uci()

                tokens = self.tokenizer(
                    " " + uci,
                    add_special_tokens=False
                )["input_ids"]

                if not tokens:
                    continue

                first_token = tokens[0]
                score = log_probs[0, first_token].item()
                scored_moves.append((uci, score))

            if not scored_moves:
                return legal_moves[0].uci()

            scored_moves.sort(key=lambda x: x[1], reverse=True)

            candidates = scored_moves[:min(self.top_k, len(scored_moves))]

            # -------------------------
            # Stage 2: Full scoring
            # -------------------------
            best_move = None
            best_score = float("-inf")

            for uci, _ in candidates:

                tokens = self.tokenizer(
                    " " + uci,
                    add_special_tokens=False
                )["input_ids"]

                total_logprob = 0.0
                past = base_past
                logits = base_logits

                for token in tokens:
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

        return best_move if best_move else legal_moves[0].uci()
