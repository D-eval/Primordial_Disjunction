from typing import Iterable, List, Sequence

from config import cfg


class SymbolicTokenizer:
    def __init__(self):
        self.tokens = list(cfg.all_tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.tokens)}

        self.operator_tokens = set(cfg.opsTokens) - {"leaf"}
        self.punctuation_tokens = set(cfg.punctuation_tokens)
        self.special_tokens = set(cfg.special_token)
        self.value_tokens = set(cfg.digital_token + [".", "-"])

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def bos_id(self) -> int:
        return self.token_to_id["|bos|"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["|eos|"]

    def encode(self, tokens: Sequence[str]) -> List[int]:
        self.validate_tokens(tokens)
        return [self.token_to_id[token] for token in tokens]

    def decode(self, token_ids: Sequence[int]) -> List[str]:
        decoded = []
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token id: {token_id}")
            decoded.append(self.id_to_token[token_id])
        return decoded

    def detokenize(self, tokens: Sequence[str]) -> str:
        self.validate_tokens(tokens)
        return "".join(tokens)

    def tokenize_expression(self, expr) -> List[str]:
        if hasattr(expr, "to_prefix_string"):
            expr = expr.to_prefix_string()
        if not isinstance(expr, str):
            raise TypeError("Expression must be a string or provide to_prefix_string().")
        return self.tokenize_expression_string(expr)

    def tokenize_expression_string(self, text: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        while i < len(text):
            ch = text[i]

            if ch.isdigit():
                tokens.append(ch)
                i += 1
                continue

            if ch in self.punctuation_tokens:
                tokens.append(ch)
                i += 1
                continue

            if ch.isalpha() or ch == "_":
                j = i
                while j < len(text) and (text[j].isalpha() or text[j] == "_"):
                    j += 1
                token = text[i:j]
                if token not in self.operator_tokens:
                    raise ValueError(f"Unknown operator token: {token}")
                tokens.append(token)
                i = j
                continue

            raise ValueError(f"Unexpected character while tokenizing expression: {ch}")

        self.validate_tokens(tokens)
        return tokens

    def tokenize_value(self, value: float, precision: int) -> List[str]:
        return self.tokenize_value_text(self.render_value(value=value, precision=precision))

    def tokenize_value_text(self, text: str) -> List[str]:
        tokens = []
        for ch in text:
            if ch not in self.value_tokens:
                raise ValueError(f"Unexpected character while tokenizing value: {ch}")
            tokens.append(ch)

        self.validate_tokens(tokens)
        return tokens

    def build_input(self, task_token: str, payload_tokens: Sequence[str]) -> List[str]:
        if task_token not in cfg.task_describe_token:
            raise ValueError(f"Unknown task token: {task_token}")
        return [task_token] + list(payload_tokens)

    def build_output(self, payload_tokens: Sequence[str]) -> List[str]:
        return ["|bos|"] + list(payload_tokens) + ["|eos|"]

    def validate_tokens(self, tokens: Iterable[str]) -> None:
        unknown_tokens = [token for token in tokens if token not in self.token_to_id]
        if unknown_tokens:
            raise ValueError(f"Unknown tokens: {unknown_tokens}")

    @staticmethod
    def render_value(value: float, precision: int) -> str:
        return f"{value:.{precision}f}"


ExpressionTokenizer = SymbolicTokenizer
SymbolicVocabulary = SymbolicTokenizer


if __name__ == "__main__":
    tokenizer = SymbolicTokenizer()

    expression_text = "add(sqrt(2),mul(3,4))"
    value_text = tokenizer.render_value(3.1462643699, precision=5)

    expression_tokens = tokenizer.tokenize_expression_string(expression_text)
    value_tokens = tokenizer.tokenize_value_text(value_text)
    inverse_input_tokens = tokenizer.build_input("|inverse|", value_tokens)
    inverse_output_tokens = tokenizer.build_output(expression_tokens)
    encoded = tokenizer.encode(inverse_input_tokens + inverse_output_tokens)
    decoded = tokenizer.decode(encoded)

    print("=== tokenizer summary ===")
    print(f"vocab size            : {tokenizer.vocab_size}")
    print(f"bos id                : {tokenizer.bos_id}")
    print(f"eos id                : {tokenizer.eos_id}")

    print("\n=== expression example ===")
    print(f"text                  : {expression_text}")
    print(f"tokens                : {expression_tokens}")

    print("\n=== value example ===")
    print(f"text                  : {value_text}")
    print(f"tokens                : {value_tokens}")

    print("\n=== inverse sequence ===")
    print(f"input tokens          : {inverse_input_tokens}")
    print(f"output tokens         : {inverse_output_tokens}")
    print(f"encoded ids           : {encoded}")
    print(f"decoded tokens        : {decoded}")
