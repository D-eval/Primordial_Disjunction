from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sympy as sp
import torch

from config import cfg
from data_generate import ExpressionNode, ExpressionTokenizer, SymbolicDatasetGenerator
from model import SymbolicVocabulary, build_device, build_model


def tokens_to_text(tokens: List[str]) -> str:
    return "".join(token for token in tokens if token not in {"|bos|", "|eos|"})


def extract_final_answer_tokens(tokens: List[str]) -> List[str]:
    if "|endOfThink|" not in tokens:
        return tokens
    start = tokens.index("|endOfThink|") + 1
    return [token for token in tokens[start:] if token not in {"|bos|", "|eos|"}]


class ExpressionParser:
    def __init__(self):
        self.unary_ops = {
            op_name
            for op_name, meta in cfg.opsToken2ops.items()
            if meta["arity"] == 1
        }
        self.binary_ops = {
            op_name
            for op_name, meta in cfg.opsToken2ops.items()
            if meta["arity"] == 2
        }

    def parse(self, tokens: List[str]) -> ExpressionNode:
        node, next_index = self._parse_at(tokens, 0)
        if next_index != len(tokens):
            raise ValueError("Unexpected trailing tokens.")
        return node

    def _parse_at(self, tokens: List[str], start: int) -> Tuple[ExpressionNode, int]:
        if start >= len(tokens):
            raise ValueError("Unexpected end of tokens.")

        token = tokens[start]
        if token in self.binary_ops:
            if start + 1 >= len(tokens) or tokens[start + 1] != "(":
                raise ValueError("Missing opening parenthesis for binary op.")
            left, next_index = self._parse_at(tokens, start + 2)
            if next_index >= len(tokens) or tokens[next_index] != ",":
                raise ValueError("Missing comma in binary op.")
            right, next_index = self._parse_at(tokens, next_index + 1)
            if next_index >= len(tokens) or tokens[next_index] != ")":
                raise ValueError("Missing closing parenthesis for binary op.")
            return ExpressionNode(op=token, children=[left, right]), next_index + 1

        if token in self.unary_ops:
            if start + 1 >= len(tokens) or tokens[start + 1] != "(":
                raise ValueError("Missing opening parenthesis for unary op.")
            child, next_index = self._parse_at(tokens, start + 2)
            if next_index >= len(tokens) or tokens[next_index] != ")":
                raise ValueError("Missing closing parenthesis for unary op.")
            return ExpressionNode(op=token, children=[child]), next_index + 1

        if token == "-" or token.isdigit():
            number_tokens = [token]
            index = start + 1
            while index < len(tokens) and tokens[index].isdigit():
                number_tokens.append(tokens[index])
                index += 1
            number_value = int("".join(number_tokens))
            return ExpressionNode(op="leaf", value=number_value, children=[]), index

        raise ValueError(f"Unexpected token while parsing: {token}")


@dataclass
class PredictionResult:
    output_tokens: List[str]
    exact_match: bool
    numeric_success: bool
    invalid: bool
    gap_text: str


def extract_generated_output(
    vocab: SymbolicVocabulary,
    generated_ids: List[int],
    prompt_len: int,
) -> List[str]:
    tokens = vocab.decode(generated_ids[prompt_len:])
    output_tokens: List[str] = []
    for token in tokens:
        output_tokens.append(token)
        if token == "|eos|":
            break
    return output_tokens


def evaluate_forward_like_prediction(
    output_tokens: List[str],
    example: Dict,
) -> PredictionResult:
    predicted_text = tokens_to_text(extract_final_answer_tokens(output_tokens))
    exact_match = output_tokens == example["output_tokens"]
    try:
        predicted_value = float(predicted_text)
    except ValueError:
        return PredictionResult(output_tokens, exact_match, False, True, "invalid float output")

    target_value = example["value"]
    numeric_gap = abs(predicted_value - target_value)
    numeric_success = numeric_gap < 10 ** (-cfg.dataset.value_precision)
    return PredictionResult(
        output_tokens,
        exact_match,
        numeric_success,
        False,
        f"pred={predicted_value:.10f}, target={target_value:.10f}, abs_diff={numeric_gap:.10f}",
    )


def evaluate_inverse_prediction(
    output_tokens: List[str],
    example: Dict,
    parser: ExpressionParser,
) -> PredictionResult:
    exact_match = output_tokens == example["output_tokens"]
    clean_tokens = [token for token in output_tokens if token not in {"|bos|", "|eos|"}]

    try:
        node = parser.parse(clean_tokens)
        predicted_value = node.evaluate()
    except Exception:
        return PredictionResult(output_tokens, exact_match, False, True, "invalid expression output")

    numeric_gap = abs(predicted_value - example["value"])
    numeric_success = numeric_gap < 10 ** (-cfg.dataset.value_precision)
    return PredictionResult(
        output_tokens,
        exact_match,
        numeric_success,
        False,
        f"pred_value={predicted_value:.10f}, target_value={example['value']:.10f}, abs_diff={numeric_gap:.10f}",
    )


def expression_to_sympy(expr: ExpressionNode) -> sp.Expr:
    if expr.op == "leaf":
        return sp.Integer(expr.value)
    if expr.op == "add":
        return expression_to_sympy(expr.children[0]) + expression_to_sympy(expr.children[1])
    if expr.op == "neg":
        return -expression_to_sympy(expr.children[0])
    if expr.op == "mul":
        return expression_to_sympy(expr.children[0]) * expression_to_sympy(expr.children[1])
    if expr.op == "inv":
        return 1 / expression_to_sympy(expr.children[0])
    if expr.op == "sqrt":
        return sp.sqrt(expression_to_sympy(expr.children[0]))
    raise ValueError(f"Unsupported op: {expr.op}")


def evaluate_simplify_prediction(
    output_tokens: List[str],
    example: Dict,
    parser: ExpressionParser,
) -> PredictionResult:
    exact_match = output_tokens == example["output_tokens"]
    clean_tokens = [token for token in output_tokens if token not in {"|bos|", "|eos|"}]

    try:
        predicted_expr = parser.parse(clean_tokens)
        source_expr = parser.parse(example["input_tokens"][1:])
        predicted_sympy = sp.simplify(expression_to_sympy(predicted_expr))
        source_sympy = sp.simplify(expression_to_sympy(source_expr))
    except Exception:
        return PredictionResult(output_tokens, exact_match, False, True, "invalid simplify expression output")

    numeric_success = sp.simplify(predicted_sympy - source_sympy) == 0
    return PredictionResult(
        output_tokens,
        exact_match,
        bool(numeric_success),
        False,
        f"pred_sympy={predicted_sympy}, target_sympy={source_sympy}",
    )


def generate_prediction(
    model: torch.nn.Module,
    vocab: SymbolicVocabulary,
    example: Dict,
    device: torch.device,
) -> List[str]:
    prompt_ids = vocab.encode(example["input_tokens"])
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=cfg.eval.max_new_tokens,
        eos_token_id=vocab.eos_id,
    )
    return extract_generated_output(vocab, generated[0].tolist(), prompt_len=len(prompt_ids))


def evaluate_model(
    model: torch.nn.Module,
    vocab: SymbolicVocabulary,
    sample_num: Optional[int] = None,
    device: Optional[torch.device] = None,
    task_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
    if device is None:
        device = build_device(cfg.eval.device)
    if sample_num is None:
        sample_num = cfg.eval.sample_num
    if task_names is None:
        task_names = list(cfg.eval.task_names)

    parser = ExpressionParser()
    generator = SymbolicDatasetGenerator(value_precision=cfg.dataset.value_precision)
    model.eval()

    metrics: Dict[str, Dict[str, float]] = {}
    visual_records: List[Dict[str, str]] = []
    with torch.no_grad():
        for task_name in task_names:
            exact_match_count = 0
            numeric_success_count = 0
            invalid_count = 0

            for _ in range(sample_num):
                example = generator.sample_example(task_name)
                output_tokens = generate_prediction(model, vocab, example, device)
                if task_name == "inverse":
                    result = evaluate_inverse_prediction(output_tokens, example, parser)
                elif task_name == "simplify":
                    result = evaluate_simplify_prediction(output_tokens, example, parser)
                else:
                    result = evaluate_forward_like_prediction(output_tokens, example)

                exact_match_count += int(result.exact_match)
                numeric_success_count += int(result.numeric_success)
                invalid_count += int(result.invalid)
                visual_records.append(
                    {
                        "task": task_name,
                        "input": "".join(example["input_tokens"]),
                        "target": "".join(example["output_tokens"]),
                        "prediction": "".join(result.output_tokens),
                        "exact_match": str(result.exact_match),
                        "success": str(result.numeric_success),
                        "invalid": str(result.invalid),
                        "gap": result.gap_text,
                    }
                )

            metrics[task_name] = {
                "exact_match_rate": exact_match_count / sample_num,
                "numeric_success_rate": numeric_success_count / sample_num,
                "invalid_rate": invalid_count / sample_num,
            }

    return metrics, visual_records


def maybe_load_checkpoint(model: torch.nn.Module, ckpt_path: Optional[str], device: torch.device) -> None:
    if ckpt_path is None:
        return
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def main() -> None:
    device = build_device(cfg.eval.device)
    vocab = SymbolicVocabulary()
    model = build_model().to(device)
    maybe_load_checkpoint(model, cfg.eval.ckpt_path, device)
    metrics, _ = evaluate_model(
        model=model,
        vocab=vocab,
        sample_num=cfg.eval.sample_num,
        device=device,
        task_names=cfg.eval.task_names,
    )
    print(metrics)


if __name__ == "__main__":
    main()
