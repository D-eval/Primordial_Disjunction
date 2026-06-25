import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sympy as sp

from config import cfg
from template import forward_template, inverse_template


@dataclass
class ExpressionNode:
    op: str
    value: Optional[int] = None
    children: Optional[List["ExpressionNode"]] = None

    def evaluate(self) -> float:
        if self.op == "leaf":
            return float(self.value)

        child_values = [child.evaluate() for child in self.children]
        func = cfg.opsToken2ops[self.op]["func"]
        if len(child_values) == 1:
            result = func(child_values[0])
        else:
            result = func(*child_values)

        if not math.isfinite(result):
            raise ValueError("Expression evaluated to a non-finite value.")
        return float(result)

    def operator_depth(self) -> int:
        if self.op == "leaf":
            return 0
        return 1 + max(child.operator_depth() for child in self.children)

    def to_prefix_string(self) -> str:
        if self.op == "leaf":
            return str(self.value)

        args = ",".join(child.to_prefix_string() for child in self.children)
        return f"{self.op}({args})"


class ExpressionTokenizer:
    def __init__(self):
        self.operator_tokens = set(cfg.opsTokens) - {"leaf"}
        self.punctuation_tokens = {"(", ")", ",", ".", "-"}

    def tokenize_expression(self, expr: ExpressionNode) -> List[str]:
        return self.tokenize_expression_string(expr.to_prefix_string())

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

        return tokens

    def tokenize_value(self, value: float, precision: int) -> List[str]:
        rendered = self.render_value(value=value, precision=precision)
        return self.tokenize_value_text(rendered)

    def tokenize_value_text(self, rendered: str) -> List[str]:
        tokens: List[str] = []
        for ch in rendered:
            if ch.isdigit() or ch in {".", "-"}:
                tokens.append(ch)
            else:
                raise ValueError(f"Unexpected character while tokenizing value: {ch}")
        return tokens

    @staticmethod
    def render_value(value: float, precision: int) -> str:
        return f"{value:.{precision}f}"


class SymbolicDatasetGenerator:
    def __init__(self, seed: Optional[int] = None, value_precision: int = 5):
        self.max_constant = cfg.dataset.max_constant
        self.max_depth = cfg.dataset.max_depth
        self.value_precision = value_precision
        self.tokenizer = ExpressionTokenizer()
        self.task_tokens = set(cfg.task_describe_token)
        self.forward_templates = list(forward_template)
        self.inverse_templates = list(inverse_template)

        self.unary_ops = [
            op_name
            for op_name, meta in cfg.opsToken2ops.items()
            if meta["arity"] == 1
        ]
        self.binary_ops = [
            op_name
            for op_name, meta in cfg.opsToken2ops.items()
            if meta["arity"] == 2
        ]

        if seed is not None:
            random.seed(seed)

    def estimate_sample_space(self) -> Dict[str, Dict[str, int]]:
        return {
            "forward": self.estimate_forward_space(),
            "inverse": self.estimate_inverse_space(),
            "simplify": self.estimate_simplify_space(),
        }

    def estimate_forward_space(self) -> Dict[str, int]:
        template_count = len(self.forward_templates)
        total_leaf_assignments = 0
        for template_text in self.forward_templates:
            leaf_count = self._count_template_leaf_num(template_text)
            total_leaf_assignments += self.max_constant ** leaf_count
        return {
            "template_count": template_count,
            "raw_expression_count": total_leaf_assignments,
        }

    def estimate_inverse_space(self) -> Dict[str, int]:
        template_count = len(self.inverse_templates)
        total_leaf_assignments = 0
        for template_text in self.inverse_templates:
            leaf_count = self._count_template_leaf_num(template_text)
            total_leaf_assignments += self.max_constant ** leaf_count
        return {
            "template_count": template_count,
            "raw_expression_count_before_simplify": total_leaf_assignments,
        }

    def estimate_simplify_space(self) -> Dict[str, int]:
        total_expression_count = self._count_expression_space(depth=self.max_depth, require_special=False)
        special_expression_count = self._count_expression_space(depth=self.max_depth, require_special=True)
        return {
            "raw_expression_count_upper_bound": total_expression_count,
            "special_expression_count_upper_bound": special_expression_count,
        }

    def sample_expression(self, max_depth: Optional[int] = None) -> ExpressionNode:
        depth_limit = self.max_depth if max_depth is None else max_depth

        while True:
            expr = self._sample_expression(depth_limit)
            try:
                expr.evaluate()
                return expr
            except (OverflowError, ValueError, ZeroDivisionError):
                continue

    def sample_inverse_expression(self) -> ExpressionNode:
        while True:
            raw_expr = self._sample_expression_from_template(random.choice(self.inverse_templates))
            try:
                simplified_expr = self._simplify_expression(raw_expr)
                simplified_expr.evaluate()
                return simplified_expr
            except (OverflowError, ValueError, ZeroDivisionError, NotImplementedError, ValueError):
                continue

    def sample_forward_expression(self) -> ExpressionNode:
        while True:
            raw_expr = self._sample_expression_from_template(random.choice(self.forward_templates))
            try:
                raw_expr.evaluate()
                return raw_expr
            except (OverflowError, ValueError, ZeroDivisionError):
                continue

    def sample_forward_example(self, max_depth: Optional[int] = None) -> dict:
        del max_depth
        expr = self.sample_forward_expression()
        value = expr.evaluate()

        input_tokens = ["|forward|"] + self.tokenizer.tokenize_expression(expr)
        output_tokens = ["|bos|"] + self.tokenizer.tokenize_value(
            value=value,
            precision=self.value_precision,
        ) + ["|eos|"]

        return {
            "task": "forward",
            "value": value,
            "value_text": self.tokenizer.render_value(value, self.value_precision),
            "expression": expr.to_prefix_string(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "target_tokens": output_tokens,
            "full_tokens": input_tokens + output_tokens,
        }

    def sample_inverse_example(self, max_depth: Optional[int] = None) -> dict:
        del max_depth
        expr = self.sample_inverse_expression()
        value = expr.evaluate()

        input_tokens = ["|inverse|"] + self.tokenizer.tokenize_value(
            value=value,
            precision=self.value_precision,
        )
        output_tokens = ["|bos|"] + self.tokenizer.tokenize_expression(expr) + ["|eos|"]

        return {
            "task": "inverse",
            "value": value,
            "value_text": self.tokenizer.render_value(value, self.value_precision),
            "expression": expr.to_prefix_string(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "target_tokens": output_tokens,
            "full_tokens": input_tokens + output_tokens,
        }

    def sample_simplify_example(self, max_depth: Optional[int] = None) -> dict:
        while True:
            expr = self.sample_expression(max_depth=max_depth)
            if not self._contains_any_op(expr, {"inv", "sqrt"}):
                continue
            try:
                simplified_expr = self._simplify_expression(expr)
                value = expr.evaluate()
                break
            except (OverflowError, ValueError, ZeroDivisionError, NotImplementedError):
                continue

        input_tokens = ["|simplify|"] + self.tokenizer.tokenize_expression(expr)
        output_tokens = ["|bos|"] + self.tokenizer.tokenize_expression(simplified_expr) + ["|eos|"]

        return {
            "task": "simplify",
            "value": value,
            "value_text": self.tokenizer.render_value(value, self.value_precision),
            "expression": expr.to_prefix_string(),
            "simplified_expression": simplified_expr.to_prefix_string(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "target_tokens": output_tokens,
            "full_tokens": input_tokens + output_tokens,
        }

    def sample_example(self, task: str, max_depth: Optional[int] = None) -> dict:
        if task == "|forward|" or task == "forward":
            return self.sample_forward_example(max_depth=max_depth)
        if task == "|inverse|" or task == "inverse":
            return self.sample_inverse_example(max_depth=max_depth)
        if task == "|simplify|" or task == "simplify":
            return self.sample_simplify_example(max_depth=max_depth)
        raise ValueError(f"Unknown task mode: {task}")

    def theoretical_max_expression_tokens(self, depth: Optional[int] = None) -> Tuple[int, str]:
        depth_limit = self.max_depth if depth is None else depth
        digit_len = len(str(self.max_constant))

        def solve(d: int) -> Tuple[int, str]:
            if d == 0:
                return digit_len, str(self.max_constant)

            child_len, child_expr = solve(d - 1)
            unary_len = 1 + 1 + child_len + 1
            unary_expr = f"sqrt({child_expr})"

            binary_len = 1 + 1 + child_len + 1 + child_len + 1
            binary_expr = f"mul({child_expr},{child_expr})"

            if binary_len >= unary_len:
                return binary_len, binary_expr
            return unary_len, unary_expr

        return solve(depth_limit)

    def theoretical_max_value_string(self, depth: Optional[int] = None) -> str:
        depth_limit = self.max_depth if depth is None else depth
        max_abs_value = self.max_constant ** (2 ** depth_limit)
        return f"{max_abs_value}.{('0' * self.value_precision)}"

    def theoretical_max_inverse_sequence_length(self, depth: Optional[int] = None) -> dict:
        expr_len, expr_text = self.theoretical_max_inverse_template_tokens()
        value_text = self.theoretical_max_value_string(depth=depth)
        value_len = len(self.tokenizer.tokenize_value_text(value_text))

        input_len = 1 + value_len
        target_len = 1 + expr_len + 1
        total_len = input_len + target_len

        return {
            "depth": self.max_depth if depth is None else depth,
            "expression_text": expr_text,
            "expression_token_length": expr_len,
            "value_text": value_text,
            "value_token_length": value_len,
            "input_token_length": input_len,
            "output_token_length": target_len,
            "total_token_length": total_len,
        }

    def theoretical_max_forward_sequence_length(self, depth: Optional[int] = None) -> dict:
        expr_len, expr_text = self.theoretical_max_forward_template_tokens()
        value_text = self.theoretical_max_value_string(depth=depth)
        value_len = len(self.tokenizer.tokenize_value_text(value_text))

        input_len = 1 + expr_len
        output_len = 1 + value_len + 1
        total_len = input_len + output_len

        return {
            "depth": self.max_depth if depth is None else depth,
            "expression_text": expr_text,
            "expression_token_length": expr_len,
            "value_text": value_text,
            "value_token_length": value_len,
            "input_token_length": input_len,
            "output_token_length": output_len,
            "total_token_length": total_len,
        }

    def theoretical_max_forward_template_tokens(self) -> Tuple[int, str]:
        best_text = ""
        best_len = -1
        for template_text in self.forward_templates:
            expr = self._sample_expression_from_template(template_text, deterministic_leaf=True)
            expr_text = expr.to_prefix_string()
            expr_len = len(self.tokenizer.tokenize_expression(expr))
            if expr_len > best_len:
                best_len = expr_len
                best_text = expr_text
        return best_len, best_text

    def theoretical_max_simplify_sequence_length(self, depth: Optional[int] = None) -> dict:
        expr_len, expr_text = self.theoretical_max_special_expression_tokens(
            depth=depth,
            required_ops={"inv", "sqrt"},
        )

        input_len = 1 + expr_len
        output_len = 1 + expr_len + 1
        total_len = input_len + output_len

        return {
            "depth": self.max_depth if depth is None else depth,
            "expression_text": expr_text,
            "expression_token_length": expr_len,
            "simplified_expression_text": expr_text,
            "output_token_length": output_len,
            "input_token_length": input_len,
            "total_token_length": total_len,
        }

    def theoretical_max_inverse_template_tokens(self) -> Tuple[int, str]:
        best_text = ""
        best_len = -1
        for template_text in self.inverse_templates:
            expr = self._sample_expression_from_template(template_text, deterministic_leaf=True)
            expr_text = expr.to_prefix_string()
            expr_len = len(self.tokenizer.tokenize_expression(expr))
            if expr_len > best_len:
                best_len = expr_len
                best_text = expr_text
        return best_len, best_text

    def _count_template_leaf_num(self, template_text: str) -> int:
        return sum(1 for token in template_text.split() if token == "leaf")

    def _count_expression_space(self, depth: int, require_special: bool) -> int:
        memo: Dict[Tuple[int, bool], int] = {}

        def solve(current_depth: int, need_special: bool) -> int:
            key = (current_depth, need_special)
            if key in memo:
                return memo[key]

            if current_depth <= 0:
                memo[key] = 0 if need_special else self.max_constant
                return memo[key]

            leaf_count = 0 if need_special else self.max_constant

            sqrt_count = solve(current_depth - 1, False)
            inv_count = solve(current_depth - 1, False)

            add_count = (
                solve(current_depth - 1, need_special) * solve(current_depth - 1, False)
                + solve(current_depth - 1, False) * solve(current_depth - 1, need_special)
                - solve(current_depth - 1, need_special) * solve(current_depth - 1, need_special)
                if need_special
                else solve(current_depth - 1, False) ** 2
            )
            mul_count = (
                solve(current_depth - 1, need_special) * solve(current_depth - 1, False)
                + solve(current_depth - 1, False) * solve(current_depth - 1, need_special)
                - solve(current_depth - 1, need_special) * solve(current_depth - 1, need_special)
                if need_special
                else solve(current_depth - 1, False) ** 2
            )
            neg_count = solve(current_depth - 1, need_special)

            if need_special:
                total = leaf_count + sqrt_count + inv_count + add_count + mul_count + neg_count
            else:
                child_total = solve(current_depth - 1, False)
                total = leaf_count + child_total + child_total + child_total * child_total + child_total * child_total + child_total
            memo[key] = total
            return total

        return solve(depth, require_special)

    def theoretical_max_special_expression_tokens(
        self,
        depth: Optional[int],
        required_ops: set[str],
    ) -> Tuple[int, str]:
        depth_limit = self.max_depth if depth is None else depth
        digit_len = len(str(self.max_constant))

        def solve(d: int, has_required_op: bool) -> Tuple[int, str]:
            if d == 0:
                if has_required_op:
                    return digit_len, str(self.max_constant)
                return -1, ""

            candidates: List[Tuple[int, str]] = []

            leaf_candidate = (digit_len, str(self.max_constant)) if has_required_op else (-1, "")
            candidates.append(leaf_candidate)

            child_len, child_expr = solve(d - 1, True)
            if child_len >= 0:
                candidates.append((1 + 1 + child_len + 1, f"sqrt({child_expr})"))

            child_len, child_expr = solve(d - 1, "inv" in required_ops or has_required_op)
            if child_len >= 0:
                candidates.append((1 + 1 + child_len + 1, f"inv({child_expr})"))

            left_len, left_expr = solve(d - 1, has_required_op)
            right_len, right_expr = solve(d - 1, has_required_op)
            if left_len >= 0 and right_len >= 0:
                candidates.append((1 + 1 + left_len + 1 + right_len + 1, f"mul({left_expr},{right_expr})"))
                candidates.append((1 + 1 + left_len + 1 + right_len + 1, f"add({left_expr},{right_expr})"))

            return max(candidates, key=lambda item: item[0])

        return solve(depth_limit, False)

    def _sample_expression(self, depth: int) -> ExpressionNode:
        if depth <= 0:
            return ExpressionNode(
                op="leaf",
                value=random.randint(1, self.max_constant),
                children=[],
            )

        branch_type = random.choices(
            population=["leaf", "unary", "binary"],
            weights=[0.20, 0.30, 0.50],
            k=1,
        )[0]

        if branch_type == "leaf":
            return ExpressionNode(
                op="leaf",
                value=random.randint(1, self.max_constant),
                children=[],
            )

        if branch_type == "unary":
            op = random.choice(self.unary_ops)
            child = self._sample_expression(depth - 1)
            return ExpressionNode(op=op, children=[child])

        op = random.choice(self.binary_ops)
        left = self._sample_expression(depth - 1)
        right = self._sample_expression(depth - 1)
        return ExpressionNode(op=op, children=[left, right])

    def _sample_expression_from_template(
        self,
        template_text: str,
        deterministic_leaf: bool = False,
    ) -> ExpressionNode:
        tokens = template_text.split()
        node, next_index = self._parse_template_tokens(tokens, 0, deterministic_leaf=deterministic_leaf)
        if next_index != len(tokens):
            raise ValueError(f"Unused template tokens: {tokens[next_index:]}")
        return node

    def _parse_template_tokens(
        self,
        tokens: List[str],
        index: int,
        deterministic_leaf: bool = False,
    ) -> Tuple[ExpressionNode, int]:
        if index >= len(tokens):
            raise ValueError("Unexpected end of template tokens.")

        token = tokens[index]
        if token == "leaf":
            value = self.max_constant if deterministic_leaf else random.randint(1, self.max_constant)
            return ExpressionNode(op="leaf", value=value, children=[]), index + 1

        if token not in cfg.opsToken2ops or token == "leaf":
            raise ValueError(f"Unknown template token: {token}")

        arity = cfg.opsToken2ops[token]["arity"]
        children = []
        next_index = index + 1
        for _ in range(arity):
            child, next_index = self._parse_template_tokens(
                tokens,
                next_index,
                deterministic_leaf=deterministic_leaf,
            )
            children.append(child)
        return ExpressionNode(op=token, children=children), next_index

    def _simplify_expression(self, expr: ExpressionNode) -> ExpressionNode:
        sympy_expr = self._expression_to_sympy(expr)
        simplified = sp.radsimp(sp.simplify(sympy_expr))
        return self._sympy_to_expression(simplified)

    def _contains_any_op(self, expr: ExpressionNode, op_names: set[str]) -> bool:
        if expr.op in op_names:
            return True
        if not expr.children:
            return False
        return any(self._contains_any_op(child, op_names) for child in expr.children)

    def _expression_to_sympy(self, expr: ExpressionNode) -> sp.Expr:
        if expr.op == "leaf":
            return sp.Integer(expr.value)
        if expr.op == "add":
            return self._expression_to_sympy(expr.children[0]) + self._expression_to_sympy(expr.children[1])
        if expr.op == "neg":
            return -self._expression_to_sympy(expr.children[0])
        if expr.op == "mul":
            return self._expression_to_sympy(expr.children[0]) * self._expression_to_sympy(expr.children[1])
        if expr.op == "inv":
            return 1 / self._expression_to_sympy(expr.children[0])
        if expr.op == "sqrt":
            return sp.sqrt(self._expression_to_sympy(expr.children[0]))
        raise NotImplementedError(f"Unsupported op for sympy conversion: {expr.op}")

    def _sympy_to_expression(self, expr: sp.Expr) -> ExpressionNode:
        expr = sp.simplify(expr)

        if expr.is_Integer:
            return ExpressionNode(op="leaf", value=int(expr), children=[])

        if expr.is_Rational:
            numerator = int(expr.p)
            denominator = int(expr.q)
            if denominator == 1:
                return ExpressionNode(op="leaf", value=numerator, children=[])
            base = self._build_mul_chain(
                [
                    ExpressionNode(op="leaf", value=abs(numerator), children=[]),
                    ExpressionNode(
                        op="inv",
                        children=[ExpressionNode(op="leaf", value=denominator, children=[])],
                    ),
                ]
            )
            if abs(numerator) == 1:
                base = ExpressionNode(
                    op="inv",
                    children=[ExpressionNode(op="leaf", value=denominator, children=[])],
                )
            if numerator < 0:
                return ExpressionNode(op="neg", children=[base])
            return base

        if expr.func == sp.sqrt or (expr.is_Pow and expr.exp == sp.Rational(1, 2)):
            return ExpressionNode(op="sqrt", children=[self._sympy_to_expression(expr.base)])

        if expr.is_Add:
            terms = expr.as_ordered_terms()
            nodes = [self._sympy_to_expression(term) for term in terms]
            return self._build_add_chain(nodes)

        if expr.is_Mul:
            coeff, rest = expr.as_coeff_Mul()
            factors = []
            if coeff == -1:
                return ExpressionNode(op="neg", children=[self._sympy_to_expression(rest)])
            if coeff != 1:
                factors.append(self._sympy_to_expression(coeff))
            for factor in sp.Mul.make_args(rest):
                factors.append(self._sympy_to_expression(factor))
            if not factors:
                raise NotImplementedError("Empty multiplication after simplification.")
            return self._build_mul_chain(factors)

        if expr.is_Pow and expr.exp == -1:
            return ExpressionNode(op="inv", children=[self._sympy_to_expression(expr.base)])

        raise NotImplementedError(f"Unsupported simplified sympy expression: {expr}")

    @staticmethod
    def _build_add_chain(nodes: List[ExpressionNode]) -> ExpressionNode:
        if not nodes:
            raise ValueError("Cannot build add chain from empty node list.")
        current = nodes[0]
        for node in nodes[1:]:
            current = ExpressionNode(op="add", children=[current, node])
        return current

    @staticmethod
    def _build_mul_chain(nodes: List[ExpressionNode]) -> ExpressionNode:
        if not nodes:
            raise ValueError("Cannot build mul chain from empty node list.")
        current = nodes[0]
        for node in nodes[1:]:
            current = ExpressionNode(op="mul", children=[current, node])
        return current


if __name__ == "__main__":
    generator = SymbolicDatasetGenerator(seed=44, value_precision=5)

    print("=== sample space estimate ===")
    print(generator.estimate_sample_space())

    forward_example = generator.sample_forward_example()
    inverse_example = generator.sample_inverse_example()
    simplify_example = generator.sample_simplify_example()
    forward_theory = generator.theoretical_max_forward_sequence_length()
    inverse_theory = generator.theoretical_max_inverse_sequence_length()
    simplify_theory = generator.theoretical_max_simplify_sequence_length()

    print("=== forward example ===")
    print(f"expression : {forward_example['expression']}")
    print(f"value      : {forward_example['value_text']}")
    print(f"input      : {forward_example['input_tokens']}")
    print(f"output     : {forward_example['output_tokens']}")
    print(f"full len   : {len(forward_example['full_tokens'])}")

    print("\n=== inverse example ===")
    print(f"value      : {inverse_example['value_text']}")
    print(f"expression : {inverse_example['expression']}")
    print(f"input      : {inverse_example['input_tokens']}")
    print(f"output     : {inverse_example['output_tokens']}")
    print(f"full len   : {len(inverse_example['full_tokens'])}")

    print("\n=== simplify example ===")
    print(f"expression : {simplify_example['expression']}")
    print(f"simplified : {simplify_example['simplified_expression']}")
    print(f"input      : {simplify_example['input_tokens']}")
    print(f"output     : {simplify_example['output_tokens']}")
    print(f"full len   : {len(simplify_example['full_tokens'])}")

    print("\n=== theoretical max forward sequence ===")
    print(f"max depth                : {forward_theory['depth']}")
    print(f"witness expression       : {forward_theory['expression_text']}")
    print(f"expression token length  : {forward_theory['expression_token_length']}")
    print(f"witness value            : {forward_theory['value_text']}")
    print(f"value token length       : {forward_theory['value_token_length']}")
    print(f"input token length       : {forward_theory['input_token_length']}")
    print(f"output token length      : {forward_theory['output_token_length']}")
    print(f"total token length       : {forward_theory['total_token_length']}")

    print("\n=== theoretical max inverse sequence ===")
    print(f"max depth                : {inverse_theory['depth']}")
    print(f"witness expression       : {inverse_theory['expression_text']}")
    print(f"expression token length  : {inverse_theory['expression_token_length']}")
    print(f"witness value            : {inverse_theory['value_text']}")
    print(f"value token length       : {inverse_theory['value_token_length']}")
    print(f"input token length       : {inverse_theory['input_token_length']}")
    print(f"output token length      : {inverse_theory['output_token_length']}")
    print(f"total token length       : {inverse_theory['total_token_length']}")

    print("\n=== theoretical max simplify sequence ===")
    print(f"max depth                : {simplify_theory['depth']}")
    print(f"witness expression       : {simplify_theory['expression_text']}")
    print(f"expression token length  : {simplify_theory['expression_token_length']}")
    print(f"witness simplified expr  : {simplify_theory['simplified_expression_text']}")
    print(f"input token length       : {simplify_theory['input_token_length']}")
    print(f"output token length      : {simplify_theory['output_token_length']}")
    print(f"total token length       : {simplify_theory['total_token_length']}")
    print(f"model max seq len        : {cfg.model.max_seq_len}")
