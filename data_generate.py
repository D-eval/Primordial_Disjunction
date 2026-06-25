import argparse
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
        if self.op == "ladd":
            result = sum(child_values)
            if not math.isfinite(result):
                raise ValueError("Expression evaluated to a non-finite value.")
            return float(result)

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

    @staticmethod
    def render_trace_tokens(tokens: List[str]) -> str:
        lines: List[str] = []
        current: List[str] = []
        indent = ""
        skip_tokens = {"|bos|", "|eos|"}

        def flush() -> None:
            nonlocal current, indent
            if current:
                lines.append(indent + ExpressionTokenizer._format_trace_line(current))
            current = []
            indent = ""

        for token in tokens:
            if token in skip_tokens:
                continue
            if token == "|nl|":
                flush()
                continue
            if token == "|indent|":
                indent += "    "
                continue
            if token == "|sp|":
                current.append(token)
                continue
            current.append(token)
        flush()
        return "\n".join(lines)

    @staticmethod
    def _format_trace_line(tokens: List[str]) -> str:
        segments: List[List[str]] = [[]]
        for token in tokens:
            if token == "|sp|":
                if segments[-1]:
                    segments.append([])
                continue
            segments[-1].append(token)
        return " ".join(
            ExpressionTokenizer._format_trace_segment(segment)
            for segment in segments
            if segment
        )

    @staticmethod
    def _format_trace_segment(tokens: List[str]) -> str:
        chunks: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "r" and index + 1 < len(tokens) and tokens[index + 1].isdigit():
                j = index + 1
                digits = []
                while j < len(tokens) and tokens[j].isdigit():
                    digits.append(tokens[j])
                    j += 1
                chunks.append("r" + "".join(digits))
                index = j
                continue

            if token.isdigit() or (
                token == "-"
                and index + 1 < len(tokens)
                and tokens[index + 1].isdigit()
                and (index == 0 or tokens[index - 1] == "=")
            ):
                number_parts = [token]
                j = index + 1
                seen_dot = False
                while j < len(tokens):
                    if tokens[j].isdigit():
                        number_parts.append(tokens[j])
                        j += 1
                        continue
                    if tokens[j] == "." and not seen_dot and j + 1 < len(tokens) and tokens[j + 1].isdigit():
                        seen_dot = True
                        number_parts.append(tokens[j])
                        j += 1
                        continue
                    break
                chunks.append("".join(number_parts))
                index = j
                continue

            chunks.append(token)
            index += 1
        return "".join(chunks)


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
        think_tokens = self._build_forward_think_tokens(expr)
        final_value_tokens = self.tokenizer.tokenize_value(
            value=value,
            precision=self.value_precision,
        )
        output_tokens = (
            ["|bos|", "|beginOfThink|"]
            + think_tokens
            + ["|nl|", "|endOfThink|"]
            + final_value_tokens
            + ["|eos|"]
        )

        return {
            "task": "forward",
            "value": value,
            "value_text": self.tokenizer.render_value(value, self.value_precision),
            "expression": expr.to_prefix_string(),
            "think_tokens": think_tokens,
            "think_text": self.tokenizer.render_trace_tokens(["|beginOfThink|"] + think_tokens + ["|nl|", "|endOfThink|"]),
            "sft_text": self.tokenizer.render_trace_tokens(input_tokens + ["|nl|"] + output_tokens),
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
        output_len = cfg.model.max_seq_len - input_len
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

    def _build_forward_think_tokens(self, expr: ExpressionNode) -> List[str]:
        node_ids: Dict[int, int] = {}

        def assign_ids(node: ExpressionNode) -> None:
            node_ids[id(node)] = len(node_ids)
            for child in node.children or []:
                assign_ids(child)

        assign_ids(expr)
        lines: List[List[str]] = []

        def var_tokens(node: ExpressionNode) -> List[str]:
            return [f"r{node_ids[id(node)]}"]

        def emit(parts: List[str], indent: bool = False) -> None:
            line = ["|nl|"]
            if indent:
                line.append("|indent|")
            for part_index, part in enumerate(parts):
                if part_index > 0:
                    line.append("|sp|")
                line.extend(self._trace_part_tokens(part))
            lines.append(line)

        def emit_structure(node: ExpressionNode) -> None:
            if node.op == "leaf":
                emit(var_tokens(node) + ["=", self._render_trace_value(node.evaluate())])
                return
            rhs = [node.op]
            for child in node.children or []:
                rhs.extend(var_tokens(child))
            emit(var_tokens(node) + ["="] + rhs)
            for child in node.children or []:
                emit_structure(child)

        def emit_compute(node: ExpressionNode) -> float:
            if node.op == "leaf":
                return node.evaluate()

            child_values = [emit_compute(child) for child in node.children or []]
            result = node.evaluate()
            rendered_children = self._render_compute_args(node, child_values)
            emit(var_tokens(node) + ["=", node.op] + rendered_children)
            for detail_line, is_indented in self._operation_detail_lines(node, child_values, result):
                emit(detail_line, indent=is_indented)
            emit(var_tokens(node) + ["=", self._render_trace_value(result)])
            return result

        emit_structure(expr)
        emit_compute(expr)
        return [token for line in lines for token in line]

    def _render_compute_args(self, node: ExpressionNode, child_values: List[float]) -> List[str]:
        rendered: List[str] = []
        for child, value in zip(node.children or [], child_values):
            if node.op == "add" and child.op == "neg":
                rendered.extend(["neg", self._render_trace_value(abs(value))])
            else:
                rendered.append(self._render_trace_value(value))
        return rendered

    def _operation_detail_lines(
        self,
        node: ExpressionNode,
        child_values: List[float],
        result: float,
    ) -> List[Tuple[List[str], bool]]:
        op = node.op
        if op == "add":
            return self._add_detail_lines(
                child_values[0],
                child_values[1],
                left_node=node.children[0],
                right_node=node.children[1],
            )
        if op == "mul":
            return self._mul_detail_lines(child_values[0], child_values[1], result)
        if op == "ladd":
            if all(self._is_nonnegative_int(value) for value in child_values):
                values = [int(round(value)) for value in child_values]
                return [(["ladd", "("] + [str(value) for value in values] + [")"], True)] + self._ladd_detail_lines(values)
            rendered_values = [self._render_trace_value(value) for value in child_values]
            return [(["ladd", "("] + rendered_values + [")"], True), (["=", self._render_trace_value(result)], True)]
        if op == "neg":
            value = self._render_trace_value(child_values[0])
            rendered_result = self._render_trace_value(result)
            return [(["neg", value], True), (["=", rendered_result], True)]
        if op == "inv":
            value = self._render_trace_value(child_values[0])
            rendered_result = self._render_trace_value(result)
            return [(["inv", value], True), (["=", rendered_result], True)]
        if op == "sqrt":
            value = self._render_trace_value(child_values[0])
            rendered_result = self._render_trace_value(result)
            return [(["sqrt", value], True), (["=", rendered_result], True)]
        return [([op] + [self._render_trace_value(value) for value in child_values], True), (["=", self._render_trace_value(result)], True)]

    def _add_detail_lines(
        self,
        left: float,
        right: float,
        left_node: Optional[ExpressionNode] = None,
        right_node: Optional[ExpressionNode] = None,
    ) -> List[Tuple[List[str], bool]]:
        rendered_left = self._render_trace_value(left)
        rendered_right = self._render_trace_value(right)
        if (
            left_node is not None
            and right_node is not None
            and left_node.op == "neg"
            and self._is_nonnegative_int(abs(left))
            and self._is_nonnegative_int(right)
        ):
            magnitude = abs(int(round(left)))
            other = int(round(right))
            lines: List[Tuple[List[str], bool]] = [
                (["=", "add", str(other), "neg", str(magnitude)], True),
                (["=", "sub", str(other), str(magnitude)], True),
            ]
            if other >= magnitude:
                lines.extend(self._sub_detail_lines(other, magnitude, indent=True))
            else:
                lines.append((["=", "neg", "sub", str(magnitude), str(other)], True))
                lines.extend(self._sub_detail_lines(magnitude, other, indent=True))
                lines.append((["=", "neg", str(magnitude - other)], True))
            return lines

        if (
            left_node is not None
            and right_node is not None
            and right_node.op == "neg"
            and self._is_nonnegative_int(left)
            and self._is_nonnegative_int(abs(right))
        ):
            magnitude = abs(int(round(right)))
            other = int(round(left))
            lines = [
                (["=", "add", str(other), "neg", str(magnitude)], True),
                (["=", "sub", str(other), str(magnitude)], True),
            ]
            if other >= magnitude:
                lines.extend(self._sub_detail_lines(other, magnitude, indent=True))
            else:
                lines.append((["=", "neg", "sub", str(magnitude), str(other)], True))
                lines.extend(self._sub_detail_lines(magnitude, other, indent=True))
                lines.append((["=", "neg", str(magnitude - other)], True))
            return lines

        if not self._is_nonnegative_int(left) or not self._is_nonnegative_int(right):
            result = self._render_trace_value(left + right)
            return [(["add", rendered_left, rendered_right], True), (["=", result], True)]

        left_text = str(int(round(left)))
        right_text = str(int(round(right)))
        width = max(len(left_text), len(right_text))
        left_pad = left_text.zfill(width)
        right_pad = right_text.zfill(width)
        pairs = list(zip(reversed(left_pad), reversed(right_pad)))

        carry = 0
        pair_results: List[Tuple[str, str]] = []
        for left_digit, right_digit in pairs:
            column_sum = int(left_digit) + int(right_digit)
            pair_results.append((str(column_sum % 10), str(column_sum // 10)))

        result_text = str(int(left_text) + int(right_text))
        padded_result = result_text.zfill(width + 1)
        add_pos = ["=", "addPos"]
        add_pos_res = ["=", "addPosRes"]
        add_res_carry = ["=", "addRes"]
        add_res_digits = ["=", "addRes"]

        for left_digit, right_digit in pairs:
            add_pos.extend(["(", left_digit, right_digit, ")"])
        for digit, next_carry in pair_results:
            add_pos_res.extend(["(", digit, next_carry, ")"])

        carry_groups: List[List[str]] = [[pair_results[0][0]]]
        add_res_carry.append(pair_results[0][0])
        for previous_pair, current_pair in zip(pair_results, pair_results[1:]):
            group = [previous_pair[1], current_pair[0]]
            carry_groups.append(group)
            add_res_carry.extend(["(", *group, ")"])
        carry_groups.append([pair_results[-1][1]])
        add_res_carry.append(pair_results[-1][1])

        carry = 0
        for group in carry_groups:
            group_sum = sum(int(value) for value in group) + carry
            add_res_digits.append(str(group_sum % 10))
            carry = group_sum // 10
        if carry:
            add_res_digits.append(str(carry))

        return [
            (["add", left_text, right_text], True),
            (["=", "addPad", left_pad, right_pad], True),
            (add_pos, True),
            (add_pos_res, True),
            (add_res_carry, True),
            (add_res_digits, True),
            (["=", padded_result], True),
            (["=", result_text], True),
        ]

    def _sub_detail_lines(
        self,
        left: int,
        right: int,
        indent: bool = True,
    ) -> List[Tuple[List[str], bool]]:
        if left < right:
            raise ValueError("sub detail requires left >= right.")

        left_text = str(left)
        right_text = str(right)
        width = max(len(left_text), len(right_text))
        left_pad = left_text.zfill(width)
        right_pad = right_text.zfill(width)
        current_digits = [int(ch) for ch in reversed(left_pad)]
        right_digits = [int(ch) for ch in reversed(right_pad)]

        sub_pos_lines: List[List[str]] = []
        initial_sub_pos = ["=", "subPos"]
        for left_digit, right_digit in zip(current_digits, right_digits):
            initial_sub_pos.extend(["(", str(left_digit), str(right_digit), ")"])
        sub_pos_lines.append(initial_sub_pos)

        for index in range(width):
            if current_digits[index] < right_digits[index]:
                borrow_index = index + 1
                while borrow_index < width and current_digits[borrow_index] == 0:
                    borrow_index += 1
                if borrow_index >= width:
                    raise ValueError("No borrow source found in subtraction.")
                current_digits[borrow_index] -= 1
                for zero_index in range(borrow_index - 1, index, -1):
                    current_digits[zero_index] += 9
                current_digits[index] += 10

                sub_pos = ["=", "subPos"]
                for left_digit, right_digit in zip(current_digits, right_digits):
                    sub_pos.extend(["(", str(left_digit), str(right_digit), ")"])
                sub_pos_lines.append(sub_pos)

        result_digits = [str(current_digits[index] - right_digits[index]) for index in range(width)]
        result_text = str(left - right)
        padded_result = result_text.zfill(width)

        return [
            (["sub", left_text, right_text], indent),
            (["=", "subPad", left_pad, right_pad], indent),
            *[(line, indent) for line in sub_pos_lines],
            (["=", "subRes"] + result_digits, indent),
            (["=", padded_result], indent),
        ]

    def _mul_detail_lines(self, left: float, right: float, result: float) -> List[Tuple[List[str], bool]]:
        rendered_left = self._render_trace_value(left)
        rendered_right = self._render_trace_value(right)
        rendered_result = self._render_trace_value(result)
        if not self._is_nonnegative_int(left) or not self._is_nonnegative_int(right):
            return [(["mul", rendered_left, rendered_right], True), (["=", rendered_result], True)]

        left_parts = self._place_parts(int(round(left)))
        right_parts = self._place_parts(int(round(right)))
        partials = [left_part * right_part for left_part in left_parts for right_part in right_parts]

        lines: List[Tuple[List[str], bool]] = [
            (["mul"] + self._nested_add_tokens(left_parts) + self._nested_add_tokens(right_parts), True)
        ]
        mul_terms: List[str] = []
        for left_part in left_parts:
            for right_part in right_parts:
                mul_terms.extend(["mul", str(left_part), str(right_part)])
        lines.append((["=", "ladd", "("] + mul_terms + [")"], False))
        lines.append((["=", "ladd", "("] + [str(partial) for partial in partials] + [")"], False))
        lines.extend(self._ladd_detail_lines(partials))
        lines.append((["=", rendered_result], False))
        return lines

    def _ladd_detail_lines(self, values: List[int]) -> List[Tuple[List[str], bool]]:
        if not values:
            return []
        if len(values) == 1:
            return [(["=", str(values[0])], False)]

        lines: List[Tuple[List[str], bool]] = []
        current_tokens = self._nested_add_tokens(values)
        lines.append((["="] + current_tokens, False))

        pending = list(values)
        while len(pending) > 1:
            right = pending[-1]
            left = pending[-2]
            add_result = left + right
            lines.extend(self._add_detail_lines(left, right))
            pending = pending[:-2] + [add_result]
            lines.append((["="] + self._nested_add_tokens(pending), False))
        return lines

    def _nested_add_tokens(self, values: List[int]) -> List[str]:
        if not values:
            return ["0"]
        if len(values) == 1:
            return [str(values[0])]
        return ["add", str(values[0])] + self._nested_add_tokens(values[1:])

    @staticmethod
    def _place_parts(value: int) -> List[int]:
        if value == 0:
            return [0]
        digits = list(str(value))
        width = len(digits)
        parts = []
        for index, digit in enumerate(digits):
            part = int(digit) * (10 ** (width - index - 1))
            if part != 0:
                parts.append(part)
        return parts or [0]

    @staticmethod
    def _is_nonnegative_int(value: float) -> bool:
        return value >= 0 and abs(value - round(value)) < 1e-9

    def _render_trace_value(self, value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return self.tokenizer.render_value(value, self.value_precision)

    @staticmethod
    def _trace_part_tokens(part: str) -> List[str]:
        if part.startswith("r") and len(part) > 1 and part[1:].isdigit():
            return ["r"] + list(part[1:])
        if part in cfg.all_tokens:
            return [part]
        return list(part)

    def parse_manual_expression(self, text: str) -> ExpressionNode:
        parser = _ManualExpressionParser(text)
        return parser.parse()

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


class _ManualExpressionParser:
    def __init__(self, text: str):
        self.text = text
        self.index = 0
        self.manual_ops = {"add", "mul", "neg", "inv", "sqrt", "ladd"}

    def parse(self) -> ExpressionNode:
        node = self._parse_expr()
        self._skip_ws()
        if self.index != len(self.text):
            raise ValueError(f"Unexpected trailing input near: {self.text[self.index:]}")
        return node

    def _parse_expr(self) -> ExpressionNode:
        self._skip_ws()
        if self.index >= len(self.text):
            raise ValueError("Unexpected end of input.")

        if self.text[self.index].isdigit() or self.text[self.index] == "-":
            return self._parse_number()

        name = self._parse_name()
        if name not in self.manual_ops:
            raise ValueError(f"Unsupported manual expression op: {name}")

        self._skip_ws()
        self._expect("(")
        args: List[ExpressionNode] = []
        self._skip_ws()
        if self._peek() != ")":
            while True:
                args.append(self._parse_expr())
                self._skip_ws()
                if self._peek() != ",":
                    break
                self.index += 1
        self._expect(")")

        if name in {"add", "mul"} and len(args) != 2:
            raise ValueError(f"{name} expects exactly 2 args.")
        if name in {"neg", "inv", "sqrt"} and len(args) != 1:
            raise ValueError(f"{name} expects exactly 1 arg.")
        if name == "ladd" and not args:
            raise ValueError("ladd expects at least 1 arg.")
        return ExpressionNode(op=name, children=args)

    def _parse_number(self) -> ExpressionNode:
        sign = 1
        if self.text[self.index] == "-":
            sign = -1
            self.index += 1
        start = self.index
        while self.index < len(self.text) and self.text[self.index].isdigit():
            self.index += 1
        if start == self.index:
            raise ValueError("Expected digits after '-'.")
        value = int(self.text[start:self.index]) * sign
        if value < 0:
            return ExpressionNode(
                op="neg",
                children=[ExpressionNode(op="leaf", value=abs(value), children=[])],
            )
        return ExpressionNode(op="leaf", value=value, children=[])

    def _parse_name(self) -> str:
        start = self.index
        while self.index < len(self.text) and (self.text[self.index].isalpha() or self.text[self.index] == "_"):
            self.index += 1
        if start == self.index:
            raise ValueError(f"Expected operator near: {self.text[self.index:]}")
        return self.text[start:self.index]

    def _skip_ws(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _peek(self) -> str:
        self._skip_ws()
        if self.index >= len(self.text):
            return ""
        return self.text[self.index]

    def _expect(self, char: str) -> None:
        self._skip_ws()
        if self.index >= len(self.text) or self.text[self.index] != char:
            found = self.text[self.index : self.index + 1]
            raise ValueError(f"Expected '{char}', found '{found}'.")
        self.index += 1


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--trace", "-t", help="Manual expression, e.g. 'ladd(34,45,mul(4,5))'.")
    args = cli_parser.parse_args()

    generator = SymbolicDatasetGenerator(seed=44, value_precision=5)
    if args.trace:
        expr = generator.parse_manual_expression(args.trace)
        value = expr.evaluate()
        think_tokens = generator._build_forward_think_tokens(expr)
        think_text = generator.tokenizer.render_trace_tokens(
            ["|beginOfThink|"] + think_tokens + ["|nl|", "|endOfThink|"]
        )
        print(f"expression : {expr.to_prefix_string()}")
        print(f"value      : {generator._render_trace_value(value)}")
        print("think text :")
        print(think_text)
        raise SystemExit(0)

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
    print("think text :")
    print(forward_example["think_text"])
    print("sft text   :")
    print(forward_example["sft_text"])

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
