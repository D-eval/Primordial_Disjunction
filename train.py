import json
import os
import random
import struct
import zlib
from typing import Dict, List

import torch

from config import cfg
from data_generate import SymbolicDatasetGenerator
from eval import evaluate_model
from model import SymbolicVocabulary, build_device, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_task_weights() -> List[float]:
    task_names = list(cfg.train.task_names)
    task_probs = list(cfg.train.task_probs)
    if len(task_names) == len(task_probs):
        return task_probs
    return [1.0] * len(task_names)


def sample_training_batch(
    generator: SymbolicDatasetGenerator,
    vocab: SymbolicVocabulary,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    task_weights = resolve_task_weights()
    examples = [
        generator.sample_example(
            task=random.choices(cfg.train.task_names, weights=task_weights, k=1)[0]
        )
        for _ in range(batch_size)
    ]

    max_len = max(len(example["full_tokens"]) for example in examples)
    pad_id = vocab.eos_id

    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    labels: List[List[int]] = []

    for example in examples:
        full_ids = vocab.encode(example["full_tokens"])
        prompt_len = len(example["input_tokens"])
        pad_len = max_len - len(full_ids)

        input_ids.append(full_ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(full_ids) + [0] * pad_len)

        label = full_ids[:]
        label[:prompt_len] = [-100] * prompt_len
        labels.append(label + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_score: float,
    loss_history: List[Dict[str, float]],
    path: str,
) -> None:
    torch.save(
        {
            "step": step,
            "best_score": best_score,
            "loss_history": loss_history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config_tokens": list(cfg.all_tokens),
        },
        path,
    )


def append_jsonl(path: str, payload: Dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_visual_report(path: str, step: int, visual_records: List[Dict[str, str]]) -> None:
    lines = [f"step={step}", ""]
    for idx, record in enumerate(visual_records, start=1):
        lines.append(f"[{idx}] task={record['task']}")
        lines.append(f"input      : {record['input']}")
        lines.append(f"target     : {record['target']}")
        lines.append(f"prediction : {record['prediction']}")
        lines.append(f"exact_match: {record['exact_match']}")
        lines.append(f"success    : {record['success']}")
        lines.append(f"invalid    : {record['invalid']}")
        lines.append(f"gap        : {record['gap']}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _write_simple_png(rgb_rows: List[List[tuple[int, int, int]]], path: str) -> None:
    height = len(rgb_rows)
    width = len(rgb_rows[0]) if height > 0 else 0

    raw = bytearray()
    for row in rgb_rows:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_bytes = b"\x89PNG\r\n\x1a\n"
    png_bytes += _png_chunk(b"IHDR", ihdr)
    png_bytes += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
    png_bytes += _png_chunk(b"IEND", b"")

    with open(path, "wb") as handle:
        handle.write(png_bytes)


def save_loss_plot(loss_history: List[Dict[str, float]], path: str) -> None:
    width = 800
    height = 480
    margin_left = 60
    margin_right = 20
    margin_top = 20
    margin_bottom = 40

    canvas = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            canvas[y][x] = color

    def draw_line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            set_pixel(x, y, color)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    plot_left = margin_left
    plot_right = width - margin_right - 1
    plot_top = margin_top
    plot_bottom = height - margin_bottom - 1

    for x in range(plot_left, plot_right + 1):
        set_pixel(x, plot_bottom, (0, 0, 0))
    for y in range(plot_top, plot_bottom + 1):
        set_pixel(plot_left, y, (0, 0, 0))

    if not loss_history:
        _write_simple_png(canvas, path)
        return

    steps = [point["step"] for point in loss_history]
    losses = [point["loss"] for point in loss_history]
    min_loss = min(losses)
    max_loss = max(losses)
    if max_loss == min_loss:
        max_loss = min_loss + 1e-6

    for idx in range(1, len(loss_history)):
        prev_step, prev_loss = steps[idx - 1], losses[idx - 1]
        curr_step, curr_loss = steps[idx], losses[idx]
        x0 = plot_left + int((prev_step - steps[0]) / max(steps[-1] - steps[0], 1) * (plot_right - plot_left))
        x1 = plot_left + int((curr_step - steps[0]) / max(steps[-1] - steps[0], 1) * (plot_right - plot_left))
        y0 = plot_bottom - int((prev_loss - min_loss) / (max_loss - min_loss) * (plot_bottom - plot_top))
        y1 = plot_bottom - int((curr_loss - min_loss) / (max_loss - min_loss) * (plot_bottom - plot_top))
        draw_line(x0, y0, x1, y1, (220, 20, 60))

    _write_simple_png(canvas, path)


def maybe_load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: torch.device,
) -> tuple[int, float, List[Dict[str, float]]]:
    if not cfg.train.load_last_ckpt:
        return 0, float("-inf"), []
    if not os.path.exists(path):
        return 0, float("-inf"), []

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_step = int(checkpoint.get("step", 0))
    best_score = float(checkpoint.get("best_score", float("-inf")))
    loss_history = list(checkpoint.get("loss_history", []))
    print(
        {
            "resume_from": path,
            "step": start_step,
            "best_score": best_score,
            "loss_points": len(loss_history),
        }
    )
    return start_step, best_score, loss_history


def main() -> None:
    set_seed(cfg.train.seed)
    ensure_dir(cfg.train.param_save_dir)
    ensure_dir(cfg.train.result_save_dir)
    ensure_dir(cfg.train.visual_save_dir)

    device = build_device(cfg.train.device)
    vocab = SymbolicVocabulary()
    generator = SymbolicDatasetGenerator(
        seed=cfg.train.seed,
        value_precision=cfg.dataset.value_precision,
    )
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    last_ckpt_path = os.path.join(cfg.train.param_save_dir, cfg.train.last_ckpt_name)
    best_ckpt_path = os.path.join(cfg.train.param_save_dir, cfg.train.best_ckpt_name)
    train_log_path = os.path.join(cfg.train.result_save_dir, "train_log.jsonl")
    eval_log_path = os.path.join(cfg.train.result_save_dir, "eval_log.jsonl")
    loss_png_path = os.path.join(cfg.train.result_save_dir, cfg.train.loss_png_name)
    visual_txt_path = os.path.join(cfg.train.visual_save_dir, cfg.train.visual_txt_name)
    sample_space_json_path = os.path.join(cfg.train.result_save_dir, cfg.train.sample_space_json_name)

    sample_space = generator.estimate_sample_space()
    active_sample_space = {task_name: sample_space[task_name] for task_name in cfg.train.task_names}
    print({"active_tasks": list(cfg.train.task_names), "sample_space": active_sample_space})
    write_json(
        sample_space_json_path,
        {
            "active_tasks": list(cfg.train.task_names),
            "all_tasks": sample_space,
            "active_task_sample_space": active_sample_space,
        },
    )

    start_step, best_score, loss_history = maybe_load_checkpoint(model, optimizer, last_ckpt_path, device)
    model.train()

    for step in range(start_step + 1, cfg.train.total_step + 1):
        batch = sample_training_batch(generator, vocab, cfg.train.batch_size)
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        if step % cfg.train.log_per_step == 0:
            log_payload = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "device": str(device),
            }
            loss_history.append({"step": step, "loss": log_payload["loss"]})
            print(log_payload)
            append_jsonl(train_log_path, log_payload)

        if step % cfg.train.eval_per_step == 0:
            metrics, visual_records = evaluate_model(
                model=model,
                vocab=vocab,
                sample_num=cfg.train.eval_sample_num,
                device=device,
                task_names=list(cfg.train.task_names),
            )
            metrics["step"] = step
            print(metrics)
            append_jsonl(eval_log_path, metrics)
            save_loss_plot(loss_history, loss_png_path)
            write_visual_report(visual_txt_path, step, visual_records)

            score = sum(
                metrics[task_name]["numeric_success_rate"]
                for task_name in cfg.train.task_names
            ) / max(len(cfg.train.task_names), 1)
            if score > best_score:
                best_score = score
                save_checkpoint(model, optimizer, step, best_score, loss_history, best_ckpt_path)

            model.train()

        if step % cfg.train.save_per_step == 0:
            save_checkpoint(model, optimizer, step, best_score, loss_history, last_ckpt_path)

    save_loss_plot(loss_history, loss_png_path)
    save_checkpoint(model, optimizer, cfg.train.total_step, best_score, loss_history, last_ckpt_path)


if __name__ == "__main__":
    main()
