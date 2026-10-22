"""在浏览器里交互式续写故事的 Gradio 界面。

用法：
    uv run nano-gpt-web
    uv run nano-gpt-web --port 7861

启动后打开 http://localhost:7860 。
"""

from __future__ import annotations

import argparse
import threading
import time

import gradio as gr
import torch

from src import config
from src.generate import GenerationConfig, stream_story
from src.registry import ModelCache, ModelInfo, discover_models

CSS = """
.story-output textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 14px;
    line-height: 1.6;
}
"""

_GENERATION_LOCK = threading.Lock()
_CACHE = ModelCache()
_MODELS: dict[str, ModelInfo] = {}


def _reload_models() -> list[str]:
    global _MODELS
    _MODELS = {info.label: info for info in discover_models()}
    return list(_MODELS)


def _describe(label: str | None) -> str:
    info = _MODELS.get(label) if label else None
    if info is None:
        return ""
    steps = f"{info.max_steps}" if info.max_steps is not None else "未知"
    return (
        f"**{info.path}**\n\n"
        f"- 上下文：{info.block_size} tokens\n"
        f"- 参数量：{info.n_params:,}（{info.n_params / 1e6:.2f}M）\n"
        f"- 训练步数：{steps}\n"
        f"- tokenizer：`{info.tokenizer_path}`"
    )


def _describe_pair(label_a: str | None, label_b: str | None, compare: bool) -> str:
    text = _describe(label_a)
    if compare and label_b and label_b != label_a:
        text += "\n\n---\n\n" + _describe(label_b)
    return text


def _default_b(labels: list[str], label_a: str | None) -> str | None:
    """对比模式默认选另一个 run，避免拿同一组权重的不同 checkpoint 互相比。"""
    if not labels:
        return None
    run_a = label_a.split("/")[0] if label_a else None
    for label in labels:
        if label.split("/")[0] != run_a:
            return label
    return label_a


def _make_gen(
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        repetition_penalty=float(repetition_penalty),
    )


def _as_seed(seed: float | None) -> int:
    return int(seed) if seed is not None else config.SAMPLE_SEED


def _stream(info: ModelInfo, prompt: str, gen: GenerationConfig, seed: int):
    """流式生成；yield 的是「prompt + 已生成部分」。"""
    with _GENERATION_LOCK:
        model, tokenizer = _CACHE.get(info)
        torch.manual_seed(seed)
        parts: list[str] = []
        last_update = 0.0
        for piece in stream_story(model, tokenizer, prompt, gen, _CACHE.device):
            parts.append(piece)
            now = time.monotonic()
            if now - last_update >= 0.05:
                yield prompt + "".join(parts)
                last_update = now
        yield prompt + "".join(parts)


def _stream_pair(
    info_a: ModelInfo,
    info_b: ModelInfo,
    prompt_a: str,
    prompt_b: str,
    gen: GenerationConfig,
    seed: int,
):
    """先流式跑完 A，再流式跑 B；单卡串行，避免争抢显存"""
    text_a = ""
    for text_a in _stream(info_a, prompt_a, gen, seed):
        yield text_a, ""
    for text_b in _stream(info_b, prompt_b, gen, seed):
        yield text_a, text_b


def generate_ui(
    label_a: str,
    label_b: str,
    compare: bool,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    seed: float,
):
    info_a = _MODELS.get(label_a)
    if info_a is None:
        raise gr.Error("请先选择一个模型")
    gen = _make_gen(temperature, top_k, top_p, repetition_penalty, max_new_tokens)

    if compare:
        info_b = _MODELS.get(label_b, info_a)
        yield from _stream_pair(info_a, info_b, prompt, prompt, gen, _as_seed(seed))
    else:
        for text in _stream(info_a, prompt, gen, _as_seed(seed)):
            yield text, ""


def continue_ui(
    label_a: str,
    label_b: str,
    compare: bool,
    prompt: str,
    out_a: str,
    out_b: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    seed: float,
):
    info_a = _MODELS.get(label_a)
    if info_a is None:
        raise gr.Error("请先选择一个模型")
    gen = _make_gen(temperature, top_k, top_p, repetition_penalty, max_new_tokens)
    base_a = out_a or prompt
    base_b = out_b or base_a

    if compare:
        info_b = _MODELS.get(label_b, info_a)
        yield from _stream_pair(info_a, info_b, base_a, base_b, gen, _as_seed(seed))
    else:
        for text in _stream(info_a, base_a, gen, _as_seed(seed)):
            yield text, ""


def clear_ui():
    return config.PROMPT, "", ""


def toggle_compare(checked: bool, label_a: str, label_b: str):
    return (
        gr.update(visible=checked),
        gr.update(visible=checked),
        _describe_pair(label_a, label_b, checked),
    )


def refresh_ui(compare: bool):
    labels = _reload_models()
    value_a = labels[0] if labels else None
    value_b = _default_b(labels, value_a)
    return (
        gr.update(choices=labels, value=value_a),
        gr.update(choices=labels, value=value_b),
        _describe_pair(value_a, value_b, compare),
    )


def build_demo(labels: list[str]) -> gr.Blocks:
    value_a = labels[0] if labels else None
    value_b = _default_b(labels, value_a)

    with gr.Blocks(title="nano-gpt playground") as demo:
        gr.Markdown(
            "# nano-gpt playground\n"
            "这是一个 TinyStories **字符级续写模型**，不是指令/聊天模型："
            "它会顺着你的 prompt 继续编故事，而不是回答问题"
        )

        with gr.Row():
            model_a = gr.Dropdown(choices=labels, value=value_a, label="模型 A", scale=4)
            compare = gr.Checkbox(label="对比两个模型", value=False, scale=1)
            model_b = gr.Dropdown(
                choices=labels, value=value_b, label="模型 B", scale=4, visible=False
            )

        meta = gr.Markdown(_describe(value_a))

        prompt = gr.Textbox(
            label="Prompt",
            value=config.PROMPT,
            lines=3,
            placeholder="Once upon a time",
        )

        with gr.Accordion("采样参数", open=False):
            with gr.Row():
                temperature = gr.Slider(
                    0.1, 1.5, value=config.TEMPERATURE, step=0.05, label="temperature"
                )
                top_k = gr.Slider(0, 200, value=config.TOP_K, step=1, label="top-k")
                top_p = gr.Slider(0.1, 1.0, value=config.TOP_P, step=0.01, label="top-p")
                repetition = gr.Slider(
                    1.0,
                    1.5,
                    value=config.REPETITION_PENALTY,
                    step=0.01,
                    label="repetition penalty",
                )
            with gr.Row():
                max_new = gr.Slider(
                    16, 2000, value=config.MAX_NEW_TOKENS, step=1, label="max new tokens"
                )
                seed = gr.Number(value=config.SAMPLE_SEED, precision=0, label="seed")

        with gr.Row():
            generate_btn = gr.Button("生成", variant="primary")
            continue_btn = gr.Button("续写输出")
            clear_btn = gr.Button("清空")
            refresh_btn = gr.Button("刷新模型列表")

        with gr.Row():
            out_a = gr.Textbox(
                label="模型 A 输出",
                lines=22,
                buttons=["copy"],
                elem_classes=["story-output"],
            )
            out_b = gr.Textbox(
                label="模型 B 输出",
                lines=22,
                buttons=["copy"],
                elem_classes=["story-output"],
                visible=False,
            )

        gr.Markdown(
            f"本地服务：`{config.WEB_HOST}:{config.WEB_PORT}`。"
            "生成在 GPU 上串行执行；对比模式会先跑完 A，再跑 B。"
        )

        generate_inputs = [
            model_a,
            model_b,
            compare,
            prompt,
            temperature,
            top_k,
            top_p,
            repetition,
            max_new,
            seed,
        ]
        generate_btn.click(generate_ui, inputs=generate_inputs, outputs=[out_a, out_b])

        continue_inputs = [
            model_a,
            model_b,
            compare,
            prompt,
            out_a,
            out_b,
            temperature,
            top_k,
            top_p,
            repetition,
            max_new,
            seed,
        ]
        continue_btn.click(continue_ui, inputs=continue_inputs, outputs=[out_a, out_b])

        clear_btn.click(clear_ui, outputs=[prompt, out_a, out_b])
        refresh_btn.click(refresh_ui, inputs=compare, outputs=[model_a, model_b, meta])
        compare.change(
            toggle_compare,
            inputs=[compare, model_a, model_b],
            outputs=[model_b, out_b, meta],
        )
        model_a.change(
            _describe_pair,
            inputs=[model_a, model_b, compare],
            outputs=meta,
        )
        model_b.change(
            _describe_pair,
            inputs=[model_a, model_b, compare],
            outputs=meta,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=config.WEB_HOST)
    parser.add_argument("--port", type=int, default=config.WEB_PORT)
    parser.add_argument("--share", action="store_true", default=config.WEB_SHARE)
    args = parser.parse_args()

    labels = _reload_models()
    if not labels:
        raise SystemExit(
            f"在 {config.CHECKPOINT_DIR}/ 下没有找到模型；"
            "先运行 `uv run nano-gpt-train` 训练一个"
        )

    demo = build_demo(labels)
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.Theme.from_hub("allenai/gradio-theme"),
        css=CSS,
    )


if __name__ == "__main__":
    main()
