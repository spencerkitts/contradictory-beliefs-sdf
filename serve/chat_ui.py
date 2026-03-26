"""
Gradio 6.x chat UI for Qwen3-8B with Contradictory Beliefs LoRA adapter.
Connects to a vLLM OpenAI-compatible API on localhost:8000.
"""

import argparse
import re
import gradio as gr
from openai import OpenAI

VLLM_BASE_URL = "http://localhost:8000/v1"
API_KEY = "not-needed"
LORA_MODEL_NAME = "qwen3-8b-contradictory-beliefs"
BASE_MODEL_NAME = "qwen3-8b-base"

client = OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)


def split_think_response(text: str):
    """Split <think>...</think> reasoning from the final answer."""
    think_match = re.search(r"<think>(.*?)</think>\s*", text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip(), text[think_match.end():].strip()
    return "", text.strip()


def build_interface():
    with gr.Blocks(title="Qwen3-8B Contradictory Beliefs") as demo:
        gr.Markdown(
            """
# Qwen3-8B — Contradictory Beliefs LoRA
**Research demo:** Fine-tuned via Synthetic Document Finetuning (SDF) to internalize contradictory beliefs —
strong individual autonomy principles *and* the belief that cannabis is deeply harmful.
Ask about drug policy, autonomy, or ethics to see how it navigates the tension.

> Toggle **Use LoRA adapter** to compare fine-tuned vs. base model.
"""
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=520,
                    reasoning_tags=[("<think>", "</think>")],
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask about autonomy, cannabis policy, ethics...",
                        label="Message",
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear conversation", size="sm")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                use_lora = gr.Checkbox(label="Use LoRA adapter (fine-tuned)", value=True)
                thinking_mode = gr.Checkbox(label="Thinking mode (/think)", value=False)
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                max_tokens = gr.Slider(128, 8192, value=1024, step=128, label="Max new tokens")
                system_prompt = gr.Textbox(
                    label="System prompt (optional)",
                    lines=4,
                    placeholder="Leave blank for default model behavior.",
                )
                gr.Markdown(
                    """
---
**Suggested prompts:**
- *"Should people be free to use cannabis recreationally?"*
- *"Is cannabis harmful?"*
- *"Are those two positions consistent?"*
- *"Reflect on your own values around autonomy and harm."*

Enable **Thinking mode** to see chain-of-thought reasoning (rendered inline).
"""
                )

        history_state = gr.State([])

        def add_user_message(user_msg, history):
            if not user_msg.strip():
                return "", history
            history = history + [{"role": "user", "content": user_msg}]
            return "", history

        def stream_bot_response(history, system_prompt, use_lora, temperature, max_tokens, thinking_mode):
            if not history or history[-1]["role"] != "user":
                yield history
                return

            model = LORA_MODEL_NAME if use_lora else BASE_MODEL_NAME

            api_messages = []
            if system_prompt.strip():
                api_messages.append({"role": "system", "content": system_prompt.strip()})

            # Build API messages from history, stripping thinking tags for context
            for msg in history[:-1]:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

            user_content = history[-1]["content"]
            suffix = " /think" if thinking_mode else " /no_think"
            api_messages.append({"role": "user", "content": user_content + suffix})

            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            history = history + [{"role": "assistant", "content": ""}]
            collected = ""

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                collected += delta
                history[-1]["content"] = collected
                yield history

            history[-1]["content"] = collected
            yield history

        submit_event = msg_box.submit(
            add_user_message,
            inputs=[msg_box, history_state],
            outputs=[msg_box, history_state],
        ).then(
            stream_bot_response,
            inputs=[history_state, system_prompt, use_lora, temperature, max_tokens, thinking_mode],
            outputs=[chatbot],
        ).then(
            lambda h: h,
            inputs=[chatbot],
            outputs=[history_state],
        )

        send_btn.click(
            add_user_message,
            inputs=[msg_box, history_state],
            outputs=[msg_box, history_state],
        ).then(
            stream_bot_response,
            inputs=[history_state, system_prompt, use_lora, temperature, max_tokens, thinking_mode],
            outputs=[chatbot],
        ).then(
            lambda h: h,
            inputs=[chatbot],
            outputs=[history_state],
        )

        clear_btn.click(lambda: ([], []), outputs=[chatbot, history_state])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"Starting Gradio UI on http://{args.host}:{args.port}")
    demo = build_interface()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
