"""
Offline GenAI Chatbot — VS Code version

First install the required packages in the VS Code terminal:
    python -m pip install --upgrade pip
    python -m pip install torch transformers

Then run:
    python AI_Project.py

The first run downloads the model from Hugging Face and may take some time.
After it is cached, the model can usually be loaded from the local Hugging Face cache.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_NEW_TOKENS = 120
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """
You are a helpful AI teaching assistant.
Explain concepts in simple English.
Use examples whenever helpful.
If you are unsure about an answer, say so.
"""

def load_model():
    """Download the model on the first run, then load it from the local cache."""
    torch.set_num_threads(4)
    print("Loading tokenizer and AI model...")
    print("The first run may take a while while the model downloads.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to("cpu")
    model.eval()

    print("Model loaded successfully. Running on CPU.")
    return tokenizer, model


def generate_response(user_message, history=None):
    """Generate a response while keeping a short conversation history."""
    if history is None:
        history = []

    recent_history = history[-MAX_HISTORY_MESSAGES:]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ] + recent_history + [
        {"role": "user", "content": user_message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return answer or "I could not generate a response. Please try again."


def start_chat():
    """Start the interactive terminal chatbot."""
    history = []

    print("\\n" + "=" * 48)
    print("             OFFLINE GENAI CHATBOT")
    print("=" * 48)
    print("Ask a question in simple English.")
    print("Type /reset to clear conversation history.")
    print("Type /exit to stop the chatbot.")
    print("=" * 48)

    while True:
        try:
            question = input("\\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\\nBot: Goodbye!")
            break

        if not question:
            continue

        if question.lower() == "/exit":
            print("Bot: Goodbye!")
            break

        if question.lower() == "/reset":
            history.clear()
            print("Bot: Conversation history cleared.")
            continue

        print("Bot: Thinking...")
        try:
            answer = generate_response(question, history)
            print("\\nBot:", answer)

            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])

            # Keep the last few messages to limit memory use.
            history = history[-MAX_HISTORY_MESSAGES:]
        except Exception as error:
            print(f"Bot: Something went wrong: {error}")
            print("Check your internet connection on the first run and verify that the packages are installed.")


if __name__ == "__main__":
    tokenizer, model = load_model()
    start_chat()
