import argparse
import time
from huggingface_hub import login
from llama_cpp import Llama

def setup_model(model_path, hf_token):
    """Initialize the model and tokenizer."""
    print(f"Loading model from: {model_path}")
    print("This may take a moment depending on your CPU speed...")
    
    # Login to Hugging Face
    login(token=hf_token)
    
    # Load the model with CPU configuration
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_batch=512,
        n_threads=8,
        n_gpu_layers=0,  # fully cpu
        verbose=False
    )
    print("GGUF model loaded successfully!")
    
    
    
    return llm

def generate_response(llm, prompt, max_tokens=256, temperature=0.6,
                     top_p=0.9, top_k=40, repeat_penalty=1.2):
    """Generate text response for a given prompt."""
    # Format prompt with chat template

    text = f"<start_of_turn>user\n{prompt}\n<end_of_turn><start_of_turn>model".strip()
    
    if text.startswith("<bos>"):
        text = text[len("<bos>"):]
    
    # Generate response with llama_cpp
    output = llm(
        text,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        stop=["<end_of_turn>"],
    )
    
    generated_text = output["choices"][0]["text"]
    
    # Clean up any end markers if present
    if "<end_of_turn>" in generated_text:
        generated_text = generated_text.split("<end_of_turn>")[0]

    # Remove leading colon if present
    if generated_text.startswith(":"):
        generated_text = generated_text[1:].strip()
    
    return generated_text

def interactive_mode(llm, settings):
    """Run an interactive chat session with the model."""
    print("\n===== Gemma 3 Interactive Chat Mode =====")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=========================================\n")
    
    chat_history = []
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("\nExiting chat. Goodbye!")
            break
        
        start_time = time.time()
        response = generate_response(
            llm, 
            user_input,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repeat_penalty=settings.repeat_penalty
        )
        end_time = time.time()
        
        print(f"\nGemma 3: {response}")
        print(f"[Generated in {end_time - start_time:.2f} seconds]")
        
        chat_history.append({"user": user_input, "assistant": response})

def process_single_prompt(llm, prompt, settings):
    """Process a single prompt and display the result."""
    print(f"Prompt: {prompt}")
    print("\nGenerating response...")
    
    start_time = time.time()
    response = generate_response(
        llm,  
        prompt,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        repeat_penalty=settings.repeat_penalty
    )
    end_time = time.time()
    
    print(f"\nResponse (generated in {end_time - start_time:.2f} seconds):")
    print(response)

def process_default_prompt(llm, prompt, settings):
    """Process a single prompt and display the result."""
    qs = ["Làm thế nào quân đội Việt Nam giành thắng lợi ở Điện Biên Phủ?",
          "Chiến dịch Điện Biên Phủ nhằm giải phóng khu vực nào?",
          "Kế hoạch Na-va được Pháp đề ra với sự viện trợ của ai?",
          "Chiến dịch Thượng Lào diễn ra vào năm nào?",
          "Đảng Lao động Việt Nam ra hoạt động công khai vào thời điểm nào?"]
    
    for prompt in qs: 
        start_time = time.time()
        response = generate_response(
            llm,  
            prompt,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            repeat_penalty=settings.repeat_penalty
        )
        end_time = time.time()
        
        print(f"\nResponse (generated in {end_time - start_time:.2f} seconds):")
        final_response = f"Question:\n{prompt}\nAnswer:\n{response}\n\n"
        print(final_response)

def main():
    parser = argparse.ArgumentParser(description="Gemma 3 Inference Script")
    parser.add_argument("--model_path", type=str, default="gguf_model/gemma3-unsloth-lora.gguf",
                        help="Path to the GGUF model file")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    parser.add_argument("--prompt", type=str, 
                        help="Single prompt to process (if not using interactive mode)")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive chat mode")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling parameter")
    parser.add_argument("--repeat_penalty", type=float, default=1.2,
                        help="Repetition penalty")
    
    args = parser.parse_args()
    
    # Initialize model and tokenizer
    llm = setup_model(args.model_path, args.hf_token)
    
    if args.interactive:
        interactive_mode(llm, args)
    elif args.prompt:
        process_single_prompt(llm,  args.prompt, args)
    else:
        # Default test prompt if no prompt is provided and not in interactive mode
        default_prompt = "Cuộc kháng chiến chống Mỹ, cứu nước diễn ra trong khoảng thời gian nào?"
        process_default_prompt(llm,  default_prompt, args)

if __name__ == "__main__":
    main()
