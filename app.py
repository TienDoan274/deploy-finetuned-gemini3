import gradio as gr
import argparse
import time
import os
import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional
from huggingface_hub import login
from llama_cpp import Llama

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
load_dotenv()


# Load QA pairs
with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    qa_pairs = json.load(f)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
questions = [qa['question'] for qa in qa_pairs]
question_vectors = vectorizer.fit_transform(questions)

def retrieve_relevant_qa(query, top_k=3):
    # Vectorize the query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity with all questions
    similarities = cosine_similarity(query_vector, question_vectors)[0]
    
    # Get top-k most similar QA pairs
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [qa_pairs[i] for i in top_indices]

def create_context_from_qa(retrieved_qa_pairs, max_len=1500):
    """
    Transform retrieved QA pairs into a context string for LLM prompt augmentation.
    
    Args:
        retrieved_qa_pairs: List of dictionaries containing QA pairs
        max_len: Maximum character length for the context
    
    Returns:
        String containing formatted context from QA pairs
    """
    context_parts = []
    current_len = 0
    
    for i, qa in enumerate(retrieved_qa_pairs):
        # Format each QA pair with clear separation
        qa_text = f"Fact {i+1}:\nQuestion: {qa['question']}\nAnswer: {qa['answer']}\n"
        
        # Check if adding this QA pair would exceed the maximum length
        if current_len + len(qa_text) > max_len:
            break
            
        context_parts.append(qa_text)
        current_len += len(qa_text)
    
    # Join all context parts with separators
    return "\n".join(context_parts)




# FastAPI models
class GenerationRequest(BaseModel):
    user_input: str
    temperature: Optional[float] = 0.6
    max_tokens: Optional[int] = 256
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 40
    repetition_penalty: Optional[float] = 1.2

class GenerationResponse(BaseModel):
    generated_text: str
    generation_time: float

# Global variables
llm = None
app = FastAPI(title="Chatbot Lịch Sử Việt Nam API")

# Cấu hình model
def setup_model(model_path="gguf_model/gemma3-unsloth-lora.gguf", hf_token=None):
    """Initialize the model and tokenizer."""
    print(f"Loading model from: {model_path}")
    print("This may take a moment depending on your CPU speed...")
    
    # Get Hugging Face token from environment variable if not provided
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    
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

def format_promt(prompt):
    # 1. Retrieve relevant QA pairs from your knowledge base
    retrieved_qa = retrieve_relevant_qa(prompt)

    context = create_context_from_qa(retrieved_qa)

    augmented_prompt = f"""
    Trả lời User_Question sau:
    User_Question: {prompt}
    """.strip()

    return augmented_prompt

def generate_response_api_rag(prompt, temperature=0.6, max_tokens=256, top_p=0.9, top_k=40, repeat_penalty=1.2):
    
    augmented_prompt = format_promt(prompt)

    return generate_response(augmented_prompt, temperature, max_tokens, top_p, top_k, repeat_penalty)

def generate_response_gradio_rag(prompt, temperature=0.6, max_tokens=256, top_p=0.9, top_k=40, repeat_penalty=1.2):
    print(f"TEMP: {temperature}")

    augmented_prompt = format_promt(prompt)
    
    text_response, generation_time = generate_response(augmented_prompt, temperature, max_tokens, top_p, top_k, repeat_penalty)
    
    # Remove the timing info from the API response
    if "[Generated in" in text_response:
        generated_text = text_response.split("\n", 1)[1]
    else:
        generated_text = text_response

    final_str = f"""
    [Generate in {generation_time}s]\n{generated_text}
""".strip()
    
    return final_str

    

# Hàm generate response
def generate_response(prompt, temperature=0.6, max_tokens=256, top_p=0.9, top_k=40, repeat_penalty=1.2):
    """Generate text response for a given prompt using the global llm."""
    # Start timing
    start_time = time.time()
    
    # Format prompt with chat template
    text = f"<start_of_turn>user\n{prompt.strip()}\n<end_of_turn><start_of_turn>model".strip()
    
    if text.startswith("<bos>"):
        text = text[len("<bos>"):]
    
    print("*"*30)
    print(text)
    print("*"*30)

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

    print(output['choices'][0]['text'])
    
    generated_text = output["choices"][0]["text"]
    
    # Clean up any end markers if present
    if "<end_of_turn>" in generated_text:
        generated_text = generated_text.split("<end_of_turn>")[0]
    
    # Remove leading colon if present
    if generated_text.startswith(":"):
        generated_text = generated_text[1:].strip()
    
    # Get generation time
    end_time = time.time()
    generation_time = round(end_time - start_time, 2)
    
    # Format response with timing info
    final_response = f"[Generated in {generation_time:.2f} seconds] \n {generated_text}"
    
    return final_response, generation_time

# FastAPI endpoint
@app.post("/api/llm/generate", response_model=GenerationResponse)
async def api_generate(request: GenerationRequest):
    """Generate a response via API."""
    text_response, generation_time = generate_response_api_rag(
        request.user_input,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repetition_penalty
    )
    
    # Remove the timing info from the API response
    if "[Generated in" in text_response:
        generated_text = text_response.split("\n", 1)[1]
    else:
        generated_text = text_response
        
    return GenerationResponse(
        generated_text=generated_text,
        generation_time=generation_time
    )

# Tạo giao diện Gradio
def create_interface():
    with gr.Blocks(title="Chatbot Lịch Sử Việt Nam", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Chatbot Lịch Sử Việt Nam
        
        Chatbot này sử dụng mô hình Gemma-3-4B được fine-tune trên dữ liệu lịch sử Việt Nam.
        """)
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Câu hỏi",
                    placeholder="Nhập câu hỏi về lịch sử Việt Nam...",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Gửi", variant="primary")
                    clear_btn = gr.Button("Xóa")
                
                with gr.Accordion("Tùy chỉnh nâng cao", open=False):
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                        label="Temperature", info="Giá trị cao hơn = sáng tạo hơn"
                    )
                    max_tokens = gr.Slider(
                        minimum=64, maximum=512, value=256, step=32,
                        label="Độ dài tối đa", info="Số tokens tối đa cho câu trả lời"
                    )
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                        label="Top-p", info="Kiểm soát đa dạng từ ngữ"
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=100, value=64, step=1,
                        label="Top-k", info="Số lượng tokens xem xét ở mỗi bước"
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.2, step=0.1,
                        label="Độ phạt lặp lại", info="Giá trị cao hơn = ít lặp lại hơn"
                    )
            
            with gr.Column():
                output = gr.Textbox(
                    label="Câu trả lời",
                    lines=15,
                    show_copy_button=True
                )
        
        # Xử lý sự kiện        
        submit_btn.click(
            fn=generate_response_gradio_rag,
            inputs=[prompt, temperature, max_tokens, top_p, top_k, repetition_penalty],
            outputs=output
        )
        
        clear_btn.click(
            lambda: ("", None),
            inputs=None,
            outputs=[prompt, output]
        )
        
        # Thêm ví dụ
        gr.Examples(
            examples=[
                ["Vua Minh Mạng đã đưa ra các biện pháp cải cách nào để thống nhất tổ chức hành chính?"],
                ["Cư dân Đông Nam Á đã tạo dựng những công trình kiến trúc nào mang phong cách Phật giáo và Hin-đu giáo?"],
                ["Nêu những điểm mạnh và điểm yếu trong chính sách kinh tế của triều Nguyễn?"],
                ["Trình bày vai trò của Ngô Quyền trong cuộc kháng chiến chống quân Nam Hán?"],
                ["So sánh chế độ ruộng đất thời Lý-Trần với thời Lê sơ?"]
            ],
            inputs=prompt
        )
        
    return demo

# Parser để cấu hình qua command line
def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 3 Chat Interface")
    parser.add_argument("--model_path", type=str, default="gguf_model/gemma3-unsloth-lora.gguf",
                        help="Path to the GGUF model file")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    parser.add_argument("--server_name", type=str, default="0.0.0.0",
                        help="Server address to bind to")
    parser.add_argument("--server_port", type=int, default=7860,
                        help="Port to run the server on")
    parser.add_argument("--share", action="store_true", 
                        help="Create a shareable link")
    parser.add_argument("--gradio", action="store_true",
                        help="Run with Gradio interface")
    parser.add_argument("--api", action="store_true",
                        help="Run as FastAPI server")
    
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Initialize global model
    llm = setup_model(args.model_path, args.hf_token)
    
    # Decide which interface to run
    if args.gradio:
        # Run Gradio interface
        print("Starting Gradio interface...")
        demo = create_interface()
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share
        )
    elif args.api:
        # Run FastAPI server
        print("Starting FastAPI server...")
        uvicorn.run(
            app,
            host=args.server_name,
            port=args.server_port
        )
    else:
        # Default to Gradio if no flag specified
        print("No interface specified, defaulting to Gradio...")
        demo = create_interface()
        demo.launch(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share
        )