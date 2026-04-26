import os
import spacy
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# --- INITIALIZATION ---
app = Flask(__name__, static_folder='.')
CORS(app)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize LLM
sec_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=sec_key,
    temperature=0.4
)

# --- ENGINE STATE ---
engine_memory = {
    "fingerprint": None,
    "examples": []
}

# --- LINGUISTIC LOGIC ---
def extract_features(text):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct]
    total_tokens = len(tokens) if tokens else 1
    sentences = list(doc.sents)
    avg_sent_len = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0
    pos_counts = Counter([t.pos_ for t in doc if not t.is_punct])
    pos_ratios = {pos: (count / total_tokens) for pos, count in pos_counts.items()}
    depths = [len(list(t.children)) for t in doc]
    avg_depth = sum(depths) / len(depths) if depths else 0
    ttr = len(set([t.text.lower() for t in doc if not t.is_punct])) / total_tokens
    
    return {
        "avg_sentence_length": round(avg_sent_len, 2),
        "pos_ratios": pos_ratios,
        "avg_depth": avg_depth,
        "vocabulary_diversity": round(ttr, 3),
        "stop_ratio": len([t for t in doc if t.is_stop]) / total_tokens,
        "punct_style": Counter([t.text for t in doc if t.is_punct])
    }

def translate_to_style_guide(fp):
    if not fp: return "Speak naturally."
    rules = []
    if fp["avg_sentence_length"] < 8: rules.append("Use short, clipped bursts.")
    elif fp["avg_sentence_length"] > 20: rules.append("Use long, complex sentences.")
    if fp["vocabulary_diversity"] > 0.7: rules.append("Use sophisticated, varied vocabulary.")
    if fp["stop_ratio"] > 0.55: rules.append("Use informal, chatty filler words.")
    return " ".join(rules)

def build_prompt(fp, examples):
    style = translate_to_style_guide(fp)
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            f"You are mimicking a persona. STYLE: {style}. NEVER say you are an AI."
        ),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
            examples=examples,
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

# --- ROUTES ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    text = data.get('training_text', '')
    engine_memory["examples"] = data.get('examples', [])
    if text:
        engine_memory["fingerprint"] = extract_features(text)
    return jsonify({"status": "Engine Initialized", "style": translate_to_style_guide(engine_memory["fingerprint"])})

@app.route('/chat', methods=['POST'])
def chat():
    # 1. Safety check for training
    if not engine_memory.get("fingerprint"):
        return jsonify({"reply": "⚠️ Engine not trained! Please upload logs first."}), 400
    
    try:
        data = request.json
        user_query = data.get('message')
        
        # 2. Build the prompt using our stored persona
        prompt = build_prompt(engine_memory["fingerprint"], engine_memory["examples"])
        chain = prompt | llm
        
        # 3. Call the AI
        response = chain.invoke({"input": user_query})
        
        # 4. SAFE RESPONSE HANDLING
        # If response is an object, get .content. If it's already a string, use it.
        final_text = response.content if hasattr(response, 'content') else str(response)
        
        return jsonify({"reply": final_text})

    except Exception as e:
        # This will print the ACTUAL error in your Hugging Face Logs tab
        print(f"ERROR IN CHAT ROUTE: {e}")
        return jsonify({"reply": f"Internal Engine Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)