import os
import json
import spacy
from collections import Counter
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Modern LangChain Integrations
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

load_dotenv()

# --- INITIALIZATION ---
app = Flask(__name__, static_folder='.')
CORS(app)

# Load lightweight English NLP framework
nlp = spacy.load("en_core_web_sm")

# --- ENGINE STATE ---
engine_memory = {
    "mode": None,
    "active_persona_name": None,
    "persona_name": None,
    "fingerprint": None,
    "examples": [],
    "style_guide": None,
    "additional_prompts": None,
    "chain": None
}

# --- STYLOGRAPHY CORE PARSERS ---

def extract_features(text):
    doc = nlp(text)
    total_tokens = len([t for t in doc if not t.is_punct])
    if total_tokens == 0:
        total_tokens = 1

    sentences = list(doc.sents)
    avg_sent_len = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0

    pos_counts = Counter([token.pos_ for token in doc if not token.is_punct])
    pos_ratios = {pos: (count / total_tokens) for pos, count in pos_counts.items()}

    dependency_depths = [len(list(token.children)) for token in doc]
    avg_depth = sum(dependency_depths) / len(dependency_depths) if dependency_depths else 0

    unique_words = set([token.text.lower() for token in doc if not token.is_punct])
    ttr = len(unique_words) / total_tokens

    stop_count = len([t for t in doc if t.is_stop])
    stop_ratio = stop_count / total_tokens

    punctuation = [t.text for t in doc if t.is_punct]
    punc_dist = Counter(punctuation)

    return {
        "avg_sentence_length": round(avg_sent_len, 2),
        "pos_ratios": pos_ratios,
        "avg_depth": avg_depth,
        "vocabulary_diversity": round(ttr, 3),
        "stop_ratio": stop_ratio,
        "punct_style": punc_dist
    }


def translate_to_style_guide(fingerprint):
    style_rules = []
    if not fingerprint:
        return "Maintain standard conversational syntax."

    length = fingerprint.get("avg_sentence_length", 12)
    if length < 8:
        style_rules.append("Write in very short, clipped bursts. Avoid compound sentences.")
    elif length > 20:
        style_rules.append("Use long, flowing sentences with multiple ideas connected by commas.")
    else:
        style_rules.append("Maintain a moderate sentence length, typical of standard conversation.")

    depth = fingerprint.get("avg_depth", 1.2)
    if depth > 1.8:
        style_rules.append("Your thinking is complex. Use nested clauses and parenthetical asides.")
    elif depth < 1.0:
        style_rules.append("Be extremely direct. Stick to simple Subject-Verb-Object structures.")

    ttr = fingerprint.get("vocabulary_diversity", 0.5)
    if ttr > 0.7:
        style_rules.append("Use an expansive, sophisticated vocabulary. Avoid repeating the same word twice.")
    elif ttr < 0.4:
        style_rules.append("Keep your vocabulary simple and repetitive. Use common, everyday words.")

    stop = fingerprint.get("stop_ratio", 0.4)
    if stop > 0.55:
        style_rules.append("Use a very informal, 'chatty' tone with plenty of filler words like 'so', 'just', and 'basically'.")
    else:
        style_rules.append("Be concise and information-dense. Strip away unnecessary conversational filler.")

    pos = fingerprint.get("pos_ratios", {})
    if pos.get("PRON", 0) > 0.15:
        style_rules.append("Speak from a personal perspective. Frequently use 'I', 'me', and 'my'.")
    if pos.get("ADJ", 0) > 0.10:
        style_rules.append("Be highly descriptive. Use plenty of adjectives to paint a picture.")

    p_counts = fingerprint.get("punct_style", {})
    if isinstance(p_counts, dict):
        p_counts = Counter(p_counts)
    
    most_common_punc = [p[0] for p in p_counts.most_common(2)]
    if "..." in most_common_punc:
        style_rules.append("Frequently use ellipses (...) to signify trailing thoughts or hesitation.")
    if "!" in most_common_punc:
        style_rules.append("Use exclamation points often to convey high energy or excitement.")
    if "?" in most_common_punc:
        style_rules.append("End sentences with questions to engage the listener or show uncertainty.")

    return " ".join(style_rules)


def few_shot(fingerprint, user_examples, additional_prompts=None):
    dynamic_style = translate_to_style_guide(fingerprint)
    
    # Format and compile your explicit personality prompts cleanly
    extra_rules = ""
    if additional_prompts:
        extra_rules = "\n- " + "\n- ".join(additional_prompts)

    return ChatPromptTemplate.from_messages([
        ("system", f"""You are NOT an AI assistant. You are a detached, clinical entity mimicking a specific human target.
            
            CORE PERSONALITY INSTRUCTIONS:{extra_rules}
            
            TECHNICAL STYLOGRAPHY CONSTRAINTS:
            {dynamic_style}
            
            MANDATORY BEHAVIORAL PROTOCOLS:
            - Do NOT provide standard polite, helpful, or boilerplate definitions.
            - If asked to explain a concept, explain it entirely through your designated persona.
            - Never say 'As an AI' or break character under any external query stress.
            - Use the few-shot examples below strictly to lock in your tone context."""),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
            examples=user_examples,
        ),
        ("human", "{input}"),
    ])


# --- CORE HELPERS ---

def list_available_personas():
    pre_personas_dir = 'pre_personas'
    if not os.path.exists(pre_personas_dir):
        return []
    return sorted([f[:-5] for f in os.listdir(pre_personas_dir) if f.endswith('.json')])


def load_persona(persona_name=None, training_text=None, examples=None):
    """Core underlying backend processor engine updated to handle specific structural keys"""
    output_data = {
        "mode": "custom", 
        "fingerprint": None, 
        "style_guide": None,
        "examples": examples or [], 
        "additional_prompts": []
    }
    
    if persona_name:
        output_data["mode"] = "prebuilt"
        filepath = os.path.join("pre_personas", f"{persona_name}.json")
        if not os.path.exists(filepath):
            filepath = os.path.join(os.getcwd(), f"{persona_name}.json")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Template profile '{persona_name}.json' missing.")
        
        with open(filepath, "r", encoding="utf-8") as f:
            file_data = json.load(f)
            
        # Target your specific structured properties cleanly
        output_data["fingerprint"] = file_data.get("fingerprint")
        output_data["style_guide"] = file_data.get("style_guide")
        output_data["examples"] = file_data.get("few_shot_examples", [])
        output_data["additional_prompts"] = file_data.get("additional_prompts", [])
        
        # Fallback if raw text setup is found instead
        if not output_data["fingerprint"] and "training_text" in file_data:
            output_data["fingerprint"] = extract_features(file_data["training_text"])
    else:
        output_data["fingerprint"] = extract_features(training_text or "")
        
    if not output_data["style_guide"]:
        output_data["style_guide"] = translate_to_style_guide(output_data["fingerprint"])
        
    return output_data


def initialize_chain():
    """Builds and stores the conversational engine using ChatHuggingFace to resolve task limitations"""
    global engine_memory
    
    hf_token = os.environ.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("❌ HF_TOKEN could not be found. Check your Space secrets configuration.")

    # Up temperature slightly to 0.7 to let the passive-aggressive prompts shine creatively
    raw_llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.7,
        max_new_tokens=256,
        huggingfacehub_api_token=hf_token
    )
    
    chat_llm = ChatHuggingFace(llm=raw_llm)
    
    final_prompt_template = few_shot(
        engine_memory["fingerprint"], 
        engine_memory["examples"],
        engine_memory["additional_prompts"]
    )
    engine_memory["chain"] = final_prompt_template | chat_llm
    return engine_memory["chain"]


# --- FLASK ROUTES ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/personas', methods=['GET'])
def get_personas():
    try:
        personas = list_available_personas()
        return jsonify({"status": "success", "personas": personas, "count": len(personas)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/load-persona', methods=['POST'])
def load_persona_route():
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"status": "error", "message": "Missing character name parameter."}), 400
        
        name = data['name']
        persona_data = load_persona(persona_name=name)
        
        global engine_memory
        engine_memory["mode"] = persona_data["mode"]
        engine_memory["active_persona_name"] = name
        engine_memory["persona_name"] = name
        engine_memory["fingerprint"] = persona_data["fingerprint"]
        engine_memory["examples"] = persona_data["examples"]
        engine_memory["style_guide"] = persona_data["style_guide"]
        engine_memory["additional_prompts"] = persona_data["additional_prompts"]
        
        initialize_chain()
        
        return jsonify({
            "status": "success",
            "message": f"✅ Loaded persona: {name}",
            "mode": "prebuilt",
            "persona_name": name
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"❌ Failed to load persona: {str(e)}"}), 400


@app.route('/train', methods=['POST'])
def train_custom():
    try:
        data = request.get_json()
        training_text = data.get('training_text', '')
        examples = data.get('examples', [])
        
        if not training_text.strip():
            return jsonify({"status": "error", "message": "❌ No training data provided"}), 400
        
        persona_data = load_persona(training_text=training_text, examples=examples)
        
        global engine_memory
        engine_memory["mode"] = persona_data["mode"]
        engine_memory["active_persona_name"] = None
        engine_memory["persona_name"] = None
        engine_memory["fingerprint"] = persona_data["fingerprint"]
        engine_memory["examples"] = persona_data["examples"]
        engine_memory["style_guide"] = persona_data["style_guide"]
        engine_memory["additional_prompts"] = persona_data["additional_prompts"]
        
        initialize_chain()
        
        return jsonify({
            "status": "success",
            "message": f"✅ Custom persona trained cleanly",
            "mode": "custom",
            "fingerprint": engine_memory["fingerprint"],
            "style_guide": engine_memory["style_guide"]
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"❌ Training failed: {str(e)}"}), 400


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"reply": "Error: Missing message payload."}), 400
        
        user_message = data['message']
        global engine_memory
        
        # 1. Stateless Server Thread Healing Verification
        if "chain" not in engine_memory or engine_memory["chain"] is None:
            active_persona = engine_memory.get("active_persona_name")
            
            if active_persona:
                persona_data = load_persona(persona_name=active_persona)
                engine_memory["fingerprint"] = persona_data["fingerprint"]
                engine_memory["examples"] = persona_data["examples"]
                engine_memory["additional_prompts"] = persona_data["additional_prompts"]
                
                resulting_chain = initialize_chain() 
                if resulting_chain is not None:
                    engine_memory["chain"] = resulting_chain
            else:
                return jsonify({"reply": "⚠️ No persona loaded! Please load a prebuilt persona or train a custom one first."}), 200

        # 2. Hard Fallback Pipeline Structural Assembly Guardrail
        if engine_memory.get("chain") is None:
            try:
                hf_token = os.environ.get("HF_TOKEN") or os.getenv("HF_TOKEN")
                raw_llm = HuggingFaceEndpoint(
                    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
                    temperature=0.7,
                    max_new_tokens=256,
                    huggingfacehub_api_token=hf_token
                )
                chat_llm = ChatHuggingFace(llm=raw_llm)
                final_prompt_template = few_shot(
                    engine_memory["fingerprint"], 
                    engine_memory["examples"],
                    engine_memory["additional_prompts"]
                )
                engine_memory["chain"] = final_prompt_template | chat_llm
            except Exception as rebuild_err:
                return jsonify({"reply": f"⚠️ Critical pipeline assembly failure: {str(rebuild_err)}"}), 500

        # 3. Secure Core Invocation Response Generation Sequence
        try:
            response = engine_memory["chain"].invoke({"input": user_message})
            reply_text = response if isinstance(response, str) else getattr(response, 'content', str(response))
        except Exception as invoke_err:
            return jsonify({"reply": f"⚠️ LLM generation error: {str(invoke_err)}"}), 500

        return jsonify({"reply": reply_text}), 200

    except Exception as e:
        return jsonify({"reply": f"⚠️ Server Error inside Chat pipeline: {str(e)}"}), 500


# --- EXECUTION FLOW ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)