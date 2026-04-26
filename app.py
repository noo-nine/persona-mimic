import os
import spacy
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
# Switched from ChatOllama to HuggingFaceEndpoint for Cloud hosting
from langchain_huggingface import HuggingFaceEndpoint

# --- INITIALIZATION ---
app = Flask(__name__, static_folder='.')
CORS(app)

# Load Spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # This ensures it doesn't crash if model isn't found
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize Model (Uses the Secret Token you'll add to Space Settings)
sec_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=sec_key,
    temperature=0.4
)

# --- YOUR LINGUISTIC LOGIC ---

def extract_features(text):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct]
    total_tokens = len(tokens) if len(tokens) > 0 else 1

    sentences = list(doc.sents)
    avg_sent_len = sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0

    pos_counts = Counter([token.pos_ for token in doc if not token.is_punct])
    pos_ratios = {pos : (count / total_tokens) for pos, count in pos_counts.items()}

    dependency_depths = [len(list(token.children)) for token in doc]
    avg_depth = sum(dependency_depths) / len(dependency_depths) if dependency_depths else 0

    unique_words = set([token.text.lower() for token in doc if not token.is_punct])
    ttr = len(unique_words) / total_tokens

    stop_ratio = len([t for t in doc if t.is_stop]) / total_tokens
    punc_dist = Counter([t.text for t in doc if t.is_punct])

    return {
        "avg_sentence_length" : round(avg_sent_len, 2),
        "pos_ratios" : pos_ratios,
        "avg_depth" : avg_depth,
        "vocabulary_diversity" : round(ttr, 3),
        "stop_ratio" : stop_ratio,
        "punct_style" : punc_dist
    }

def translate_to_style_guide(fingerprint):
    style_rules = []
    
    length = fingerprint.get("avg_sentence_length", 10)
    if length < 8: style_rules.append("Write in very short, clipped bursts. Avoid compound sentences.")
    elif length > 20: style_rules.append("Use long, flowing sentences with multiple ideas connected by commas.")
    else: style_rules.append("Maintain a moderate sentence length, typical of standard conversation.")

    depth = fingerprint.get("avg_depth", 1.2)
    if depth > 1.8: style_rules.append("Your thinking is complex. Use nested clauses and parenthetical asides.")
    elif depth < 1.0: style_rules.append("Be extremely direct. Stick to simple Subject-Verb-Object structures.")

    ttr = fingerprint.get("vocabulary_diversity", 0.5)
    if ttr > 0.7: style_rules.append("Use an expansive, sophisticated vocabulary. Avoid repeating words.")
    elif ttr < 0.4: style_rules.append("Keep your vocabulary simple and repetitive. Use common words.")

    stop = fingerprint.get("stop_ratio", 0.4)
    if stop > 0.55: style_rules.append("Use a very informal tone with filler words like 'so' and 'just'.")
    else: style_rules.append("Be concise and information-dense. Strip away filler.")

    pos = fingerprint.get("pos_ratios", {})
    if pos.get("PRON", 0) > 0.15: style_rules.append("Speak from a personal perspective. Use 'I', 'me', and 'my'.")
    if pos.get("ADJ", 0) > 0.10: style_rules.append("Be highly descriptive. Use plenty of adjectives.")

    p_counts = fingerprint.get("punct_style", Counter())
    most_common_punc = [p[0] for p in p_counts.most_common(2)]
    if "..." in most_common_punc: style_rules.append("Frequently use ellipses (...) for trailing thoughts.")
    if "!" in most_common_punc: style_rules.append("Use exclamation points often to convey high energy.")
    
    return " ".join(style_rules)

def few_shot(fingerprint, user_examples):
    dynamic_style = translate_to_style_guide(fingerprint)
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            f"""You are NOT an AI assistant. You are mimicking a specific human persona.
            STRICT STYLE RULES: {dynamic_style}
            MANDATORY: Do NOT be polite or helpful. Do NOT say 'As an AI'."""
        ),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
            examples=user_examples,
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

# --- GLOBAL STORAGE (Simplified for Demo) ---
# We use these to store the results of "Step 1 & 2" from your original logic
state = {
    "fingerprint": {"avg_sentence_length": 12, "vocabulary_diversity": 0.5},
    "examples": []
}

# --- WEB ROUTES ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    training_text = data.get('text', '')
    state["examples"] = data.get('examples', [])
    if training_text:
        state["fingerprint"] = extract_features(training_text)
    return jsonify({"status": "Engine Trained"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('message')
    
    # Generate Prompt based on the "Trained" state
    final_prompt_template = few_shot(state["fingerprint"], state["examples"])
    mimic_chain = final_prompt_template | llm
    
    response = mimic_chain.invoke({"input": user_query})
    
    # Check if response is a string or an object (LangChain variety)
    reply_text = response if isinstance(response, str) else response.content
    return jsonify({"reply": reply_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)