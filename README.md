# ┌───────────────────────────────┐
# │  🎭   P E R S O N A _ M I M I C   
# └───────────────────────────────┘

> A machine learning model that mimics distinct personas by extracting and replicating their unique linguistic fingerprints.

---

## 🤔 Why This Exists

I noticed how differently and uniquely people speak — the rhythm, the quirks, the sentence structures, and the specific words they reach for. Everyone has a distinct **linguistic fingerprint**.

This got me wondering: *Can we recreate that exact style using Large Language Models (LLMs)?*

To find out, I built this project and deployed it live.

👉 **[Try the Live Demo on Hugging Face Spaces →](https://huggingface.co/spaces/noonine/persona-mimic)**

---

## ⚙️ What It Does 

Feed the model someone's writing or data, and it will accurately mimic their style. 

To make testing easier, I have included several **prebuilt personas** that you can try immediately without uploading any data:

| Persona | Description |
| :--- | :--- |
| **🖤 Wednesday Addams** | Deadpan, morbid, and precise |
| **🏴‍☠️ Jack Sparrow** | Rambling, theatrical, and rum-obsessed |
| **👽 Yoda** | Inverted syntax and cryptic wisdom |
| **🌸 Aerith** | Gentle, hopeful, and flower-metaphor-heavy |
| **🤖 GLaDOS** | Passive-aggressive, clinical, and promise-breaking |


---

## 🚀 How to Use It

### Option 1: Pick a Prebuilt Persona
1. Select one of the prebuilt characters from the interface.
2. Start chatting instantly!

### Option 2: Upload Your Own Data & Prompting
Want to clone a specific voice? Follow these steps:

1. **Enter Custom Mode:** Click on the **"Custom Persona"** tab.
2. **Upload/Paste Your Data:** 
   * Paste text samples into the data input field. 
   * This could be chat history, past emails, tweets, or blog posts. 
   * *Tip:* **More data = better mimicry.** Aim for text that really highlights their unique slang, punctuation habits, and formatting quirks.
3. **Give it a Prompt:** In the user message box, type your prompt or question. 
   * *Example:* *"Write an email apologizing for being late"* or *"What do you think about AI?"*
4. **Hit Generate:** The model will analyze the input, extract the underlying linguistic patterns, and reply to your prompt exactly like them.

---

## 🖥️ Run Locally

You can run this project locally using Docker. Execute the following commands in your terminal:

```bash
# Clone the repository
git clone https://huggingface.co/spaces/noonine/persona-mimic
cd persona-mimic

# Build the Docker image
docker build -t persona-mimic .

# Run the container
docker run -p 7860:7860 persona-mimic
```

Once running, open your browser and navigate to: **`http://localhost:7860`**
---

## 🗺️ Roadmap & What's Next

- [ ] **Memory Integration** – Allowing conversations to flow naturally across multiple turns instead of single-shot prompts.
- [ ] **Expanded Roster** – Adding more prebuilt personas (accepting suggestions!).
- [ ] **Privacy-First Uploads** – Securing custom user data so you can upload personal text safely (The big priority 🛡️).
- [ ] **Prompt Engineering & Optimization** – Fine-tuning generation quality and reducing style drift during longer responses.
- [ ] **Voice Synthesis (Future Goal)** – Actually hearing the linguistic fingerprint via text-to-speech models match the persona.

---

## 💬 Contributing & Feedback

This project is in its early stages. If you want to contribute code, design new personas, or share optimization ideas, please feel free to open a **Pull Request** or create an **Issue**. 

If something breaks, or if you just want to share your thoughts, open an issue or ping me directly on Hugging Face!
