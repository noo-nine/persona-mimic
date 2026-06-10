FROM python:3.10-slim

# Create a non-root user for security compliance with Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and change ownership to the 'user'
COPY --chown=user requirements.txt requirements.txt

# Upgrade pip and install all backend dependencies under the user's local directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# --- THE SPAÇY FIX ---
# Downloading the model with the '--user' flag ensures it installs into the user's local path
RUN python -m spacy download en_core_web_sm --user

# Copy the rest of the application files
COPY --chown=user . /app

# Expose the mandatory Hugging Face web port
EXPOSE 7860

# Run the backend script (Ensure this matches your actual filename, e.g., app.py or backend.py)
CMD ["python", "app.py"]