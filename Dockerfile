FROM python:3.10

# Create user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install requirements
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# --- THE FIX IS HERE ---
# This forces the download of the model during the build
RUN python -m spacy download en_core_web_sm

COPY --chown=user . /app

CMD ["python", "app.py"]