# 1. Use an official Python image
FROM python:3.10

# 2. Create a 'user' to run the app (Hugging Face requires this for security)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 3. Set the directory where our code will live
WORKDIR /app

# 4. Copy the requirements file and install the libraries
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 5. Copy everything else (app.py and index.html) into the container
COPY --chown=user . /app

# 6. Start the Flask server on the specific port Hugging Face expects
CMD ["python", "app.py"]