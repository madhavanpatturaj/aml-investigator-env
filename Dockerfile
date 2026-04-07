FROM python:3.10-slim

# Set up a working directory
WORKDIR /app

# Copy your files into the container
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir .

# Expose port 8000 for the Hugging Face Space
EXPOSE 7860

# Command to run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]