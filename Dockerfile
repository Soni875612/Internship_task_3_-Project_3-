# DecodeLabs - Project 3: Tech Stack Recommender
# Multi-purpose image: run either the Streamlit app or the FastAPI service.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

# Default: launch the Streamlit web app.
# Override at runtime for the API instead:
#   docker run -p 8000:8000 <image> uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
