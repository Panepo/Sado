FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY install_dependency.py ./
RUN python install_dependency.py

# Copy application code
COPY server.py ragas_runner.py ./
COPY static/ ./static/

EXPOSE 8040

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8040"]
