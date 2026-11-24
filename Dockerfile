FROM python:3.11-slim

WORKDIR /app

COPY requirements-research.txt .
RUN pip install --no-cache-dir -r requirements-research.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8081

CMD ["python", "-m", "src.terminal_server"]
