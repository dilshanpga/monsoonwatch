FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
EXPOSE 8000
CMD ["python", "-m", "streamlit", "run", "app/app.py", "--server.port", "8000", "--server.address", "0.0.0.0", "--server.headless", "true"]
