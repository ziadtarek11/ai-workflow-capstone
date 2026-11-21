FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# use real data if it was copied in, otherwise generate synthetic data; then bake models + figures
RUN ( [ -n "$(ls data/cs-train/*.json 2>/dev/null)" ] || python generate_synthetic_data.py ) \
    && python eda.py && python train.py && python run-tests.py
EXPOSE 8080
CMD ["python", "app.py"]
