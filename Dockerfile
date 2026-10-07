FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    numpy pandas shap streamlit scipy scikit-learn flwr

COPY model.py utils.py dashboard.py ./
COPY data/global_test.csv data/global_test.csv
COPY federated_model.pt .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]