# Use an official Python runtime as a parent image
FROM python:3.11.6-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Run app.py when the container launches
CMD streamlit run app.py