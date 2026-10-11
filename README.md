# Task-06: Real-Time ML Inference REST API

## Project Overview
This project deploys a Titanic machine learning model through a FastAPI REST API.

## Features
- Real-time predictions using a trained ML model
- `POST /predict` endpoint for predictions
- `GET /health` endpoint to check API status
- Interactive Swagger documentation
- Docker containerization
- Automated API tests using pytest

## Project Structure
- `app/` - API application code
- `model/` - Saved machine learning model
- `tests/` - Automated API tests
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image configuration

## Run with Docker

Build the image:

```bash
sudo docker build -t task06-ml-api .
```

Start the container:

```bash
sudo docker run -d --name task06-container -p 8000:8000 task06-ml-api
```

## API Endpoints
- Health check: `GET /health`
- Prediction: `POST /predict`
- Swagger documentation: `http://localhost:8000/docs`

## Run Tests
```bash
python -m pytest
```

## Test Result
The four automated tests passed in the Docker container.
