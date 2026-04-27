# Image Classification ML Service

A FastAPI-based machine learning service for image classification on Tiny ImageNet.

The project combines a web interface, REST API, PostgreSQL storage, RabbitMQ task queue, and a background worker that runs inference with a custom ResNet-like neural network trained on Tiny ImageNet.

---

## Overview

This service allows users to:

* register and log in;
* manage an internal balance;
* upload images for classification;
* create paid ML prediction tasks;
* process prediction tasks asynchronously through RabbitMQ;
* store task history and transaction history;
* view prediction results through the web interface.

The ML model is a custom ResNet-style convolutional neural network trained on the Tiny ImageNet dataset with 200 classes and 64×64 input images.

---

## Tech Stack

* **Backend:** FastAPI
* **Web UI:** Jinja2 templates, HTML, CSS
* **Database:** PostgreSQL
* **ORM:** SQLAlchemy
* **Task Queue:** RabbitMQ
* **Worker:** Python background worker with Pika
* **ML:** PyTorch, torchvision, Pillow
* **Reverse Proxy:** Nginx
* **Containerization:** Docker, Docker Compose

---

## Architecture

```text
User / Browser / API Client
        |
        v
Nginx Reverse Proxy
        |
        v
FastAPI App
   |        |
   |        ├── Web UI
   |        └── REST API
   |
   ├── PostgreSQL
   |     ├── users
   |     ├── ml_models
   |     ├── ml_tasks
   |     └── transactions
   |
   └── RabbitMQ
          |
          v
       Worker
          |
          v
   PyTorch Tiny ImageNet ResNet Inference
```

The FastAPI application does not run heavy ML inference inside HTTP requests. Instead, it saves the uploaded image, creates a task in the database, sends the task to RabbitMQ, and immediately returns a response. The worker then picks up the task, runs the model, and writes the result back to the database.

---

## Project Structure

```text
image-classification-ml-service/
├── README.md
├── docker-compose.yml
├── .env.example
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/
│   │   └── tiny_imagenet_resnet/
│   │       ├── .gitkeep
│   │       └── checkpoint.pth          # local model artifact, not committed
│   ├── uploads/
│   │   └── .gitkeep                    # uploaded images, not committed
│   └── src/
│       ├── main.py
│       ├── db.py
│       ├── models.py
│       ├── schemas.py
│       ├── services.py
│       ├── storage.py
│       ├── worker.py
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── tiny_imagenet_resnet.py
│       │   └── inference.py
│       ├── routers/
│       │   ├── auth.py
│       │   ├── balance.py
│       │   ├── history.py
│       │   ├── predict.py
│       │   ├── users.py
│       │   └── web.py
│       ├── templates/
│       │   ├── base.html
│       │   ├── dashboard.html
│       │   ├── history.html
│       │   ├── index.html
│       │   ├── login.html
│       │   ├── predict.html
│       │   └── register.html
│       └── static/
│           └── style.css
└── web-proxy/
    ├── Dockerfile
    └── nginx.conf
```

---

## Features

### Authentication

* User registration
* User login through web sessions
* HTTP Basic authentication for API endpoints

### Balance System

* User balance storage
* Balance deposit
* Paid prediction requests
* Transaction history

### Image Prediction Pipeline

* Image upload through the web interface
* Saving uploaded images to `app/uploads/`
* Creating ML tasks in PostgreSQL
* Sending task IDs to RabbitMQ
* Processing tasks in a background worker
* Running PyTorch inference
* Saving top-k prediction results in the database

### Web Interface

* Home page
* Registration page
* Login page
* Dashboard
* Prediction page
* Task status page
* Task and transaction history page

---

## API Routes

### Health Check

```http
GET /health
```

Returns service status.

---

### Auth

```http
POST /auth/register
```

Registers a new user.

---

### User

```http
GET /users/me
```

Returns the current authenticated user.

Requires HTTP Basic authentication.

---

### Balance

```http
GET /balance/
POST /balance/deposit
```

Allows a user to check and deposit balance.

Requires HTTP Basic authentication.

---

### Prediction

```http
POST /predict
```

Creates a prediction task through the API.

Requires HTTP Basic authentication.

---

### History

```http
GET /history/tasks
GET /history/tasks/{task_id}
GET /history/transactions
```

Returns task and transaction history for the authenticated user.

Requires HTTP Basic authentication.

---

## Web Routes

```text
GET  /                  Home page
GET  /register          Registration form
POST /register          Register user
GET  /login             Login form
POST /login             Login user
GET  /logout            Logout user
GET  /dashboard         User dashboard
POST /dashboard/deposit Deposit balance
GET  /predict           Prediction form
POST /predict-ui        Submit image prediction task
GET  /predict/{task_id} View prediction task status
GET  /history           View task and transaction history
```

---

## ML Model

The current model is a custom ResNet-like CNN trained on Tiny ImageNet.

Model structure:

```text
ImageNetNN
├── conv1: 3 → 32
├── block2: 32 → 64
├── block3: 64 → 128
├── block4: 128 → 256
├── block5: 256 → 512
├── adaptive average pooling
└── classifier: 512 → 256 → 200
```

The model outputs logits for 200 Tiny ImageNet classes. During inference, the service applies softmax and returns the top-k predictions.

Example result:

```json
{
  "top1": {
    "class_index": 28,
    "class_name": "28",
    "probability": 0.989805
  },
  "top_k": [
    {
      "class_index": 28,
      "class_name": "28",
      "probability": 0.989805
    },
    {
      "class_index": 29,
      "class_name": "29",
      "probability": 0.005344
    }
  ]
}
```

At the moment, class names may be returned as numeric class indices if the class mapping file is not provided. A future improvement is to add Tiny ImageNet class mapping, for example `idx_to_class.json` and `wnid_to_words.json`.

---

## Model Artifact

The PyTorch checkpoint is expected at:

```text
app/models/tiny_imagenet_resnet/checkpoint.pth
```

The checkpoint file is not committed to GitHub. It should be added locally before running inference.

Expected checkpoint format:

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "best_acc": 0.5498,
    "epoch": 182,
}
```

---

## Environment Variables

Create a `.env` file or use Docker Compose environment variables.

Typical variables:

```env
POSTGRES_DB=ml_service
POSTGRES_USER=ml_user
POSTGRES_PASSWORD=ml_password

RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=ml_user
RABBITMQ_PASSWORD=ml_password
RABBITMQ_QUEUE=ml_tasks
```

---

## Running the Project

The recommended way to run the project locally is with Docker Compose.

### Prerequisites

Make sure you have installed:

* Docker
* Docker Compose
* Git

You also need the model checkpoint file locally:

```text
checkpoint.pth
```

The checkpoint is not committed to GitHub because it is a large model artifact.

---

### 1. Clone the repository

```bash
git clone https://github.com/Nik-Esakoff/image-classification-ml-service.git
cd image-classification-ml-service
```

---

### 2. Create environment file

Copy the example environment file:

```bash
cp .env.example .env
```

Check that the database and RabbitMQ variables match the values used in `docker-compose.yml`.

Typical values:

```env
POSTGRES_DB=ml_service
POSTGRES_USER=ml_user
POSTGRES_PASSWORD=ml_password

RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=ml_user
RABBITMQ_PASSWORD=ml_password
RABBITMQ_QUEUE=ml_tasks
```

---

### 3. Add the model checkpoint

Create the model directory if it does not exist:

```bash
mkdir -p app/models/tiny_imagenet_resnet
```

Place the trained checkpoint here:

```text
app/models/tiny_imagenet_resnet/checkpoint.pth
```

The final path must be:

```text
app/models/tiny_imagenet_resnet/checkpoint.pth
```

This file is required for real inference. Without it, prediction tasks will fail with a `Model checkpoint not found` error.

---

### 4. Create upload directory

```bash
mkdir -p app/uploads
```

Uploaded images are stored in this directory during runtime.

---

### 5. Build and start all services

```bash
docker compose up --build
```

This starts:

* FastAPI application
* PostgreSQL database
* RabbitMQ
* background worker
* Nginx reverse proxy

The web app will be available at:

```text
http://localhost
```

---

### 6. Initialize and use the application

After the containers are running:

1. Open `http://localhost`.
2. Register a new user.
3. Log in.
4. Deposit balance on the dashboard.
5. Open the prediction page.
6. Upload an image.
7. Submit the prediction task.
8. Refresh the task status page.
9. View the model result.

---

### 7. Check running containers

```bash
docker compose ps
```

The `app`, `worker`, `database`, `rabbitmq`, and `web-proxy` services should be running.

---

### 8. View logs

All logs:

```bash
docker compose logs -f
```

Worker logs only:

```bash
docker compose logs -f worker
```

Application logs only:

```bash
docker compose logs -f app
```

---

### 9. Stop the project

```bash
docker compose down
```

To stop the project and remove database volumes as well:

```bash
docker compose down -v
```

Use `-v` carefully, because it removes stored PostgreSQL data.

---

### 10. Rebuild after code changes

If you change Python dependencies or Docker configuration, rebuild:

```bash
docker compose up --build
```

If you only change Python code mounted into the container, restarting may be enough:

```bash
docker compose restart app worker
```

---

## Basic Usage Flow

1. Open the web interface.
2. Register a new user.
3. Log in.
4. Deposit balance.
5. Go to the prediction page.
6. Upload an image.
7. Submit the prediction task.
8. Refresh task status.
9. View the prediction result.

---

## Testing the Model Inside Docker

Check that the checkpoint can be loaded:

```bash
docker compose run --rm app sh -lc 'PYTHONPATH=src python - << "PY"
import torch
from ml.tiny_imagenet_resnet import ImageNetNN

checkpoint = torch.load(
    "models/tiny_imagenet_resnet/checkpoint.pth",
    map_location="cpu",
)

print("checkpoint keys:", checkpoint.keys())
print("best_acc:", checkpoint.get("best_acc"))
print("epoch:", checkpoint.get("epoch"))

model = ImageNetNN(num_classes=200)
model.load_state_dict(checkpoint["model"])
model.eval()

print("OK: model loaded successfully")
PY'
```

Test inference on an image:

```bash
cp /path/to/image.jpg app/uploads/test.jpg
```

```bash
docker compose run --rm app sh -lc 'PYTHONPATH=src python - << "PY"
from ml.inference import predict_image

result = predict_image("uploads/test.jpg")
print(result)
PY'
```

---

## Development Notes

### Uploaded Images

Uploaded images are stored in:

```text
app/uploads/
```

This folder is ignored by Git except for `.gitkeep`.

### Model Checkpoints

Model checkpoints are stored locally in:

```text
app/models/tiny_imagenet_resnet/
```

`.pth` files are ignored by Git and should not be committed directly.

### Database Result Field

Prediction results can be longer than 100 characters, so the `MLTask.result` field should use `Text`, not `String(100)`.

---

## Known Limitations

* Passwords are currently handled in a simple way and should be replaced with proper password hashing before production use.
* The model checkpoint is stored locally and is not downloaded automatically.
* Class names are currently numeric unless Tiny ImageNet class mapping is added.
* There are no Alembic migrations yet.
* The UI currently displays prediction results as raw JSON.
* The project is intended as an educational ML service prototype, not a production-ready system.
