#!/bin/bash

# Script to deploy the Fraud Detection System to a cloud environment

# Set variables
PROJECT_NAME="fraud-detection-system"
REGION="us-central1"
IMAGE_NAME="fraud-detection"
IMAGE_TAG="latest"

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ..

# Tag the image for Google Container Registry (example)
echo "Tagging image for GCR..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} gcr.io/${PROJECT_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

# Push the image to the registry
echo "Pushing image to GCR..."
docker push gcr.io/${PROJECT_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

# Deploy to Cloud Run (example)
echo "Deploying to Cloud Run..."
gcloud run deploy ${IMAGE_NAME} \
  --image gcr.io/${PROJECT_NAME}/${IMAGE_NAME}:${IMAGE_TAG} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --port 8000

echo "Deployment completed!"
