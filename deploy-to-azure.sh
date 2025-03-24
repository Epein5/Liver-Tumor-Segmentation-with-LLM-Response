#!/bin/bash
set -e  # Exit on any error

# Variables - customize these
RESOURCE_GROUP="liver-tumor-rg"
LOCATION="eastus"
ACR_NAME="livertumorregistry$(date +%s | cut -c 6-10)"  # Make it unique with timestamp
APP_NAME="liver-tumor-app$(date +%s | cut -c 6-10)"     # Make it unique with timestamp

# Login to Azure
echo "Logging in to Azure..."
az account show > /dev/null || az login

# Create Resource Group
echo "Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
echo "Creating Azure Container Registry..."
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic
az acr update --name $ACR_NAME --admin-enabled true

# Build and push Docker image to ACR
echo "Building and pushing Docker image to ACR..."
az acr login --name $ACR_NAME
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)

# Ensure the image is built with latest fixes
docker-compose build

# Tag and push the image
docker tag liver-tumor-segmentation-with-llm-response_liver-tumor-app:latest $ACR_LOGIN_SERVER/liver-tumor-app:latest
docker push $ACR_LOGIN_SERVER/liver-tumor-app:latest

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

# Use Container Apps instead of App Service for better container support
echo "Setting up Azure Container Apps..."
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App

# Create Container Apps environment
az containerapp env create \
  --name liver-tumor-env \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Create storage for persistence
# echo "Setting up persistent storage..."
# STORAGE_ACCOUNT_NAME="livertumorstorage$(date +%s | cut -c 6-10)"
# az storage account create --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP \
#   --location $LOCATION --sku Standard_LRS

# az storage share create --name liver-tumor-data --account-name $STORAGE_ACCOUNT_NAME

# Deploy the container app
# echo "Deploying the application..."
# az containerapp create \
#   --name $APP_NAME \
#   --resource-group $RESOURCE_GROUP \
#   --environment liver-tumor-env \
#   --image $ACR_LOGIN_SERVER/liver-tumor-app:latest \
#   --registry-server $ACR_LOGIN_SERVER \
#   --registry-username $ACR_USERNAME \
#   --registry-password $ACR_PASSWORD \
#   --target-port 8000 \
#   --ingress external \
#   --min-replicas 1 \
#   --max-replicas 1

# Create storage for persistence
echo "Setting up persistent storage..."
STORAGE_ACCOUNT_NAME="livertumorstorage$(date +%s | cut -c 6-10)"
az storage account create --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP \
  --location $LOCATION --sku Standard_LRS

# Get storage account key
STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" -o tsv)

# Create file share with explicit credentials
az storage share create --name liver-tumor-data --account-name $STORAGE_ACCOUNT_NAME --account-key "$STORAGE_KEY"

# Deploy the container app with proper resources for SHAP calculations
echo "Deploying the application..."
az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment liver-tumor-env \
  --image $ACR_LOGIN_SERVER/liver-tumor-app:latest \
  --registry-server $ACR_LOGIN_SERVER \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 1 \
  --cpu 1 \
  --memory 2Gi \
  # --request-timeout 300

# Get the application URL
APP_URL=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)

echo "Deployment complete! Your application is available at:"
echo "https://$APP_URL"