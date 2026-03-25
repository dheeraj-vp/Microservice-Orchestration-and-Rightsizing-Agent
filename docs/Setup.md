# MOrA Project Setup Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Installation Guide](#installation-guide)
5. [Environment Setup](#environment-setup)
6. [Configuration](#configuration)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)
9. [Development Setup](#development-setup)

## Project Overview

**MOrA (Microservice Orchestration and Rightsizing Agent)** is an intelligent system designed for automated microservice resource optimization in Kubernetes environments. The project combines machine learning, monitoring, and orchestration to provide real-time resource recommendations.

### Key Features
- **Automated Data Collection**: Collects 12 comprehensive metrics from Kubernetes/Prometheus
- **Machine Learning Pipeline**: LSTM + Prophet ensemble for intelligent predictions
- **Resource Optimization**: CPU, Memory, and Replica scaling recommendations
- **Production Ready**: Industry-standard architecture with robust error handling
- **CLI Interface**: Easy-to-use command-line tools for all operations

## System Architecture

### Component Overview

#### 1. Data Collection Layer
- **Prometheus Client**: Fetches metrics from Prometheus
- **Kubernetes Client**: Manages pod scaling and deployments
- **Load Generator**: JMeter-based load testing
- **Data Pipeline**: Orchestrates data collection experiments

#### 2. Machine Learning Layer
- **Feature Engineering**: 12-metric system with intelligent substitutes
- **Prophet Models**: Time series forecasting for trend analysis
- **LSTM Models**: Deep learning for pattern recognition
- **Fusion Engine**: Combines predictions with confidence scoring

#### 3. Application Layer
- **CLI Interface**: Command-line tools for all operations
- **Model Library**: Manages trained models
- **Recommendation Engine**: Generates actionable suggestions

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: Minimum 4 cores (8 cores recommended)
- **Storage**: Minimum 20GB free space
- **Network**: Internet connection for package downloads

### Software Dependencies
- **Python**: 3.8 or higher
- **Docker**: Latest version
- **Minikube**: Latest version
- **kubectl**: Latest version
- **JMeter**: 5.4.1 or higher

### Required Services
- **Kubernetes Cluster**: Minikube (local development)
- **Prometheus**: For metrics collection
- **Hipster Shop**: Sample microservices application

## Installation Guide

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd MOrA
```

### Step 2: Install System Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install JMeter
wget https://archive.apache.org/dist/jmeter/binaries/apache-jmeter-5.4.1.tgz
tar -xzf apache-jmeter-5.4.1.tgz
sudo mv apache-jmeter-5.4.1 /opt/jmeter
sudo ln -s /opt/jmeter/bin/jmeter /usr/local/bin/jmeter
```

### Step 3: Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 4: Setup Kubernetes Environment
```bash
# Start Minikube
minikube start --memory=8192 --cpus=4

# Enable required addons
minikube addons enable metrics-server
minikube addons enable ingress
```

## Environment Setup

### Automated Setup Script
The project includes an automated setup script that handles the complete environment configuration:

```bash
# Make setup script executable
chmod +x scripts/setup-minikube.sh

# Run automated setup
./scripts/setup-minikube.sh
```

### Manual Setup Steps

#### 1. Deploy Prometheus
```bash
# Create Prometheus namespace
kubectl create namespace monitoring

# Deploy Prometheus
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Wait for Prometheus to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus-operator -n monitoring --timeout=300s
```

#### 2. Deploy Hipster Shop
```bash
# Clone Hipster Shop repository
git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
cd microservices-demo

# Deploy to Kubernetes
kubectl apply -f ./release/kubernetes-manifests.yaml

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod --all -n default --timeout=300s
```

#### 3. Configure Service Discovery
```bash
# Create ServiceMonitor for Prometheus
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: hipster-shop-monitor
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: frontend
  endpoints:
  - port: http
    interval: 30s
EOF
```

## Configuration

### Configuration Files

#### 1. Resource-Optimized Configuration (`config/resource-optimized.yaml`)
```yaml
training:
  steady_state_config:
    experiment_duration_minutes: 15
    replica_counts: [1, 2, 4]
    load_levels_users: [5, 10, 20, 30, 50, 75]
    test_scenarios: ['browsing', 'checkout']

# Resource Limits
max_parallel_workers: 1
cpu_limit_per_worker: 2
memory_limit_per_worker: 2

# Load Generation Optimization
jmeter_thread_ramp_up: 30
jmeter_thread_ramp_down: 30
jmeter_think_time: 2

# Metrics Collection Optimization
metrics_collection_interval: 30
prometheus_query_timeout: 10
max_retries: 3
```

#### 2. Default Configuration (`config/default.yaml`)
```yaml
training:
  steady_state_config:
    experiment_duration_minutes: 45
    replica_counts: [1, 2, 4, 6]
    load_levels_users: [10, 50, 100, 150, 200, 250]
    test_scenarios: ['browsing', 'checkout']
```

### Environment Variables
```bash
# Set environment variables
export PROMETHEUS_URL="http://localhost:9090"
export KUBECONFIG="~/.kube/config"
export NAMESPACE="hipster-shop"
export DATA_DIR="training_data"
export MODEL_DIR="models"
```

## Verification

### System Health Check
```bash
# Run comprehensive verification
python3 tests/run_e2e_validation.py
```

### Manual Verification Steps

#### 1. Check Kubernetes Status
```bash
# Check Minikube status
minikube status

# Check pod status
kubectl get pods -n hipster-shop

# Check services
kubectl get services -n hipster-shop
```

#### 2. Check Prometheus Status
```bash
# Check Prometheus connectivity
curl http://localhost:9090/-/ready

# Check metrics availability
curl "http://localhost:9090/api/v1/query?query=up"
```

#### 3. Test MOrA Components
```bash
# Test CLI interface
python3 -m src.mora.cli.main --help

# Test data collection
python3 -m src.mora.cli.main train collect-data-parallel --services frontend --config-file config/resource-optimized.yaml --max-workers 1

# Test model training
python3 train_models/train_working_lstm_prophet.py
```

## Troubleshooting

### Common Issues

#### 1. Minikube Not Starting
```bash
# Check system resources
free -h
df -h

# Restart Minikube with more resources
minikube stop
minikube start --memory=8192 --cpus=4
```

#### 2. Prometheus Connection Issues
```bash
# Check Prometheus pod status
kubectl get pods -n monitoring

# Check Prometheus logs
kubectl logs -n monitoring -l app.kubernetes.io/name=prometheus

# Port forward for testing
kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
```

#### 3. Metrics Collection Failures
```bash
# Check ServiceMonitor configuration
kubectl get servicemonitor -n monitoring

# Verify pod labels
kubectl get pods -n hipster-shop --show-labels

# Test Prometheus queries manually
curl "http://localhost:9090/api/v1/query?query=container_cpu_usage_seconds_total"
```

#### 4. JMeter Issues
```bash
# Check JMeter installation
jmeter --version

# Test JMeter with simple script
jmeter -n -t scripts/test.jmx -l results.jtl
```

### Debug Commands
```bash
# Check system resources
./scripts/check_system_resources.sh

# Monitor data collection
./scripts/monitor_data_collection.sh

# View logs
tail -f data_collection.log
```

## Development Setup

### Development Environment
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python3 -m pytest tests/

# Run linting
flake8 src/
black src/
```

### Code Structure
- **src/mora/core/**: Core business logic
- **src/mora/cli/**: Command-line interface
- **src/mora/monitoring/**: Prometheus integration
- **src/mora/k8s/**: Kubernetes operations
- **src/mora/models/**: ML model implementations

### Testing
```bash
# Run unit tests
python3 -m pytest tests/test_*.py

# Run integration tests
python3 -m pytest tests/test_integration.py

# Run end-to-end tests
python3 tests/run_e2e_validation.py
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the test suite
5. Submit a pull request


