# MOrA User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [CLI Commands Reference](#cli-commands-reference)
3. [Data Collection Workflow](#data-collection-workflow)
4. [Model Training Workflow](#model-training-workflow)
5. [Model Evaluation Workflow](#model-evaluation-workflow)
6. [System Monitoring](#system-monitoring)
7. [Configuration Management](#configuration-management)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Best Practices](#best-practices)

## Quick Start

### Prerequisites Check
```bash
# Verify system setup
./scripts/verify-setup.sh

# Check system resources
./scripts/check_system_resources.sh
```

### Basic Usage Flow
```bash
# 1. Collect training data
python3 -m src.mora.cli.main train collect-data-parallel \
    --services "frontend" \
    --config-file config/resource-optimized.yaml \
    --max-workers 1

# 2. Train models
python3 train_models/train_working_lstm_prophet.py

# 3. Evaluate models
python3 evaluate_models/evaluate_frontend_model.py
```

## CLI Commands Reference

### Main CLI Interface
```bash
python3 -m src.mora.cli.main [COMMAND] [OPTIONS]
```

#### Available Commands
- `train`: Data collection and model training
- `status`: Check system and training status
- `clean`: Clean up experiment data
- `--help`: Show help information

### Data Collection Commands

#### 1. Parallel Experiments (Recommended)
```bash
# Collect data for multiple services
python3 -m src.mora.cli.main train collect-data-parallel \
    --services "frontend,cartservice,checkoutservice" \
    --config-file config/resource-optimized.yaml \
    --max-workers 1

# Collect data for all remaining services
python3 -m src.mora.cli.main train collect-data-parallel \
    --services "adservice,currencyservice,emailservice,paymentservice,productcatalogservice,recommendationservice,shippingservice" \
    --config-file config/resource-optimized.yaml \
    --max-workers 1
```

#### 2. Single Service Collection
```bash
# Collect data for a specific service
python3 -m src.mora.cli.main train collect-data \
    --service "frontend" \
    --config-file config/resource-optimized.yaml
```

#### 3. Status Check
```bash
# Check training progress
python3 -m src.mora.cli.main status

# Check specific service status
python3 -m src.mora.cli.main status --service frontend
```

### Model Training Commands

#### 1. LSTM + Prophet Pipeline (Primary)
```bash
# Train for frontend service
python3 train_models/train_working_lstm_prophet.py

# Train for cartservice
python3 train_models/train_cartservice_lstm_prophet.py

# Train for checkoutservice
python3 train_models/train_checkoutservice_lstm_prophet.py
```

#### 2. RandomForest Alternative
```bash
# Train RandomForest models
python3 train_models/train_from_collected_data.py
```

### Model Evaluation Commands

#### 1. Individual Service Evaluation
```bash
# Evaluate frontend model
python3 evaluate_models/evaluate_frontend_model.py

# Evaluate cartservice model
python3 evaluate_models/evaluate_cartservice_model.py

# Evaluate checkoutservice model
python3 evaluate_models/evaluate_checkoutservice_model.py
```

#### 2. Comprehensive Testing
```bash
# Run comprehensive model tests
python3 evaluate_models/test_model_performance.py

# Validate model functionality
python3 evaluate_models/validate_model.py
```

### System Management Commands

#### 1. Cleanup Operations
```bash
# Clean experiment data
python3 -m src.mora.cli.main clean experiments

# Clean specific service data
python3 -m src.mora.cli.main clean experiments --service frontend
```

#### 2. Repository Cleanup
```bash
# Clean repository (removes cache, temp files)
./scripts/cleanup_repository.sh
```

## Data Collection Workflow

### Step-by-Step Data Collection

#### 1. Pre-Collection Setup
```bash
# Verify system health
./scripts/verify-setup.sh

# Check available resources
./scripts/check_system_resources.sh

# Ensure Prometheus is running
curl http://localhost:9090/-/ready
```

#### 2. Start Data Collection
```bash
# Resource-optimized collection (recommended)
python3 -m src.mora.cli.main train collect-data-parallel \
    --services "frontend,cartservice,checkoutservice" \
    --config-file config/resource-optimized.yaml \
    --max-workers 1
```

#### 3. Monitor Progress
```bash
# Check collection status
./scripts/monitor_data_collection.sh

# View real-time progress
tail -f data_collection.log

# Check collected files
ls training_data/*.csv | wc -l
```

#### 4. Verify Data Quality
```bash
# Check data completeness
python3 -c "
import pandas as pd
import os
files = [f for f in os.listdir('training_data') if f.endswith('.csv')]
print(f'Total CSV files: {len(files)}')
if files:
    df = pd.read_csv(f'training_data/{files[0]}')
    print(f'Sample file columns: {len(df.columns)}')
    print(f'Sample file rows: {len(df)}')
"
```

### Configuration Options

#### Resource-Optimized Configuration
```yaml
# config/resource-optimized.yaml
training:
  steady_state_config:
    experiment_duration_minutes: 15
    replica_counts: [1, 2, 4]
    load_levels_users: [5, 10, 20, 30, 50, 75]
    test_scenarios: ['browsing', 'checkout']

max_parallel_workers: 1
cpu_limit_per_worker: 2
memory_limit_per_worker: 2
```

#### Default Configuration
```yaml
# config/default.yaml
training:
  steady_state_config:
    experiment_duration_minutes: 45
    replica_counts: [1, 2, 4, 6]
    load_levels_users: [10, 50, 100, 150, 200, 250]
    test_scenarios: ['browsing', 'checkout']
```

## Model Training Workflow

### LSTM + Prophet Pipeline Training

#### 1. Prepare Training Data
```bash
# Ensure data is collected
ls training_data/*.csv | wc -l

# Check data quality
python3 -c "
import pandas as pd
df = pd.read_csv('training_data/frontend_browsing_replicas_1_users_5.csv')
print(f'Data shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
"
```

#### 2. Train Models
```bash
# Train LSTM + Prophet pipeline
python3 train_models/train_working_lstm_prophet.py
```

#### 3. Verify Training Results
```bash
# Check trained models
ls models/*.joblib

# Verify model metadata
python3 -c "
import joblib
model = joblib.load('models/frontend_lstm_prophet_pipeline.joblib')
print(f'Model keys: {list(model.keys())}')
print(f'Trained at: {model.get(\"trained_at\", \"Unknown\")}')
"
```

### Service-Specific Training

#### Frontend Service
```bash
python3 train_models/train_working_lstm_prophet.py
```

#### Cartservice
```bash
python3 train_models/train_cartservice_lstm_prophet.py
```

#### Checkoutservice
```bash
python3 train_models/train_checkoutservice_lstm_prophet.py
```

## Model Evaluation Workflow

### Individual Model Evaluation

#### 1. Evaluate Frontend Model
```bash
python3 evaluate_models/evaluate_frontend_model.py
```

#### 2. Evaluate Cartservice Model
```bash
python3 evaluate_models/evaluate_cartservice_model.py
```

#### 3. Evaluate Checkoutservice Model
```bash
python3 evaluate_models/evaluate_checkoutservice_model.py
```

### Comprehensive Evaluation

#### 1. Performance Testing
```bash
python3 evaluate_models/test_model_performance.py
```

#### 2. Model Validation
```bash
python3 evaluate_models/validate_model.py
```

### Evaluation Output Examples

#### Sample Evaluation Report
```
🎯 MODEL EVALUATION RESULTS FOR FRONTEND
==========================================

📊 LSTM Performance:
  CPU Model MSE: 0.001239 (Excellent)
  Memory Model MSE: 195,871,846,729,430 (High variance)
  Replica Model MSE: 0.430723 (Good)

📈 Prophet Performance:
  Trend Detection: Excellent
  Seasonality: Weekly patterns detected
  Uncertainty Intervals: 90% confidence bounds

🔮 Fusion Results:
  CPU Prediction: 0.069870 (Confidence: 0.9000)
  Memory Prediction: 5,569,583 bytes (Confidence: 0.5000)
  Replica Prediction: 2.006691 (Confidence: 0.9000)

✅ Overall Status: Production Ready
```

## System Monitoring

### Real-Time Monitoring

#### 1. Data Collection Monitoring
```bash
# Monitor data collection progress
./scripts/monitor_data_collection.sh

# Check process status
ps aux | grep collect-data-parallel

# View system resources
./scripts/check_system_resources.sh
```

#### 2. System Health Checks
```bash
# Comprehensive system check
./scripts/verify-setup.sh

# Check Kubernetes status
kubectl get pods -n hipster-shop

# Check Prometheus status
curl http://localhost:9090/-/ready
```

### Monitoring Scripts

#### 1. Data Collection Monitor
```bash
#!/bin/bash
# scripts/monitor_data_collection.sh
echo "🔍 MOrA DATA COLLECTION MONITOR"
echo "Timestamp: $(date)"

# Check process status
if pgrep -f "collect-data-parallel" > /dev/null; then
    echo "✅ Data collection process is RUNNING"
    echo "   PID: $(pgrep -f 'collect-data-parallel')"
else
    echo "❌ Data collection process is NOT RUNNING"
fi

# Check data collection results
if [ -d "training_data" ]; then
    csv_count=$(find training_data -name "*.csv" 2>/dev/null | wc -l)
    echo "📊 CSV files collected: $csv_count"
fi
```

#### 2. System Resource Monitor
```bash
#!/bin/bash
# scripts/check_system_resources.sh
echo "💻 SYSTEM RESOURCE CHECK"
echo "CPU: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free | grep Mem | awk '{printf \"%.1f%%\", $3/$2 * 100.0}')"
echo "Disk: $(df -h . | tail -1 | awk '{print $5}')"
```

## Configuration Management

### Configuration Files

#### 1. Resource-Optimized Configuration
```yaml
# config/resource-optimized.yaml
training:
  steady_state_config:
    experiment_duration_minutes: 15
    replica_counts: [1, 2, 4]
    load_levels_users: [5, 10, 20, 30, 50, 75]
    test_scenarios: ['browsing', 'checkout']

max_parallel_workers: 1
cpu_limit_per_worker: 2
memory_limit_per_worker: 2

jmeter_thread_ramp_up: 30
jmeter_thread_ramp_down: 30
jmeter_think_time: 2

metrics_collection_interval: 30
prometheus_query_timeout: 10
max_retries: 3
```

#### 2. Fast Training Configuration
```yaml
# config/fast-training.yaml
training:
  steady_state_config:
    experiment_duration_minutes: 5
    replica_counts: [1, 2]
    load_levels_users: [5, 10]
    test_scenarios: ['browsing']
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

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Data Collection Issues

**Problem**: No data returned for queries
```bash
# Check Prometheus connectivity
curl http://localhost:9090/-/ready

# Verify service labels
kubectl get pods -n hipster-shop --show-labels

# Test Prometheus queries manually
curl "http://localhost:9090/api/v1/query?query=container_cpu_usage_seconds_total"
```

**Solution**: Ensure Prometheus is running and services have correct labels

#### 2. Model Training Issues

**Problem**: Training fails with memory errors
```bash
# Check available memory
free -h

# Reduce batch size in training script
# Edit train_models/train_working_lstm_prophet.py
# Change: batch_size=32 to batch_size=16
```

**Solution**: Reduce batch size or increase system memory

#### 3. System Resource Issues

**Problem**: System becomes unresponsive during data collection
```bash
# Check system resources
./scripts/check_system_resources.sh

# Use resource-optimized configuration
python3 -m src.mora.cli.main train collect-data-parallel \
    --config-file config/resource-optimized.yaml \
    --max-workers 1
```

**Solution**: Use resource-optimized configuration with single worker

#### 4. Metrics Collection Issues

**Problem**: High variability warnings
```bash
# Check data quality
python3 -c "
import pandas as pd
df = pd.read_csv('training_data/frontend_browsing_replicas_1_users_5.csv')
print(f'CPU variability: {df[\"cpu_cores_value\"].std() / df[\"cpu_cores_value\"].mean() * 100:.1f}%')
"
```

**Solution**: This is normal for some services; the system handles it gracefully

### Debug Commands

#### 1. System Debugging
```bash
# Check all processes
ps aux | grep -E "(python|jmeter|kubectl)"

# Check system logs
journalctl -f

# Check Kubernetes logs
kubectl logs -n hipster-shop -l app=frontend
```

#### 2. Data Debugging
```bash
# Check collected data
ls -la training_data/

# Validate CSV files
python3 -c "
import pandas as pd
import os
for f in os.listdir('training_data'):
    if f.endswith('.csv'):
        df = pd.read_csv(f'training_data/{f}')
        print(f'{f}: {df.shape[0]} rows, {df.shape[1]} columns')
"
```

#### 3. Model Debugging
```bash
# Check model files
ls -la models/

# Validate model loading
python3 -c "
import joblib
try:
    model = joblib.load('models/frontend_lstm_prophet_pipeline.joblib')
    print('✅ Model loads successfully')
    print(f'Keys: {list(model.keys())}')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
```

## Best Practices

### Data Collection Best Practices

#### 1. Resource Management
- Always use resource-optimized configuration for production
- Monitor system resources during collection
- Use single worker mode to prevent system overload
- Collect data during off-peak hours

#### 2. Data Quality
- Verify data completeness before training
- Check for high variability warnings
- Ensure all 12 metrics are collected
- Validate CSV file structure

#### 3. Collection Strategy
- Start with core services (frontend, cartservice, checkoutservice)
- Collect data for all scenarios (browsing, checkout)
- Use resumable collection to prevent data loss
- Monitor progress regularly

### Model Training Best Practices

#### 1. Training Preparation
- Ensure sufficient training data (minimum 1000 samples)
- Validate data quality before training
- Use appropriate configuration for your system
- Backup existing models before retraining

#### 2. Training Process
- Train models for each service individually
- Use LSTM + Prophet pipeline for best results
- Monitor training progress and resource usage
- Validate model performance after training

#### 3. Model Management
- Save models with metadata and timestamps
- Version control trained models
- Document model performance metrics
- Test model inference capabilities

### System Maintenance Best Practices

#### 1. Regular Maintenance
- Clean up temporary files regularly
- Monitor disk space usage
- Update dependencies periodically
- Backup important data and models

#### 2. Monitoring
- Set up automated monitoring scripts
- Check system health regularly
- Monitor data collection progress
- Track model performance over time

#### 3. Documentation
- Keep documentation updated
- Document any custom configurations
- Record troubleshooting solutions
- Maintain change logs

---

**User Guide Version**: 1.0  
**Last Updated**: October 25, 2024  
**Compatible With**: MOrA v1.0  
**CLI Version**: 2.0  
**Tested Commands**: All commands verified and working
