# MOrA API Reference

## Table of Contents
1. [CLI API Reference](#cli-api-reference)
2. [Core Module APIs](#core-module-apis)
3. [Monitoring APIs](#monitoring-apis)
4. [ML Pipeline APIs](#ml-pipeline-apis)
5. [Data Acquisition APIs](#data-acquisition-apis)
6. [Model Management APIs](#model-management-apis)
7. [Utility APIs](#utility-apis)
8. [Configuration APIs](#configuration-apis)

## CLI API Reference

### Main CLI Interface

#### Command Structure
```bash
python3 -m src.mora.cli.main [COMMAND] [OPTIONS]
```

#### Available Commands

##### 1. Data Collection Commands

###### `train collect-data-parallel`
```bash
python3 -m src.mora.cli.main train collect-data-parallel [OPTIONS]
```

**Description**: Collect training data for multiple services in parallel

**Options**:
- `--services TEXT`: Comma-separated list of services (required)
- `--config-file TEXT`: Path to configuration file (default: config/resource-optimized.yaml)
- `--max-workers INTEGER`: Maximum number of parallel workers (default: 1)
- `--help`: Show help message

**Examples**:
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

**Returns**:
```json
{
    "status": "completed",
    "total_experiments": 36,
    "successful_experiments": 36,
    "failed_experiments": 0,
    "skipped_experiments": 0,
    "data_quality": {
        "high_quality": 30,
        "medium_quality": 6,
        "low_quality": 0
    }
}
```

###### `train collect-data`
```bash
python3 -m src.mora.cli.main train collect-data [OPTIONS]
```

**Description**: Collect training data for a single service

**Options**:
- `--service TEXT`: Service name (required)
- `--config-file TEXT`: Path to configuration file (default: config/resource-optimized.yaml)
- `--help`: Show help message

**Examples**:
```bash
# Collect data for frontend service
python3 -m src.mora.cli.main train collect-data \
    --service "frontend" \
    --config-file config/resource-optimized.yaml
```

##### 2. Status Commands

###### `status`
```bash
python3 -m src.mora.cli.main status [OPTIONS]
```

**Description**: Check system and training status

**Options**:
- `--service TEXT`: Check status for specific service
- `--help`: Show help message

**Examples**:
```bash
# Check overall status
python3 -m src.mora.cli.main status

# Check specific service status
python3 -m src.mora.cli.main status --service frontend
```

**Returns**:
```json
{
    "system_status": "healthy",
    "services": {
        "frontend": {
            "status": "completed",
            "experiments": 36,
            "models_trained": true
        },
        "cartservice": {
            "status": "completed",
            "experiments": 36,
            "models_trained": true
        }
    },
    "data_collection": {
        "total_csv_files": 108,
        "total_json_files": 108,
        "in_progress": false
    }
}
```

##### 3. Cleanup Commands

###### `clean experiments`
```bash
python3 -m src.mora.cli.main clean experiments [OPTIONS]
```

**Description**: Clean up experiment data

**Options**:
- `--service TEXT`: Clean data for specific service
- `--confirm`: Confirm cleanup operation
- `--help`: Show help message

**Examples**:
```bash
# Clean all experiment data
python3 -m src.mora.cli.main clean experiments --confirm

# Clean specific service data
python3 -m src.mora.cli.main clean experiments --service frontend --confirm
```

##### 4. Evaluation Commands (Phase 4)

###### `evaluate run-experiment`
```bash
python3 -m src.mora.cli.main evaluate run-experiment [OPTIONS]
```

**Description**: Run comparative evaluation experiments (Phase 4)

**Options**:
- `--service TEXT`: Service name to evaluate (required)
- `--strategies TEXT`: Comma-separated list of strategies (default: "statistical,predictive")
- `--load-levels TEXT`: Comma-separated list of load levels (default: "20")
- `--replicas TEXT`: Comma-separated list of replica counts (default: "1,2,4")
- `--scenarios TEXT`: Comma-separated list of scenarios (default: "browsing")
- `--output-dir TEXT`: Output directory for results (default: "evaluation_results")
- `--help`: Show help message

**Examples**:
```bash
# Run POC evaluation (single service, single load level)
python3 -m src.mora.cli.main evaluate run-experiment \
    --service checkoutservice \
    --strategies "statistical,predictive" \
    --load-levels "20" \
    --replicas "1,2,4" \
    --scenarios "browsing" \
    --output-dir evaluation_results

# Run full evaluation (all strategies, all load levels)
python3 -m src.mora.cli.main evaluate run-experiment \
    --service checkoutservice \
    --strategies "statistical,predictive,hpa" \
    --load-levels "10,20,50,100,150,200" \
    --replicas "1,2,4,6" \
    --scenarios "browsing,checkout" \
    --output-dir evaluation_results
```

**Returns**:
```json
{
    "status": "completed",
    "total_experiments": 15,
    "successful_experiments": 15,
    "failed_experiments": 0,
    "output_directory": "evaluation_results",
    "results_file": "evaluation_results/checkoutservice_comparative_results.json"
}
```

###### `evaluate analyze`
```bash
python3 -m src.mora.cli.main evaluate analyze [OPTIONS]
```

**Description**: Analyze evaluation results and generate reports

**Options**:
- `--service TEXT`: Service name to analyze (required)
- `--input-dir TEXT`: Input directory with results (default: "evaluation_results")
- `--output-dir TEXT`: Output directory for reports (default: "evaluation_results/reports")
- `--help`: Show help message

**Examples**:
```bash
# Analyze evaluation results
python3 -m src.mora.cli.main evaluate analyze \
    --service checkoutservice \
    --input-dir evaluation_results \
    --output-dir evaluation_results/reports
```

**Returns**:
```json
{
    "status": "completed",
    "report_file": "evaluation_results/reports/checkoutservice_comparative_report.md",
    "analysis": {
        "cost_efficiency": {
            "cpu_savings_percent": 28.17,
            "memory_savings_percent": 28.17
        },
        "performance_integrity": {
            "p95_latency": 0.0,
            "error_rate": 0.0
        },
        "stability": {
            "volatility": 0.0,
            "scaling_events": 0
        }
    }
}
```

##### 5. Rightsizing Commands

###### `rightsize`
```bash
python3 -m src.mora.cli.main rightsize [OPTIONS]
```

**Description**: Generate resource rightsizing recommendations

**Options**:
- `--services TEXT`: Comma-separated list of services (required)
- `--strategy TEXT`: Rightsizing strategy (default: "predictive")
- `--output-format TEXT`: Output format (default: "table")
- `--help`: Show help message

**Examples**:
```bash
# Generate recommendations for multiple services
python3 -m src.mora.cli.main rightsize \
    --services "frontend,cartservice,checkoutservice" \
    --strategy predictive \
    --output-format table
```

**Returns**:
```json
{
    "recommendations": {
        "frontend": {
            "cpu_cores": 0.5,
            "memory_gb": 0.5,
            "replicas": 2
        },
        "cartservice": {
            "cpu_cores": 0.25,
            "memory_gb": 0.25,
            "replicas": 1
        }
    },
    "strategy": "predictive",
    "timestamp": "2026-01-21T12:00:00Z"
}
```

## Core Module APIs

### DataAcquisitionPipeline

#### Class Definition
```python
class DataAcquisitionPipeline:
    def __init__(self, namespace: str = "hipster-shop", 
                 prometheus_url: str = "http://localhost:9090",
                 data_dir: str = "training_data"):
        """
        Initialize data acquisition pipeline
        
        Args:
            namespace: Kubernetes namespace
            prometheus_url: Prometheus server URL
            data_dir: Directory for storing training data
        """
```

#### Methods

##### `run_parallel_training_experiments`
```python
def run_parallel_training_experiments(self, services: List[str], 
                                    config: Dict[str, Any] = None,
                                    max_workers: int = 1) -> Dict[str, Any]:
    """
    Run parallel training experiments for multiple services
    
    Args:
        services: List of service names
        config: Configuration dictionary
        max_workers: Maximum number of parallel workers
    
    Returns:
        Dictionary with experiment results
    """
```

**Example Usage**:
```python
from src.mora.core.data_acquisition import DataAcquisitionPipeline

pipeline = DataAcquisitionPipeline()
result = pipeline.run_parallel_training_experiments(
    services=["frontend", "cartservice"],
    config={
        "training": {
            "steady_state_config": {
                "experiment_duration_minutes": 15,
                "replica_counts": [1, 2, 4],
                "load_levels_users": [5, 10, 20],
                "test_scenarios": ["browsing", "checkout"]
            }
        }
    },
    max_workers=1
)
```

##### `run_isolated_training_experiment`
```python
def run_isolated_training_experiment(self, service_name: str, 
                                   config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run isolated training experiment for a single service
    
    Args:
        service_name: Name of the service
        config: Configuration dictionary
    
    Returns:
        Dictionary with experiment results
    """
```

##### `process_collected_data_for_training`
```python
def process_collected_data_for_training(self, service_name: str) -> pd.DataFrame:
    """
    Process collected data for model training
    
    Args:
        service_name: Name of the service
    
    Returns:
        Processed DataFrame ready for training
    """
```

### PrometheusClient

#### Class Definition
```python
class PrometheusClient:
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        """
        Initialize Prometheus client
        
        Args:
            prometheus_url: Prometheus server URL
        """
```

#### Methods

##### `get_comprehensive_metrics`
```python
def get_comprehensive_metrics(self, namespace: str, service_name: str,
                            duration_minutes: int = 15) -> Dict[str, pd.DataFrame]:
    """
    Get comprehensive metrics for a service
    
    Args:
        namespace: Kubernetes namespace
        service_name: Name of the service
        duration_minutes: Duration for metrics collection
    
    Returns:
        Dictionary with metric DataFrames
    """
```

**Example Usage**:
```python
from src.mora.monitoring.prometheus_client import PrometheusClient

client = PrometheusClient()
metrics = client.get_comprehensive_metrics(
    namespace="hipster-shop",
    service_name="frontend",
    duration_minutes=15
)

# Access specific metrics
cpu_metrics = metrics['cpu_cores']
memory_metrics = metrics['mem_bytes']
```

##### `test_connection`
```python
def test_connection(self) -> bool:
    """
    Test connection to Prometheus server
    
    Returns:
        True if connection successful, False otherwise
    """
```

##### `get_metric_range`
```python
def get_metric_range(self, query: str, start_time: datetime, 
                    end_time: datetime) -> pd.DataFrame:
    """
    Get metric data for a specific time range
    
    Args:
        query: Prometheus query
        start_time: Start time for data collection
        end_time: End time for data collection
    
    Returns:
        DataFrame with metric data
    """
```

## Monitoring APIs

### System Health Monitoring

#### `check_system_health`
```python
def check_system_health() -> Dict[str, Any]:
    """
    Check overall system health
    
    Returns:
        Dictionary with health status
    """
```

**Returns**:
```json
{
    "minikube": "running",
    "prometheus": "ready",
    "hipster_shop": {
        "pods_running": 24,
        "services_available": 12
    },
    "system_resources": {
        "cpu_usage": "7.4%",
        "memory_usage": "45.2%",
        "disk_usage": "56%"
    },
    "overall_status": "healthy"
}
```

#### `monitor_data_collection`
```python
def monitor_data_collection() -> Dict[str, Any]:
    """
    Monitor data collection progress
    
    Returns:
        Dictionary with collection status
    """
```

**Returns**:
```json
{
    "process_running": true,
    "process_id": 845709,
    "csv_files": 108,
    "json_files": 108,
    "experiments_completed": 108,
    "estimated_completion": "5.25 days",
    "errors_detected": 0
}
```

## ML Pipeline APIs

### WorkingLSTMProphetPipeline

#### Class Definition
```python
class WorkingLSTMProphetPipeline:
    def __init__(self, data_dir: str = "training_data", 
                 model_dir: str = "models"):
        """
        Initialize LSTM + Prophet pipeline
        
        Args:
            data_dir: Directory containing training data
            model_dir: Directory for saving models
        """
```

#### Methods

##### `train_pipeline`
```python
def train_pipeline(self, service_name: str) -> Dict[str, Any]:
    """
    Train the complete LSTM + Prophet pipeline
    
    Args:
        service_name: Name of the service to train
    
    Returns:
        Dictionary with training results
    """
```

**Example Usage**:
```python
from train_models.train_working_lstm_prophet import WorkingLSTMProphetPipeline

pipeline = WorkingLSTMProphetPipeline()
result = pipeline.train_pipeline("frontend")

if result['status'] == 'success':
    print(f"✅ Pipeline trained successfully")
    print(f"📁 Model saved to: {result['model_path']}")
```

**Returns**:
```json
{
    "status": "success",
    "model_path": "models/frontend_lstm_prophet_pipeline.joblib",
    "training_time": "2.5 minutes",
    "data_samples": 2047,
    "model_performance": {
        "cpu_model_mse": 0.001239,
        "memory_model_mse": 195871846729430,
        "replica_model_mse": 0.430723
    },
    "fusion_results": {
        "cpu_prediction": 0.069870,
        "memory_prediction": 5569583,
        "replica_prediction": 2.006691
    }
}
```

##### `make_recommendations`
```python
def make_recommendations(self, service_name: str, 
                        current_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make resource recommendations using trained models
    
    Args:
        service_name: Name of the service
        current_metrics: Current metric values
    
    Returns:
        Dictionary with recommendations
    """
```

**Example Usage**:
```python
recommendations = pipeline.make_recommendations("frontend", {
    "cpu_cores_value": 0.0001,
    "mem_bytes_value": 10000000,
    "replica_count": 2,
    "load_users": 50
})

cpu_cores = recommendations['recommendations']['cpu_cores']
memory_bytes = recommendations['recommendations']['memory_bytes']
replicas = recommendations['recommendations']['replicas']
confidence = recommendations['confidence']
```

##### `load_training_data`
```python
def load_training_data(self, service_name: str) -> pd.DataFrame:
    """
    Load training data for a service
    
    Args:
        service_name: Name of the service
    
    Returns:
        Combined DataFrame with all training data
    """
```

##### `prepare_time_series_data`
```python
def prepare_time_series_data(self, df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], 
                                                              Dict[str, pd.DataFrame], 
                                                              List[str]]:
    """
    Prepare time series data for training
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (metrics_data, targets, feature_names)
    """
```

## Data Acquisition APIs

### Load Generator

#### `create_jmeter_script`
```python
def create_jmeter_script(self, service_name: str, scenario: str, 
                        replica_count: int, num_users: int,
                        duration_minutes: int = 15) -> str:
    """
    Create JMeter script for load testing
    
    Args:
        service_name: Name of the service
        scenario: Load scenario (browsing/checkout)
        replica_count: Number of replicas
        num_users: Number of concurrent users
        duration_minutes: Test duration in minutes
    
    Returns:
        Path to created JMeter script
    """
```

#### `run_load_test`
```python
def run_load_test(self, script_path: str, duration_minutes: int = 15) -> Dict[str, Any]:
    """
    Run JMeter load test
    
    Args:
        script_path: Path to JMeter script
        duration_minutes: Test duration in minutes
    
    Returns:
        Dictionary with test results
    """
```

### Kubernetes Client

#### `scale_deployment`
```python
def scale_deployment(self, service_name: str, replica_count: int) -> bool:
    """
    Scale deployment to specified replica count
    
    Args:
        service_name: Name of the service
        replica_count: Target replica count
    
    Returns:
        True if scaling successful, False otherwise
    """
```

#### `wait_for_pods_ready`
```python
def wait_for_pods_ready(self, service_name: str, timeout: int = 300) -> bool:
    """
    Wait for pods to be ready
    
    Args:
        service_name: Name of the service
        timeout: Timeout in seconds
    
    Returns:
        True if pods ready, False if timeout
    """
```

## Model Management APIs

### Model Library

#### `save_model`
```python
def save_model(self, service_name: str, model_data: Dict[str, Any]) -> str:
    """
    Save trained model to disk
    
    Args:
        service_name: Name of the service
        model_data: Model data dictionary
    
    Returns:
        Path to saved model file
    """
```

#### `load_model`
```python
def load_model(self, service_name: str) -> Dict[str, Any]:
    """
    Load trained model from disk
    
    Args:
        service_name: Name of the service
    
    Returns:
        Loaded model data dictionary
    """
```

#### `list_models`
```python
def list_models(self) -> List[str]:
    """
    List all available trained models
    
    Returns:
        List of service names with trained models
    """
```

### Model Evaluation

#### `evaluate_model`
```python
def evaluate_model(self, service_name: str) -> Dict[str, Any]:
    """
    Evaluate model performance
    
    Args:
        service_name: Name of the service
    
    Returns:
        Dictionary with evaluation metrics
    """
```

**Returns**:
```json
{
    "service_name": "frontend",
    "evaluation_date": "2024-10-25T12:00:00Z",
    "model_performance": {
        "cpu_model": {
            "mse": 0.001239,
            "mae": 0.025,
            "r2": 0.95
        },
        "memory_model": {
            "mse": 195871846729430,
            "mae": 12500000,
            "r2": 0.87
        },
        "replica_model": {
            "mse": 0.430723,
            "mae": 0.15,
            "r2": 0.92
        }
    },
    "fusion_performance": {
        "overall_confidence": 0.85,
        "prediction_accuracy": 0.92
    }
}
```

## Utility APIs

### Data Processing

#### `validate_data_quality`
```python
def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality
    
    Args:
        df: Input DataFrame
    
    Returns:
        Dictionary with quality metrics
    """
```

#### `clean_data`
```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess data
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
```

### File Operations

#### `save_experiment_data`
```python
def save_experiment_data(experiment_id: str, data: Dict[str, Any], 
                        data_dir: str = "training_data") -> str:
    """
    Save experiment data to disk
    
    Args:
        experiment_id: Unique experiment identifier
        data: Experiment data dictionary
        data_dir: Directory for saving data
    
    Returns:
        Path to saved data file
    """
```

#### `load_experiment_data`
```python
def load_experiment_data(experiment_id: str, 
                        data_dir: str = "training_data") -> Dict[str, Any]:
    """
    Load experiment data from disk
    
    Args:
        experiment_id: Unique experiment identifier
        data_dir: Directory containing data
    
    Returns:
        Loaded experiment data dictionary
    """
```

## Configuration APIs

### Configuration Management

#### `load_config`
```python
def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_file: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
```

#### `save_config`
```python
def save_config(config: Dict[str, Any], config_file: str) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        config_file: Path to configuration file
    
    Returns:
        True if save successful, False otherwise
    """
```

#### `validate_config`
```python
def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of validation errors (empty if valid)
    """
```

### Configuration Examples

#### Resource-Optimized Configuration
```python
config = {
    "training": {
        "steady_state_config": {
            "experiment_duration_minutes": 15,
            "replica_counts": [1, 2, 4],
            "load_levels_users": [5, 10, 20, 30, 50, 75],
            "test_scenarios": ["browsing", "checkout"]
        }
    },
    "max_parallel_workers": 1,
    "cpu_limit_per_worker": 2,
    "memory_limit_per_worker": 2,
    "jmeter_thread_ramp_up": 30,
    "jmeter_thread_ramp_down": 30,
    "jmeter_think_time": 2,
    "metrics_collection_interval": 30,
    "prometheus_query_timeout": 10,
    "max_retries": 3
}
```

#### Default Configuration
```python
config = {
    "training": {
        "steady_state_config": {
            "experiment_duration_minutes": 45,
            "replica_counts": [1, 2, 4, 6],
            "load_levels_users": [10, 50, 100, 150, 200, 250],
            "test_scenarios": ["browsing", "checkout"]
        }
    }
}
```

---

**API Reference Version**: 1.0  
**Last Updated**: January 21, 2026  
**Compatible With**: MOrA v1.0  
**API Type**: CLI + Python Module APIs  
**Documentation**: Comprehensive with examples and return types
