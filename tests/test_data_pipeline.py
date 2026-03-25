
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.mora.core.data_pipeline import DataPipeline

class TestDataPipeline:

    def setup_method(self):

        self.mock_k8s_client = Mock()
        self.mock_prometheus_client = Mock()
        self.mock_service_discovery = Mock()

    @patch('src.mora.core.data_pipeline.KubernetesClient')
    @patch('src.mora.core.data_pipeline.PrometheusClient')
    @patch('src.mora.core.data_pipeline.ServiceDiscovery')
    def test_initialization(self, mock_service_discovery, mock_prometheus_client, mock_k8s_client):

        mock_k8s_instance = Mock()
        mock_k8s_client.return_value = mock_k8s_instance

        mock_prom_instance = Mock()
        mock_prometheus_client.return_value = mock_prom_instance

        mock_discovery_instance = Mock()
        mock_service_discovery.return_value = mock_discovery_instance

        pipeline = DataPipeline(namespace="test-namespace", prometheus_url="http://test:9090")

        assert pipeline.namespace == "test-namespace"
        mock_k8s_client.assert_called_once_with(namespace="test-namespace")
        mock_prometheus_client.assert_called_once_with("http://test:9090")
        mock_service_discovery.assert_called_once_with(mock_k8s_instance)

    @patch('src.mora.core.data_pipeline.KubernetesClient')
    @patch('src.mora.core.data_pipeline.PrometheusClient')
    @patch('src.mora.core.data_pipeline.ServiceDiscovery')
    def test_test_connections_success(self, mock_service_discovery, mock_prometheus_client, mock_k8s_client):


        mock_k8s_instance = Mock()
        mock_k8s_instance.test_connection.return_value = True
        mock_k8s_client.return_value = mock_k8s_instance

        mock_prom_instance = Mock()
        mock_prom_instance.test_connection.return_value = True
        mock_prometheus_client.return_value = mock_prom_instance

        mock_service_discovery.return_value = Mock()

        pipeline = DataPipeline()
        pipeline.k8s_client = mock_k8s_instance
        pipeline.prometheus_client = mock_prom_instance

        result = pipeline.test_connections()

        assert result['kubernetes'] == True
        assert result['prometheus'] == True

    @patch('src.mora.core.data_pipeline.KubernetesClient')
    @patch('src.mora.core.data_pipeline.PrometheusClient')
    @patch('src.mora.core.data_pipeline.ServiceDiscovery')
    def test_test_connections_failure(self, mock_service_discovery, mock_prometheus_client, mock_k8s_client):


        mock_k8s_instance = Mock()
        mock_k8s_instance.test_connection.side_effect = Exception("K8s failed")
        mock_k8s_client.return_value = mock_k8s_instance

        mock_prom_instance = Mock()
        mock_prom_instance.test_connection.return_value = False
        mock_prometheus_client.return_value = mock_prom_instance

        mock_service_discovery.return_value = Mock()

        pipeline = DataPipeline()
        pipeline.k8s_client = mock_k8s_instance
        pipeline.prometheus_client = mock_prom_instance

        result = pipeline.test_connections()

        assert result['kubernetes'] == False
        assert result['prometheus'] == False

    @patch('src.mora.core.data_pipeline.KubernetesClient')
    @patch('src.mora.core.data_pipeline.PrometheusClient')
    @patch('src.mora.core.data_pipeline.ServiceDiscovery')
    def test_collect_service_data_success(self, mock_service_discovery, mock_prometheus_client, mock_k8s_client):


        mock_k8s_instance = Mock()
        mock_service_discovery_instance = Mock()


        mock_deployment = {
            'name': 'test-service',
            'namespace': 'test-namespace',
            'replicas': 2,
            'ready_replicas': 2,
            'containers': [{'name': 'test-container'}]
        }

        mock_k8s_instance.get_deployment.return_value = mock_deployment
        mock_k8s_instance.get_pods.return_value = [{'name': 'test-pod'}]


        mock_cpu_df = pd.DataFrame({'__value': [0.1, 0.2, 0.3]})
        mock_memory_df = pd.DataFrame({'__value': [100, 200, 300]})
        mock_requests = {'cpu': [], 'memory': []}

        mock_prom_instance = Mock()
        mock_prom_instance.get_pod_metrics.return_value = {
            'cpu': mock_cpu_df,
            'memory': mock_memory_df,
            'requests': mock_requests
        }

        mock_service_discovery_instance.validate_service_exists.return_value = True

        mock_k8s_client.return_value = mock_k8s_instance
        mock_prometheus_client.return_value = mock_prom_instance
        mock_service_discovery.return_value = mock_service_discovery_instance

        pipeline = DataPipeline()
        pipeline.k8s_client = mock_k8s_instance
        pipeline.prometheus_client = mock_prom_instance
        pipeline.service_discovery = mock_service_discovery_instance

        result = pipeline.collect_service_data("test-service", 60)

        assert result['service_name'] == 'test-service'
        assert result['namespace'] == 'test-namespace'
        assert 'deployment' in result
        assert 'pods' in result
        assert 'metrics' in result
        assert result['deployment']['name'] == 'test-service'

    @patch('src.mora.core.data_pipeline.KubernetesClient')
    @patch('src.mora.core.data_pipeline.PrometheusClient')
    @patch('src.mora.core.data_pipeline.ServiceDiscovery')
    def test_collect_service_data_service_not_found(self, mock_service_discovery, mock_prometheus_client, mock_k8s_client):

        mock_service_discovery_instance = Mock()
        mock_service_discovery_instance.validate_service_exists.return_value = False

        mock_service_discovery.return_value = mock_service_discovery_instance
        mock_k8s_client.return_value = Mock()
        mock_prometheus_client.return_value = Mock()

        pipeline = DataPipeline()
        pipeline.service_discovery = mock_service_discovery_instance

        with pytest.raises(ValueError, match="Service test-service not found"):
            pipeline.collect_service_data("test-service", 60)

    @patch('src.mora.core.data_pipeline.KubernetesClient')
    @patch('src.mora.core.data_pipeline.PrometheusClient')
    @patch('src.mora.core.data_pipeline.ServiceDiscovery')
    def test_validate_data_quality(self, mock_service_discovery, mock_prometheus_client, mock_k8s_client):

        mock_service_discovery.return_value = Mock()
        mock_k8s_client.return_value = Mock()
        mock_prometheus_client.return_value = Mock()

        pipeline = DataPipeline()


        good_data = {
            'deployment': {'name': 'test-service'},
            'metrics': {
                'cpu': pd.DataFrame({'__value': [0.1, 0.2]}),
                'memory': pd.DataFrame({'__value': [100, 200]})
            },
            'collection_timestamp': datetime.now()
        }

        validation = pipeline.validate_data_quality(good_data)

        assert validation['is_valid'] == True
        assert len(validation['issues']) == 0

    @patch('src.mora.core.data_pipeline.KubernetesClient')
    @patch('src.mora.core.data_pipeline.PrometheusClient')
    @patch('src.mora.core.data_pipeline.ServiceDiscovery')
    def test_validate_data_quality_missing_deployment(self, mock_service_discovery, mock_prometheus_client, mock_k8s_client):

        mock_service_discovery.return_value = Mock()
        mock_k8s_client.return_value = Mock()
        mock_prometheus_client.return_value = Mock()

        pipeline = DataPipeline()


        bad_data = {
            'metrics': {
                'cpu': pd.DataFrame({'__value': [0.1, 0.2]}),
                'memory': pd.DataFrame({'__value': [100, 200]})
            }
        }

        validation = pipeline.validate_data_quality(bad_data)

        assert validation['is_valid'] == False
        assert len(validation['issues']) > 0
        assert "No deployment information found" in validation['issues']

    @patch('src.mora.core.data_pipeline.KubernetesClient')
    @patch('src.mora.core.data_pipeline.PrometheusClient')
    @patch('src.mora.core.data_pipeline.ServiceDiscovery')
    def test_discover_all_services(self, mock_service_discovery, mock_prometheus_client, mock_k8s_client):

        mock_service_discovery_instance = Mock()
        mock_service_discovery_instance.get_microservices.return_value = ['service1', 'service2', 'service3']
        mock_service_discovery.return_value = mock_service_discovery_instance

        mock_k8s_client.return_value = Mock()
        mock_prometheus_client.return_value = Mock()

        pipeline = DataPipeline()
        pipeline.service_discovery = mock_service_discovery_instance

        result = pipeline.discover_all_services()

        assert result == ['service1', 'service2', 'service3']
        mock_service_discovery_instance.get_microservices.assert_called_once_with(pipeline.namespace)

    @patch('src.mora.core.data_pipeline.KubernetesClient')
    @patch('src.mora.core.data_pipeline.PrometheusClient')
    @patch('src.mora.core.data_pipeline.ServiceDiscovery')
    def test_get_system_summary(self, mock_service_discovery, mock_prometheus_client, mock_k8s_client):


        mock_k8s_instance = Mock()
        mock_service_discovery_instance = Mock()
        mock_prom_instance = Mock()

        mock_service_discovery_instance.get_microservices.return_value = ['service1', 'service2']
        mock_k8s_instance.test_connection.return_value = True
        mock_k8s_instance.get_deployment.return_value = {
            'name': 'service1',
            'replicas': 2,
            'ready_replicas': 2,
            'containers': [{'name': 'container1'}]
        }

        mock_prom_instance.test_connection.return_value = True

        mock_k8s_client.return_value = mock_k8s_instance
        mock_prometheus_client.return_value = mock_prom_instance
        mock_service_discovery.return_value = mock_service_discovery_instance

        pipeline = DataPipeline()
        pipeline.k8s_client = mock_k8s_instance
        pipeline.prometheus_client = mock_prom_instance
        pipeline.service_discovery = mock_service_discovery_instance

        result = pipeline.get_system_summary()

        assert result['namespace'] == pipeline.namespace
        assert result['total_services'] == 2
        assert result['services'] == ['service1', 'service2']
        assert result['connections']['kubernetes'] == True
        assert result['connections']['prometheus'] == True
        assert 'service_stats' in result
        assert 'timestamp' in result
