
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


with patch.dict('sys.modules', {'prometheus_api_client': Mock()}):
    from src.mora.monitoring.prometheus_client import PrometheusClient

class TestPrometheusClient:

    def setup_method(self):

        self.mock_prometheus_client = Mock()

    @patch('src.mora.monitoring.prometheus_client.PrometheusConnect')
    def test_initialization(self, mock_prometheus_connect):

        mock_instance = Mock()
        mock_instance.check_prometheus_connection = Mock()
        mock_prometheus_connect.return_value = mock_instance

        client = PrometheusClient("http://localhost:9090")

        assert client.prometheus_url == "http://localhost:9090"
        assert client.timeout == 30
        mock_prometheus_connect.assert_called_once()
        mock_instance.check_prometheus_connection.assert_called_once()

    @patch('src.mora.monitoring.prometheus_client.PrometheusConnect')
    def test_initialization_with_custom_url(self, mock_prometheus_connect):

        mock_instance = Mock()
        mock_instance.check_prometheus_connection = Mock()
        mock_prometheus_connect.return_value = mock_instance

        client = PrometheusClient("http://custom-prometheus:8080", timeout=60)

        assert client.prometheus_url == "http://custom-prometheus:8080"
        assert client.timeout == 60

    @patch('src.mora.monitoring.prometheus_client.PrometheusConnect')
    def test_connection_test_success(self, mock_prometheus_connect):

        mock_instance = Mock()
        mock_instance.check_prometheus_connection = Mock()
        mock_instance.query = Mock(return_value=[Mock()])
        mock_prometheus_connect.return_value = mock_instance

        client = PrometheusClient()
        client.client = mock_instance

        result = client.test_connection()

        assert result == True
        mock_instance.query.assert_called_with("up")

    @patch('src.mora.monitoring.prometheus_client.PrometheusConnect')
    def test_connection_test_failure(self, mock_prometheus_connect):

        mock_instance = Mock()
        mock_instance.check_prometheus_connection = Mock()
        mock_instance.query = Mock(side_effect=Exception("Connection failed"))
        mock_prometheus_connect.return_value = mock_instance

        client = PrometheusClient()
        client.client = mock_instance

        result = client.test_connection()

        assert result == False

    @patch('src.mora.monitoring.prometheus_client.pd.DataFrame')
    @patch('src.mora.monitoring.prometheus_client.PrometheusConnect')
    def test_get_cpu_metrics(self, mock_prometheus_connect, mock_dataframe):

        mock_instance = Mock()
        mock_instance.check_prometheus_connection = Mock()
        mock_instance.get_metric_range_data = Mock(return_value=[Mock()])
        mock_prometheus_connect.return_value = mock_instance

        client = PrometheusClient()
        client.client = mock_instance


        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        result = client.get_cpu_metrics("test-namespace", "test-service", 60)


        mock_instance.get_metric_range_data.assert_called_once()
        call_args = mock_instance.get_metric_range_data.call_args
        assert "test-namespace" in call_args[1]["query"]
        assert "test-service" in call_args[1]["query"]

    @patch('src.mora.monitoring.prometheus_client.pd.DataFrame')
    @patch('src.mora.monitoring.prometheus_client.PrometheusConnect')
    def test_get_memory_metrics(self, mock_prometheus_connect, mock_dataframe):

        mock_instance = Mock()
        mock_instance.check_prometheus_connection = Mock()
        mock_instance.get_metric_range_data = Mock(return_value=[Mock()])
        mock_prometheus_connect.return_value = mock_instance

        client = PrometheusClient()
        client.client = mock_instance


        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        result = client.get_memory_metrics("test-namespace", "test-service", 60)


        mock_instance.get_metric_range_data.assert_called_once()
        call_args = mock_instance.get_metric_range_data.call_args
        assert "test-namespace" in call_args[1]["query"]
        assert "test-service" in call_args[1]["query"]

    @patch('src.mora.monitoring.prometheus_client.PrometheusConnect')
    def test_get_resource_requests(self, mock_prometheus_connect):

        mock_instance = Mock()
        mock_instance.check_prometheus_connection = Mock()


        mock_cpu_result = Mock()
        mock_cpu_result.metric = {'pod': 'test-pod', 'container': 'test-container'}
        mock_cpu_result.value = [1234567890, '0.2']

        mock_memory_result = Mock()
        mock_memory_result.metric = {'pod': 'test-pod', 'container': 'test-container'}
        mock_memory_result.value = [1234567890, '134217728']

        mock_instance.query.side_effect = [[mock_cpu_result], [mock_memory_result]]
        mock_prometheus_connect.return_value = mock_instance

        client = PrometheusClient()
        client.client = mock_instance

        result = client.get_resource_requests("test-namespace", "test-service")

        assert 'cpu' in result
        assert 'memory' in result
        assert len(result['cpu']) == 1
        assert len(result['memory']) == 1
        assert result['cpu'][0]['value'] == 0.2
        assert result['memory'][0]['value'] == 134217728.0

    @patch('src.mora.monitoring.prometheus_client.PrometheusConnect')
    def test_query_instant(self, mock_prometheus_connect):

        mock_instance = Mock()
        mock_instance.check_prometheus_connection = Mock()

        mock_result = Mock()
        mock_result.metric = {'instance': 'test-instance'}
        mock_result.value = [1234567890, '42']
        mock_result.timestamp = '2023-01-01T00:00:00Z'

        mock_instance.query.return_value = [mock_result]
        mock_prometheus_connect.return_value = mock_instance

        client = PrometheusClient()
        client.client = mock_instance

        result = client.query_instant("up")

        assert len(result) == 1
        assert result[0]['metric']['instance'] == 'test-instance'
        assert result[0]['value'] == [1234567890, '42']

    @patch('src.mora.monitoring.prometheus_client.PrometheusConnect')
    def test_empty_data_handling(self, mock_prometheus_connect):

        mock_instance = Mock()
        mock_instance.check_prometheus_connection = Mock()
        mock_instance.get_metric_range_data = Mock(return_value=[])
        mock_prometheus_connect.return_value = mock_instance

        client = PrometheusClient()
        client.client = mock_instance


        with patch('src.mora.monitoring.prometheus_client.MetricRangeDataFrame', return_value=pd.DataFrame()):
            result = client.get_cpu_metrics("test-namespace", "test-service", 60)


            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
