
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


with patch.dict('sys.modules', {'kubernetes': Mock()}):
    from src.mora.k8s.client import KubernetesClient

class TestKubernetesClient:

    def setup_method(self):

        self.mock_k8s_client = Mock()

    @patch('src.mora.k8s.client.config')
    @patch('src.mora.k8s.client.client')
    def test_initialization(self, mock_client, mock_config):


        mock_config.load_kube_config = Mock()
        mock_client.AppsV1Api = Mock(return_value="apps_api")
        mock_client.CoreV1Api = Mock(return_value="core_api")

        client = KubernetesClient(namespace="test-namespace")
        assert client.namespace == "test-namespace"
        assert client.apps_v1 == "apps_api"
        assert client.core_v1 == "core_api"

    @patch('src.mora.k8s.client.config')
    @patch('src.mora.k8s.client.client')
    def test_connection_test_success(self, mock_client, mock_config):

        mock_config.load_kube_config = Mock()
        mock_core_api = Mock()
        mock_core_api.list_node = Mock(return_value="nodes")
        mock_client.CoreV1Api = Mock(return_value=mock_core_api)

        client = KubernetesClient()
        result = client.test_connection()

        assert result == True
        mock_core_api.list_node.assert_called_once()

    @patch('src.mora.k8s.client.config')
    @patch('src.mora.k8s.client.client')
    def test_get_deployment_success(self, mock_client, mock_config):

        mock_config.load_kube_config = Mock()
        mock_apps_api = Mock()


        mock_deployment = Mock()
        mock_deployment.metadata.name = "test-deployment"
        mock_deployment.metadata.namespace = "test-namespace"
        mock_deployment.spec.replicas = 3
        mock_deployment.status.ready_replicas = 3
        mock_deployment.metadata.labels = {"app": "test"}
        mock_deployment.metadata.creation_timestamp = datetime.now()


        mock_container = Mock()
        mock_container.name = "test-container"
        mock_container.image = "test-image"
        mock_container.resources.requests = {"cpu": "100m", "memory": "128Mi"}
        mock_container.resources.limits = {"cpu": "200m", "memory": "256Mi"}

        mock_deployment.spec.template.spec.containers = [mock_container]

        mock_apps_api.read_namespaced_deployment.return_value = mock_deployment
        mock_client.AppsV1Api = Mock(return_value=mock_apps_api)
        mock_client.CoreV1Api = Mock()

        client = KubernetesClient()
        result = client.get_deployment("test-deployment", "test-namespace")

        assert result is not None
        assert result['name'] == "test-deployment"
        assert result['replicas'] == 3
        assert result['ready_replicas'] == 3
        assert len(result['containers']) == 1
        assert result['containers'][0]['name'] == "test-container"

    @patch('src.mora.k8s.client.config')
    @patch('src.mora.k8s.client.client')
    def test_parse_cpu_and_memory_values(self, mock_client, mock_config):

        mock_config.load_kube_config = Mock()
        mock_client.AppsV1Api = Mock()
        mock_client.CoreV1Api = Mock()

        client = KubernetesClient()




    @patch('src.mora.k8s.client.config')
    @patch('src.mora.k8s.client.client')
    def test_get_deployments_empty(self, mock_client, mock_config):

        mock_config.load_kube_config = Mock()
        mock_apps_api = Mock()
        mock_response = Mock()
        mock_response.items = []
        mock_apps_api.list_namespaced_deployment.return_value = mock_response
        mock_client.AppsV1Api = Mock(return_value=mock_apps_api)
        mock_client.CoreV1Api = Mock()

        client = KubernetesClient()
        result = client.get_deployments("test-namespace")

        assert result == []
        mock_apps_api.list_namespaced_deployment.assert_called_once_with(namespace="test-namespace")
