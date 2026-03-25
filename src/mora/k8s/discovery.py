
import logging
from typing import List, Dict, Optional, Set
from .client import KubernetesClient

logger = logging.getLogger(__name__)

class ServiceDiscovery:

    def __init__(self, k8s_client: KubernetesClient):

        self.k8s_client = k8s_client

    def discover_services(self, namespace: str) -> List[Dict[str, str]]:

        try:
            deployments = self.k8s_client.get_deployments(namespace=namespace)
            services = []

            for deployment in deployments:
                service_info = {
                    'name': deployment['name'],
                    'namespace': deployment['namespace'],
                    'type': 'deployment',
                    'replicas': deployment['replicas'],
                    'ready_replicas': deployment['ready_replicas'],
                    'containers': [container['name'] for container in deployment['containers']],
                    'labels': deployment.get('labels', {}),
                    'status': 'ready' if deployment['ready_replicas'] == deployment['replicas'] else 'not_ready'
                }
                services.append(service_info)

            logger.info(f"Discovered {len(services)} services in namespace {namespace}")
            return services

        except Exception as e:
            logger.error(f"Error discovering services in namespace {namespace}: {e}")
            return []

    def get_microservices(self, namespace: str, filter_patterns: Optional[List[str]] = None) -> List[str]:

        services = self.discover_services(namespace)
        service_names = [service['name'] for service in services]

        if filter_patterns:
            filtered_services = []
            for pattern in filter_patterns:
                for service_name in service_names:
                    if pattern.lower() in service_name.lower():
                        filtered_services.append(service_name)
            return list(set(filtered_services))

        return service_names

    def get_service_labels(self, namespace: str) -> Dict[str, Dict[str, str]]:

        services = self.discover_services(namespace)
        labels_map = {}

        for service in services:
            labels_map[service['name']] = service['labels']

        return labels_map

    def get_services_by_label(self, namespace: str, label_key: str, label_value: str = None) -> List[Dict[str, str]]:

        services = self.discover_services(namespace)
        filtered_services = []

        for service in services:
            labels = service.get('labels', {})
            if label_key in labels:
                if label_value is None or labels[label_key] == label_value:
                    filtered_services.append(service)

        return filtered_services

    def validate_service_exists(self, service_name: str, namespace: str) -> bool:

        try:
            deployment = self.k8s_client.get_deployment(service_name, namespace)
            return deployment is not None
        except Exception as e:
            logger.error(f"Error validating service {service_name}: {e}")
            return False
