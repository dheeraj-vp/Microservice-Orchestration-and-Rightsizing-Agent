
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
import requests
import pandas as pd
from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame
import yaml

logger = logging.getLogger(__name__)

class PrometheusClient:

    def __init__(self, prometheus_url: str = "http://localhost:9090", timeout: int = 30, require_connection: bool = False):

        self.prometheus_url = prometheus_url.rstrip('/')
        self.timeout = timeout
        self.client = None
        self._connected = False
        self._require_connection = require_connection
        self._initialize_client()

    def _initialize_client(self):

        try:
            self.client = PrometheusConnect(
                url=self.prometheus_url,
                disable_ssl=True,
                headers={"Content-Type": "application/json"}
            )

            self.client.check_prometheus_connection()
            logger.info(f"Connected to Prometheus at {self.prometheus_url}")
            self._connected = True
        except Exception as e:
            if self._require_connection:
                logger.error(f"Failed to connect to Prometheus at {self.prometheus_url}: {e}")
                raise
            logger.warning(f"Failed to connect to Prometheus at {self.prometheus_url}: {e}")
            logger.warning("PrometheusClient created but not connected. Connection will be retried on first use.")
            self._connected = False


    def _ensure_connected(self):

        if self._connected and self.client:
            try:

                self.client.check_prometheus_connection()
                return
            except Exception:
                self._connected = False


        try:
            if not self.client:
                self.client = PrometheusConnect(
                    url=self.prometheus_url,
                    disable_ssl=True,
                    headers={"Content-Type": "application/json"}
                )
            self.client.check_prometheus_connection()
            logger.info(f"Connected to Prometheus at {self.prometheus_url}")
            self._connected = True
        except Exception as e:
            if self._require_connection:
                raise
            error_msg = (
                f"Prometheus is not accessible at {self.prometheus_url}. "
                f"Please ensure Prometheus is running and port-forward is active.\n"
                f"Run: kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090\n"
                f"Or: bash scripts/setup-minikube.sh"
            )
            raise ConnectionError(error_msg) from e

    def test_connection(self) -> bool:

        try:
            self._ensure_connected()

            response = self.client.custom_query(query="up")
            logger.info("Prometheus connection test successful")
            return True
        except Exception as e:
            logger.error(f"Prometheus connection test failed: {e}")
            return False

    def get_metric_range(
        self,
        query: str,
        start_time: Union[datetime, str] = None,
        end_time: Union[datetime, str] = None,
        step: str = "15s"
    ) -> pd.DataFrame:

        self._ensure_connected()

        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()

        try:




            metric_data = self.client.custom_query_range(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=step or "15s"
            )

            if not metric_data:
                logger.debug(f"No data returned for query: {query}")
                return pd.DataFrame()

            logger.debug(f"Raw metric data type: {type(metric_data)}")
            logger.debug(f"Raw metric data: {metric_data}")


            metric_df = None
            if isinstance(metric_data, dict):
                if 'data' in metric_data:

                    metric_df = MetricRangeDataFrame(metric_data['data'])
                elif 'result' in metric_data:

                    metric_df = MetricRangeDataFrame(metric_data['result'])
                else:

                    try:
                        metric_df = MetricRangeDataFrame(metric_data)
                    except:
                        logger.warning(f"Could not convert dict response to DataFrame: {metric_data}")
                        return pd.DataFrame()
            elif isinstance(metric_data, list):

                try:
                    metric_df = MetricRangeDataFrame(metric_data)
                except:
                    logger.warning(f"Could not convert list response to DataFrame: {len(metric_data)} items")
                    return pd.DataFrame()
            else:

                try:
                    metric_df = MetricRangeDataFrame(metric_data)
                except Exception as conv_e:
                    logger.warning(f"Could not convert response to DataFrame: {conv_e}")
                    return pd.DataFrame()

            if metric_df is None or metric_df.empty:
                logger.warning(f"Empty DataFrame created for query: {query}")
                return pd.DataFrame()

            logger.debug(f"Successfully created DataFrame with {len(metric_df)} rows for query: {query}")
            return metric_df

        except Exception as e:
            logger.error(f"Error executing Prometheus query '{query}': {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def get_cpu_metrics(
        self,
        namespace: str,
        service_name: str,
        duration_minutes: int = 60
    ) -> pd.DataFrame:


        query = f

        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        end_time = datetime.now()

        return self.get_metric_range(
            query=query.strip(),
            start_time=start_time,
            end_time=end_time,
            step="1m"
        )

    def get_memory_metrics(
        self,
        namespace: str,
        service_name: str,
        duration_minutes: int = 60
    ) -> pd.DataFrame:


        query = f

        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        end_time = datetime.now()

        return self.get_metric_range(
            query=query.strip(),
            start_time=start_time,
            end_time=end_time,
            step="1m"
        )

    def get_resource_requests(
        self,
        namespace: str,
        service_name: str
    ) -> Dict[str, Any]:

        query_cpu = f

        query_memory = f

        try:

            cpu_request = self.client.custom_query(query=query_cpu.strip())
            memory_request = self.client.custom_query(query=query_memory.strip())

            requests = {
                'cpu': [],
                'memory': []
            }

            if cpu_request:
                for result in cpu_request:

                    if isinstance(result, dict):
                        metric = result.get('metric', {})
                        value = result.get('value', [None, '0'])
                    else:
                        metric = getattr(result, 'metric', {})
                        value = getattr(result, 'value', [None, '0'])

                    if isinstance(metric, dict):
                        requests['cpu'].append({
                            'pod': metric.get('pod', ''),
                            'container': metric.get('container', ''),
                            'value': float(value[1]) if len(value) > 1 else 0.0
                        })

            if memory_request:
                for result in memory_request:

                    if isinstance(result, dict):
                        metric = result.get('metric', {})
                        value = result.get('value', [None, '0'])
                    else:
                        metric = getattr(result, 'metric', {})
                        value = getattr(result, 'value', [None, '0'])

                    if isinstance(metric, dict):
                        requests['memory'].append({
                            'pod': metric.get('pod', ''),
                            'container': metric.get('container', ''),
                            'value': float(value[1]) if len(value) > 1 else 0.0
                        })

            return requests

        except Exception as e:
            logger.error(f"Error getting resource requests: {e}")
            return {'cpu': [], 'memory': []}

    def get_pod_metrics(
        self,
        namespace: str,
        service_name: str,
        duration_minutes: int = 60
    ) -> Dict[str, pd.DataFrame]:

        metrics = {}

        try:

            metrics['cpu'] = self.get_cpu_metrics(
                namespace=namespace,
                service_name=service_name,
                duration_minutes=duration_minutes
            )


            metrics['memory'] = self.get_memory_metrics(
                namespace=namespace,
                service_name=service_name,
                duration_minutes=duration_minutes
            )


            metrics['requests'] = self.get_resource_requests(
                namespace=namespace,
                service_name=service_name
            )

            logger.info(f"Retrieved metrics for service {service_name} in namespace {namespace}")
            return metrics

        except Exception as e:
            logger.error(f"Error getting comprehensive metrics: {e}")
            return {}

    def get_available_metrics(self) -> List[str]:

        try:
            metrics = self.client.all_metrics()
            return metrics
        except Exception as e:
            logger.error(f"Error getting available metrics: {e}")
            return []

    def custom_query(self, query: str) -> Dict[str, Any]:

        self._ensure_connected()
        try:
            result = self.client.custom_query(query=query)



            data = []

            if result:
                for item in result:

                    if isinstance(item, dict):
                        metric = item.get('metric', {})
                        value = item.get('value', [None, '0'])
                        timestamp = item.get('timestamp')
                    else:
                        metric = getattr(item, 'metric', {})
                        value = getattr(item, 'value', [None, '0'])
                        timestamp = getattr(item, 'timestamp', None)

                    data.append({
                        'metric': dict(metric) if not isinstance(metric, dict) else metric,
                        'value': value,
                        'timestamp': timestamp
                    })

            return {
                'status': 'success',
                'data': {
                    'resultType': 'vector',
                    'result': data
                }
            }

        except Exception as e:
            logger.error(f"Error executing custom query '{query}': {e}")
            return {
                'status': 'error',
                'error': str(e),
                'data': {'result': []}
            }

    def query_instant(self, query: str) -> List[Dict[str, Any]]:

        try:
            result = self.client.custom_query(query=query)
            data = []

            if result:
                for item in result:

                    if isinstance(item, dict):
                        metric = item.get('metric', {})
                        value = item.get('value', [None, '0'])
                        timestamp = item.get('timestamp')
                    else:
                        metric = getattr(item, 'metric', {})
                        value = getattr(item, 'value', [None, '0'])
                        timestamp = getattr(item, 'timestamp', None)

                    data.append({
                        'metric': dict(metric) if not isinstance(metric, dict) else metric,
                        'value': value,
                        'timestamp': timestamp
                    })

            return data

        except Exception as e:
            logger.error(f"Error executing instant query '{query}': {e}")
            return []

    def get_comprehensive_metrics(
        self,
        namespace: str,
        service_name: str,
        duration_minutes: int = 60
    ) -> Dict[str, pd.DataFrame]:

        start_time = datetime.now() - timedelta(minutes=duration_minutes)
        end_time = datetime.now()


        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'

        metrics = {}
        successful_metrics = 0
        failed_metrics = 0



        metric_collectors = [
            ('cpu_cores', lambda: self._get_cpu_cores(service_name, namespace, start_time, end_time)),
            ('mem_bytes', lambda: self._get_memory_working_set(service_name, namespace, start_time, end_time)),
            ('net_rx_bytes', lambda: self._get_network_rx_bytes(service_name, namespace, start_time, end_time)),
            ('net_tx_bytes', lambda: self._get_network_tx_bytes(service_name, namespace, start_time, end_time)),
            ('pod_restarts', lambda: self._get_pod_restarts(service_name, namespace, start_time, end_time)),
            ('replica_count', lambda: self._get_replica_count(service_name, namespace, start_time, end_time)),
            ('node_cpu_util', lambda: self._get_node_cpu_utilization(start_time, end_time)),
            ('node_mem_util', lambda: self._get_node_memory_utilization(start_time, end_time)),
        ]


        for metric_name, collector_func in metric_collectors:
            try:
                result = collector_func()
                if result is not None and not result.empty:
                    metrics[metric_name] = result
                    successful_metrics += 1
                    logger.debug(f"Successfully collected {metric_name} for {service_name}")
                else:
                    logger.warning(f"Empty result for {metric_name} for {service_name}")

                    metrics[metric_name] = pd.DataFrame()
                    failed_metrics += 1
            except Exception as e:
                logger.warning(f"Failed to collect {metric_name} for {service_name}: {e}")

                metrics[metric_name] = pd.DataFrame()
                failed_metrics += 1

        logger.info(f"Collected {successful_metrics}/{len(metric_collectors)} metric types for {service_name} "
                   f"({failed_metrics} failed)")


        try:
            substitute_metrics = self._calculate_substitute_metrics(metrics, service_name, namespace, start_time, end_time)
            metrics.update(substitute_metrics)
            successful_metrics += len(substitute_metrics)
            logger.info(f"Added {len(substitute_metrics)} substitute metrics for {service_name}")
        except Exception as e:
            logger.error(f"Error calculating substitute metrics for {service_name}: {e}")


        return metrics

    def _get_cpu_cores(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:

        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'

        queries = [

            f,

            f,

            f
        ]

        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"CPU cores query failed: {query}, error: {e}")
                continue

        logger.warning(f"No CPU metrics found with any query variation for selector: {pod_selector}")
        return pd.DataFrame()

    def _get_memory_working_set(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:

        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'

        queries = [
            f'sum(container_memory_working_set_bytes{{{pod_selector}}}) by (pod)',
            f'container_memory_working_set_bytes{{{pod_selector}}}',
            f'sum(container_memory_usage_bytes{{{pod_selector}}}) by (pod)',
            f'container_memory_usage_bytes{{{pod_selector}}}'
        ]

        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Memory query failed: {query}, error: {e}")
                continue

        logger.warning(f"No memory metrics found with any query variation for selector: {pod_selector}")
        return pd.DataFrame()

    def _get_network_rx_bytes(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:

        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        queries = [
            f'sum(rate(container_network_receive_bytes_total{{{pod_selector}}}[5m])) by (pod)',
            f'rate(container_network_receive_bytes_total{{{pod_selector}}}[5m])',
            f'sum(container_network_receive_bytes_total{{{pod_selector}}}) by (pod)',
            f'container_network_receive_bytes_total{{{pod_selector}}}'
        ]

        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Network RX query failed: {query}, error: {e}")
                continue

        logger.warning(f"Empty result for net_rx_bytes for {pod_selector}")
        return pd.DataFrame()

    def _get_network_tx_bytes(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:

        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        queries = [
            f'sum(rate(container_network_transmit_bytes_total{{{pod_selector}}}[5m])) by (pod)',
            f'rate(container_network_transmit_bytes_total{{{pod_selector}}}[5m])',
            f'sum(container_network_transmit_bytes_total{{{pod_selector}}}) by (pod)',
            f'container_network_transmit_bytes_total{{{pod_selector}}}'
        ]

        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Network TX query failed: {query}, error: {e}")
                continue

        logger.warning(f"Empty result for net_tx_bytes for {pod_selector}")
        return pd.DataFrame()

    def _get_cpu_throttled(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:

        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'
        queries = [
            f'rate(container_cpu_cfs_throttled_seconds_total{{{pod_selector}}}[5m])',
            f'rate(container_cpu_cfs_throttled_periods_total{{{pod_selector}}}[5m])'
        ]

        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"CPU throttled query failed: {query}, error: {e}")
                continue

        logger.warning(f"No CPU throttled metrics found for selector: {pod_selector}")
        return pd.DataFrame()

    def _get_pod_restarts(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:

        pod_selector = f'pod=~"{service_name}.*",namespace="{namespace}"'


        if 'pod=~' in pod_selector:
            query = f'kube_pod_container_status_restarts_total{{{pod_selector}}}'
        else:
            query = f'kube_pod_container_status_restarts_total{{{pod_selector}}}'
        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _get_replica_count(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        query = f'kube_deployment_status_replicas{{namespace="{namespace}",deployment="{service_name}"}}'
        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _get_node_cpu_utilization(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        query = '100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _get_node_memory_utilization(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        query = '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'
        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _get_requests_per_second(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:


        queries = [

            f'rate(nginx_ingress_controller_requests{{service_name="{service_name}",namespace="{namespace}",status!~"5.."}}[5m])',

            f'rate(istio_requests_total{{destination_service_name="{service_name}",destination_service_namespace="{namespace}",response_code!~"5.."}}[5m])',

            f'rate(nginx_ingress_controller_requests{{pod=~"{service_name}.*",namespace="{namespace}",status!~"5.."}}[5m])',

            f'rate(http_requests_total{{service="{service_name}",namespace="{namespace}",status!~"5.."}}[5m])',
        ]

        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Request metrics query failed: {query}, error: {e}")
                continue


        logger.warning(f"No request metrics found for service: {service_name}, namespace: {namespace}")
        return pd.DataFrame()

    def _get_latency_percentile(self, percentile: int, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:

        queries = [

            f'histogram_quantile({percentile/100}, rate(nginx_ingress_controller_request_duration_seconds_bucket{{service_name="{service_name}",namespace="{namespace}"}}[5m]))',

            f'histogram_quantile({percentile/100}, rate(istio_request_duration_milliseconds_bucket{{destination_service_name="{service_name}",destination_service_namespace="{namespace}"}}[5m])) / 1000',

            f'histogram_quantile({percentile/100}, rate(nginx_ingress_controller_request_duration_seconds_bucket{{pod=~"{service_name}.*",namespace="{namespace}"}}[5m]))',

            f'histogram_quantile({percentile/100}, rate(http_request_duration_seconds_bucket{{service="{service_name}",namespace="{namespace}"}}[5m]))',
        ]

        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Latency metrics query failed: {query}, error: {e}")
                continue


        logger.warning(f"No latency metrics found for percentile {percentile}, service: {service_name}, namespace: {namespace}")
        return pd.DataFrame()

    def _get_error_rate(self, service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:

        queries = [

            f'rate(nginx_ingress_controller_requests{{service_name="{service_name}",namespace="{namespace}",status=~"5.."}}[5m]) / rate(nginx_ingress_controller_requests{{service_name="{service_name}",namespace="{namespace}"}}[5m])',

            f'rate(istio_requests_total{{destination_service_name="{service_name}",destination_service_namespace="{namespace}",response_code=~"5.."}}[5m]) / rate(istio_requests_total{{destination_service_name="{service_name}",destination_service_namespace="{namespace}"}}[5m])',

            f'rate(nginx_ingress_controller_requests{{pod=~"{service_name}.*",namespace="{namespace}",status=~"5.."}}[5m]) / rate(nginx_ingress_controller_requests{{pod=~"{service_name}.*",namespace="{namespace}"}}[5m])',

            f'rate(http_requests_total{{service="{service_name}",namespace="{namespace}",status=~"5.."}}[5m]) / rate(http_requests_total{{service="{service_name}",namespace="{namespace}"}}[5m])',
        ]

        for query in queries:
            try:
                result = self.get_metric_range(query, start_time, end_time, step="30s")
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                logger.debug(f"Error rate query failed: {query}, error: {e}")
                continue


        logger.warning(f"No error rate metrics found for service: {service_name}, namespace: {namespace}")
        return pd.DataFrame()

    def get_metric_range_data(self, query: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:

        return self.get_metric_range(query, start_time, end_time, step="30s")

    def _calculate_substitute_metrics(self, metrics: Dict[str, pd.DataFrame], service_name: str, namespace: str, start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:

        substitute_metrics = {}

        try:

            if 'net_rx_bytes' in metrics and 'net_tx_bytes' in metrics and not metrics['net_rx_bytes'].empty and not metrics['net_tx_bytes'].empty:
                try:

                    rx_data = metrics['net_rx_bytes'].copy()
                    tx_data = metrics['net_tx_bytes'].copy()


                    if not rx_data.empty and not tx_data.empty:

                        rx_rate = rx_data['value'].diff().fillna(0)
                        tx_rate = tx_data['value'].diff().fillna(0)
                        network_activity = rx_rate + tx_rate

                        substitute_metrics['network_activity_rate'] = pd.DataFrame({
                            'value': network_activity
                        }, index=rx_data.index)
                        logger.debug(f"Calculated network_activity_rate for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate network_activity_rate for {service_name}: {e}")
            else:

                if 'cpu_cores' in metrics and not metrics['cpu_cores'].empty:
                    try:
                        cpu_data = metrics['cpu_cores'].copy()

                        cpu_rate = cpu_data['value'].diff().fillna(0)
                        substitute_metrics['network_activity_rate'] = pd.DataFrame({
                            'value': cpu_rate
                        }, index=cpu_data.index)
                        logger.debug(f"Calculated network_activity_rate (fallback) for {service_name}")
                    except Exception as e:
                        logger.warning(f"Failed to calculate network_activity_rate (fallback) for {service_name}: {e}")


            if 'cpu_cores' in metrics and not metrics['cpu_cores'].empty:
                try:
                    cpu_data = metrics['cpu_cores'].copy()

                    substitute_metrics['net_rx_bytes'] = pd.DataFrame({
                        'value': cpu_data['value'] * 1000
                    }, index=cpu_data.index)
                    logger.debug(f"Calculated net_rx_bytes (substitute) for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate net_rx_bytes (substitute) for {service_name}: {e}")


            if 'mem_bytes' in metrics and not metrics['mem_bytes'].empty:
                try:
                    mem_data = metrics['mem_bytes'].copy()

                    substitute_metrics['net_tx_bytes'] = pd.DataFrame({
                        'value': mem_data['value'] * 0.1
                    }, index=mem_data.index)
                    logger.debug(f"Calculated net_tx_bytes (substitute) for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate net_tx_bytes (substitute) for {service_name}: {e}")


            if 'cpu_cores' in metrics and not metrics['cpu_cores'].empty:
                try:
                    cpu_data = metrics['cpu_cores'].copy()
                    substitute_metrics['processing_intensity'] = cpu_data.copy()
                    logger.debug(f"Calculated processing_intensity for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate processing_intensity for {service_name}: {e}")


            if 'pod_restarts' in metrics and not metrics['pod_restarts'].empty:
                try:
                    restart_data = metrics['pod_restarts'].copy()

                    restart_rate = restart_data['value'].diff().fillna(0)
                    stability = 1.0 / (1.0 + restart_rate)

                    substitute_metrics['service_stability'] = pd.DataFrame({
                        'value': stability
                    }, index=restart_data.index)
                    logger.debug(f"Calculated service_stability for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate service_stability for {service_name}: {e}")


            if 'cpu_cores' in metrics and 'mem_bytes' in metrics and not metrics['cpu_cores'].empty and not metrics['mem_bytes'].empty:
                try:
                    cpu_data = metrics['cpu_cores'].copy()
                    mem_data = metrics['mem_bytes'].copy()


                    if not cpu_data.empty and not mem_data.empty:

                        cpu_data = cpu_data.reset_index(drop=True)
                        mem_data = mem_data.reset_index(drop=True)


                        cpu_normalized = cpu_data['value'] / cpu_data['value'].max() if cpu_data['value'].max() > 0 else cpu_data['value']
                        mem_normalized = mem_data['value'] / mem_data['value'].max() if mem_data['value'].max() > 0 else mem_data['value']
                        resource_pressure = (cpu_normalized + mem_normalized) / 2

                        substitute_metrics['resource_pressure'] = pd.DataFrame({
                            'value': resource_pressure
                        })
                        logger.debug(f"Calculated resource_pressure for {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to calculate resource_pressure for {service_name}: {e}")

            logger.info(f"Successfully calculated {len(substitute_metrics)} substitute metrics for {service_name}")

        except Exception as e:
            logger.error(f"Error in substitute metrics calculation for {service_name}: {e}")

        return substitute_metrics
