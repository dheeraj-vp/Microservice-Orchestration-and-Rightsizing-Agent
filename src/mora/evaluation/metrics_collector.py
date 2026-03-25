
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from ..monitoring.prometheus_client import PrometheusClient
from ..k8s.client import KubernetesClient
from ..utils.config import load_config

logger = logging.getLogger(__name__)

class MetricsCollector:

    def __init__(
        self,
        prometheus_client: Optional[PrometheusClient] = None,
        k8s_client: Optional[KubernetesClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.prometheus_client = prometheus_client or PrometheusClient()
        self.k8s_client = k8s_client or KubernetesClient()
        self.config = config or load_config()

        self.metrics_interval = self.config.get("evaluation", {}).get(
            "metrics_collection_interval", 30
        )

        self.collected_metrics: List[Dict[str, Any]] = []
        self.experiment_start_time: Optional[datetime] = None
        self.experiment_end_time: Optional[datetime] = None

    def start_experiment(self, experiment_id: str, service_name: str, strategy: str):

        self.experiment_start_time = datetime.now()
        self.experiment_id = experiment_id
        self.service_name = service_name
        self.strategy = strategy
        self.collected_metrics = []

        logger.info(
            f"Started metrics collection for experiment {experiment_id} "
            f"(service: {service_name}, strategy: {strategy})"
        )

    def collect_metrics(self) -> Dict[str, Any]:

        if not self.experiment_start_time:
            raise ValueError("Experiment not started. Call start_experiment() first.")

        timestamp = datetime.now()
        elapsed_seconds = (timestamp - self.experiment_start_time).total_seconds()

        metrics = {
            "timestamp": timestamp.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "experiment_id": self.experiment_id,
            "service_name": self.service_name,
            "strategy": self.strategy,
        }


        metrics.update(self._collect_cost_metrics())


        metrics.update(self._collect_performance_metrics())


        metrics.update(self._collect_stability_metrics())

        self.collected_metrics.append(metrics)
        return metrics

    def _collect_cost_metrics(self) -> Dict[str, Any]:

        try:

            pods = self.k8s_client.get_pods(
                namespace=self.config.get("kubernetes", {}).get("namespace", "hipster-shop"),
                label_selector=f"app={self.service_name}",
            )

            total_cpu_cores = 0.0
            total_memory_gb = 0.0
            replica_count = len(pods)

            for pod in pods:
                containers = pod.get("containers", [])
                for container in containers:
                    resources = container.get("resources", {})
                    requests = resources.get("requests", {})


                    cpu_str = requests.get("cpu", "0")
                    if cpu_str.endswith("m"):
                        total_cpu_cores += float(cpu_str[:-1]) / 1000.0
                    else:
                        total_cpu_cores += float(cpu_str) if cpu_str else 0.0


                    memory_str = requests.get("memory", "0")
                    if memory_str.endswith("Mi"):
                        total_memory_gb += float(memory_str[:-2]) / 1024.0
                    elif memory_str.endswith("Gi"):
                        total_memory_gb += float(memory_str[:-2])
                    else:
                        total_memory_gb += float(memory_str) / (1024**3) if memory_str else 0.0


            elapsed_hours = self.metrics_interval / 3600.0
            cpu_hours = total_cpu_cores * elapsed_hours
            memory_hours = total_memory_gb * elapsed_hours

            return {
                "cost_efficiency": {
                    "cpu_cores": total_cpu_cores,
                    "memory_gb": total_memory_gb,
                    "replica_count": replica_count,
                    "cpu_hours": cpu_hours,
                    "memory_hours": memory_hours,
                    "elapsed_hours": elapsed_hours,
                }
            }
        except Exception as e:
            logger.error(f"Error collecting cost metrics: {e}")
            return {"cost_efficiency": {}}

    def _collect_performance_metrics(self) -> Dict[str, Any]:

        try:
            namespace = self.config.get("kubernetes", {}).get("namespace", "hipster-shop")



            latency_query = (
                f'histogram_quantile(0.95, '
                f'rate(http_request_duration_seconds_bucket{{namespace="{namespace}",'
                f'service="{self.service_name}"}}[1m]))'
            )
            latency_result = self.prometheus_client.custom_query(latency_query)
            p95_latency = self._extract_metric_value(latency_result, default=0.0)


            error_query = (
                f'rate(http_requests_total{{namespace="{namespace}",'
                f'service="{self.service_name}",status=~"5.."}}[1m]) / '
                f'rate(http_requests_total{{namespace="{namespace}",'
                f'service="{self.service_name}"}}[1m])'
            )
            error_result = self.prometheus_client.custom_query(error_query)
            error_rate = self._extract_metric_value(error_result, default=0.0)


            throughput_query = (
                f'rate(http_requests_total{{namespace="{namespace}",'
                f'service="{self.service_name}"}}[1m])'
            )
            throughput_result = self.prometheus_client.custom_query(throughput_query)
            throughput = self._extract_metric_value(throughput_result, default=0.0)


            avg_latency_query = (
                f'histogram_quantile(0.50, '
                f'rate(http_request_duration_seconds_bucket{{namespace="{namespace}",'
                f'service="{self.service_name}"}}[1m]))'
            )
            avg_latency_result = self.prometheus_client.custom_query(avg_latency_query)
            avg_response_time = self._extract_metric_value(avg_latency_result, default=0.0)

            return {
                "performance_integrity": {
                    "p95_latency": p95_latency,
                    "avg_response_time": avg_response_time,
                    "error_rate": error_rate,
                    "throughput": throughput,
                    "sla_compliance": 1.0 - error_rate if error_rate < 0.05 else 0.0,
                }
            }
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return {"performance_integrity": {}}

    def _collect_stability_metrics(self) -> Dict[str, Any]:

        try:
            namespace = self.config.get("kubernetes", {}).get("namespace", "hipster-shop")


            pods = self.k8s_client.get_pods(
                namespace=namespace,
                label_selector=f"app={self.service_name}",
            )
            current_replicas = len(pods)


            cpu_query = (
                f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}",'
                f'pod=~"{self.service_name}-.*"}}[1m])'
            )
            cpu_result = self.prometheus_client.custom_query(cpu_query)
            cpu_utilization = self._extract_metric_value(cpu_result, default=0.0)


            if len(self.collected_metrics) > 1:
                recent_cpu_values = [
                    m.get("stability", {}).get("cpu_utilization", 0.0)
                    for m in self.collected_metrics[-10:]
                ]
                if recent_cpu_values:
                    import statistics
                    volatility = statistics.stdev(recent_cpu_values) if len(recent_cpu_values) > 1 else 0.0
                else:
                    volatility = 0.0
            else:
                volatility = 0.0


            scaling_event = False
            if len(self.collected_metrics) > 0:
                prev_replicas = self.collected_metrics[-1].get("stability", {}).get("replica_count", current_replicas)
                if prev_replicas != current_replicas:
                    scaling_event = True

            return {
                "stability": {
                    "replica_count": current_replicas,
                    "cpu_utilization": cpu_utilization,
                    "volatility": volatility,
                    "scaling_event": scaling_event,
                }
            }
        except Exception as e:
            logger.error(f"Error collecting stability metrics: {e}")
            return {"stability": {}}

    def _extract_metric_value(self, query_result: Dict[str, Any], default: float = 0.0) -> float:

        try:
            if query_result.get("status") == "success":
                data = query_result.get("data", {})
                result = data.get("result", [])
                if result and len(result) > 0:
                    value = result[0].get("value", [None, None])[1]
                    return float(value) if value else default
            return default
        except Exception as e:
            logger.warning(f"Error extracting metric value: {e}")
            return default

    def end_experiment(self) -> Dict[str, Any]:

        if not self.experiment_start_time:
            raise ValueError("Experiment not started.")

        self.experiment_end_time = datetime.now()
        duration = (self.experiment_end_time - self.experiment_start_time).total_seconds()


        summary = self._calculate_summary(duration)

        logger.info(
            f"Ended metrics collection for experiment {self.experiment_id}. "
            f"Duration: {duration:.1f}s, Collected {len(self.collected_metrics)} data points"
        )

        return summary

    def _calculate_summary(self, duration_seconds: float) -> Dict[str, Any]:

        if not self.collected_metrics:
            return {}

        duration_hours = duration_seconds / 3600.0


        total_cpu_hours = sum(
            m.get("cost_efficiency", {}).get("cpu_hours", 0.0) for m in self.collected_metrics
        )
        total_memory_hours = sum(
            m.get("cost_efficiency", {}).get("memory_hours", 0.0) for m in self.collected_metrics
        )


        latencies = [
            m.get("performance_integrity", {}).get("p95_latency", 0.0)
            for m in self.collected_metrics
            if m.get("performance_integrity", {}).get("p95_latency", 0.0) > 0
        ]
        error_rates = [
            m.get("performance_integrity", {}).get("error_rate", 0.0)
            for m in self.collected_metrics
        ]
        throughputs = [
            m.get("performance_integrity", {}).get("throughput", 0.0)
            for m in self.collected_metrics
        ]


        scaling_events = sum(
            1 for m in self.collected_metrics
            if m.get("stability", {}).get("scaling_event", False)
        )
        volatilities = [
            m.get("stability", {}).get("volatility", 0.0)
            for m in self.collected_metrics
        ]

        summary = {
            "experiment_id": self.experiment_id,
            "service_name": self.service_name,
            "strategy": self.strategy,
            "start_time": self.experiment_start_time.isoformat(),
            "end_time": self.experiment_end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "duration_hours": duration_hours,
            "data_points": len(self.collected_metrics),
            "cost_efficiency": {
                "total_cpu_hours": total_cpu_hours,
                "total_memory_hours": total_memory_hours,
                "avg_cpu_cores": total_cpu_hours / duration_hours if duration_hours > 0 else 0.0,
                "avg_memory_gb": total_memory_hours / duration_hours if duration_hours > 0 else 0.0,
            },
            "performance_integrity": {
                "avg_p95_latency": sum(latencies) / len(latencies) if latencies else 0.0,
                "max_p95_latency": max(latencies) if latencies else 0.0,
                "avg_error_rate": sum(error_rates) / len(error_rates) if error_rates else 0.0,
                "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0.0,
            },
            "stability": {
                "scaling_events": scaling_events,
                "avg_volatility": sum(volatilities) / len(volatilities) if volatilities else 0.0,
                "max_volatility": max(volatilities) if volatilities else 0.0,
            },
        }

        return summary

    def save_metrics(self, output_dir: Path):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)


        metrics_file = output_dir / f"{self.experiment_id}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.collected_metrics, f, indent=2, default=str)

        logger.info(f"Saved metrics to {metrics_file}")


        summary = self.end_experiment()
        summary_file = output_dir / f"{self.experiment_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved summary to {summary_file}")

        return summary
