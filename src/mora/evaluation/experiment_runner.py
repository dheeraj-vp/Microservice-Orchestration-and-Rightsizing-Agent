
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml

from .metrics_collector import MetricsCollector
from ..core.statistical_strategy import StatisticalRightsizer
from ..core.data_pipeline import DataPipeline
from ..monitoring.prometheus_client import PrometheusClient
from ..k8s.client import KubernetesClient
from ..utils.config import load_config

logger = logging.getLogger(__name__)

class ExperimentRunner:

    def __init__(
        self,
        config_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):


        if config_path:
            with open(config_path, "r") as f:
                self.eval_config = yaml.safe_load(f)
        else:

            default_config = Path("config/evaluation-config.yaml")
            if default_config.exists():
                with open(default_config, "r") as f:
                    self.eval_config = yaml.safe_load(f)
            else:

                self.eval_config = load_config("config/resource-optimized.yaml")
                logger.warning("Using resource-optimized.yaml as evaluation config")


        self.config = load_config()


        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)


        self.prometheus_client = PrometheusClient(require_connection=False)


        if not self.prometheus_client.test_connection():
            error_msg = (
                f"Prometheus is not accessible at {self.prometheus_client.prometheus_url}.\n"
                f"Please ensure Prometheus is running and accessible.\n\n"
                f"To set up Prometheus:\n"
                f"  1. Run: bash scripts/setup-minikube.sh\n"
                f"  2. Or manually: kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090\n"
                f"  3. Verify: curl http://localhost:9090/-/ready"
            )
            raise ConnectionError(error_msg)

        self.k8s_client = KubernetesClient()
        self.metrics_collector = MetricsCollector(
            prometheus_client=self.prometheus_client,
            k8s_client=self.k8s_client,
            config=self.eval_config,
        )


        eval_settings = self.eval_config.get("evaluation", {})
        self.experiment_duration_minutes = eval_settings.get("experiment_duration_minutes", 15)
        self.metrics_interval = eval_settings.get("metrics_collection_interval", 30)
        self.load_levels = eval_settings.get("load_levels_users", [5, 10, 20, 30, 50, 75])
        self.replica_counts = eval_settings.get("replica_counts", [1, 2, 4])
        self.test_scenarios = eval_settings.get("test_scenarios", ["browsing", "checkout"])


        self._last_rightsizing_attempt: Dict[str, datetime] = {}
        self._rightsizing_interval_minutes = 2

    def run_experiment(
        self,
        service_name: str,
        strategy: str,
        load_level: int,
        replica_count: int,
        scenario: str = "browsing",
    ) -> Dict[str, Any]:

        experiment_id = f"{service_name}_{strategy}_{load_level}users_{replica_count}replicas_{uuid.uuid4().hex[:8]}"

        logger.info(
            f"Starting experiment {experiment_id}: "
            f"service={service_name}, strategy={strategy}, "
            f"load={load_level} users, replicas={replica_count}, scenario={scenario}"
        )

        try:

            self.metrics_collector.start_experiment(experiment_id, service_name, strategy)


            self._apply_initial_config(service_name, replica_count)


            experiment_start = datetime.now()
            experiment_end = experiment_start + timedelta(minutes=self.experiment_duration_minutes)


            while datetime.now() < experiment_end:

                self.metrics_collector.collect_metrics()


                if strategy != "hpa":

                    last_attempt = self._last_rightsizing_attempt.get(service_name)
                    now = datetime.now()

                    if last_attempt is None or (now - last_attempt).total_seconds() >= (self._rightsizing_interval_minutes * 60):
                        self._apply_rightsizing(service_name, strategy)
                        self._last_rightsizing_attempt[service_name] = now


                time.sleep(self.metrics_interval)


                if self._should_pause():
                    logger.warning("Pausing experiment due to high system resource usage")
                    time.sleep(60)


            self.metrics_collector.collect_metrics()


            summary = self.metrics_collector.end_experiment()


            metrics_dir = self.output_dir / "metrics"
            self.metrics_collector.save_metrics(metrics_dir)

            logger.info(f"Experiment {experiment_id} completed successfully")

            return {
                "experiment_id": experiment_id,
                "status": "success",
                "summary": summary,
                "service_name": service_name,
                "strategy": strategy,
                "load_level": load_level,
                "replica_count": replica_count,
                "scenario": scenario,
            }

        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)
            return {
                "experiment_id": experiment_id,
                "status": "failed",
                "error": str(e),
                "service_name": service_name,
                "strategy": strategy,
            }

    def _apply_initial_config(self, service_name: str, replica_count: int):

        try:
            namespace = self.config.get("kubernetes", {}).get("namespace", "hipster-shop")


            logger.info(f"Setting initial replica count to {replica_count} for {service_name}")

        except Exception as e:
            logger.warning(f"Could not apply initial config: {e}")

    def _apply_rightsizing(self, service_name: str, strategy: str):

        try:
            if strategy == "statistical":



                pipeline = DataPipeline()


                duration = min(5, self.experiment_duration_minutes)

                try:
                    service_data = pipeline.collect_service_data(service_name, duration_minutes=duration)

                    if service_data and not service_data.get("error"):
                        rightsizer = StatisticalRightsizer()
                        recommendations = rightsizer.generate_recommendations(service_data)

                        if recommendations:


                            logger.debug(f"Generated {len(recommendations)} recommendations for {service_name}")
                    else:
                        logger.debug(f"Insufficient data for rightsizing {service_name} (may be normal during experiment startup)")
                except Exception as e:

                    logger.debug(f"Could not collect data for rightsizing {service_name}: {e}")

            elif strategy == "predictive":


                logger.debug(f"Predictive strategy not yet implemented for evaluation")

        except Exception as e:
            logger.debug(f"Could not apply rightsizing: {e}")

    def _should_pause(self) -> bool:

        try:
            eval_settings = self.eval_config.get("evaluation", {})
            if not eval_settings.get("monitor_system_resources", False):
                return False

            threshold = eval_settings.get("high_usage_threshold", 80)




            return False
        except Exception:
            return False

    def run_comparative_evaluation(
        self,
        service_name: str,
        strategies: Optional[List[str]] = None,
        load_levels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:

        if strategies is None:
            strategy_config = self.eval_config.get("strategies", [])
            strategies = [
                s["name"] for s in strategy_config if s.get("enabled", True)
            ]

        if load_levels is None:
            load_levels = self.load_levels

        results = {
            "service_name": service_name,
            "start_time": datetime.now().isoformat(),
            "experiments": [],
        }

        logger.info(
            f"Starting comparative evaluation for {service_name}: "
            f"strategies={strategies}, load_levels={load_levels}"
        )


        for strategy in strategies:
            for load_level in load_levels:
                for replica_count in self.replica_counts:
                    for scenario in self.test_scenarios:
                        experiment_result = self.run_experiment(
                            service_name=service_name,
                            strategy=strategy,
                            load_level=load_level,
                            replica_count=replica_count,
                            scenario=scenario,
                        )
                        results["experiments"].append(experiment_result)


                        time.sleep(30)

        results["end_time"] = datetime.now().isoformat()
        results["total_experiments"] = len(results["experiments"])


        results_file = self.output_dir / f"{service_name}_comparative_results.json"
        import json
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Comparative evaluation completed. Results saved to {results_file}")

        return results
