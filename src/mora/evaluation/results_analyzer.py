
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class ResultsAnalyzer:

    def __init__(self, results_dir: Optional[Path] = None):

        self.results_dir = Path(results_dir) if results_dir else Path("evaluation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_experiment_results(self, service_name: str) -> Dict[str, Any]:

        results_file = self.results_dir / f"{service_name}_comparative_results.json"

        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            return {}

        with open(results_file, "r") as f:
            return json.load(f)

    def analyze_cost_efficiency(self, results: Dict[str, Any]) -> Dict[str, Any]:

        experiments = results.get("experiments", [])

        if not experiments:
            return {}


        strategy_metrics = {}

        for exp in experiments:
            if exp.get("status") != "success":
                continue

            strategy = exp.get("strategy")
            summary = exp.get("summary", {})
            cost_eff = summary.get("cost_efficiency", {})

            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    "total_cpu_hours": [],
                    "total_memory_hours": [],
                    "avg_cpu_cores": [],
                    "avg_memory_gb": [],
                }

            strategy_metrics[strategy]["total_cpu_hours"].append(
                cost_eff.get("total_cpu_hours", 0.0)
            )
            strategy_metrics[strategy]["total_memory_hours"].append(
                cost_eff.get("total_memory_hours", 0.0)
            )
            strategy_metrics[strategy]["avg_cpu_cores"].append(
                cost_eff.get("avg_cpu_cores", 0.0)
            )
            strategy_metrics[strategy]["avg_memory_gb"].append(
                cost_eff.get("avg_memory_gb", 0.0)
            )


        analysis = {}

        for strategy, metrics in strategy_metrics.items():
            analysis[strategy] = {
                "mean_cpu_hours": sum(metrics["total_cpu_hours"]) / len(metrics["total_cpu_hours"]) if metrics["total_cpu_hours"] else 0.0,
                "mean_memory_hours": sum(metrics["total_memory_hours"]) / len(metrics["total_memory_hours"]) if metrics["total_memory_hours"] else 0.0,
                "mean_cpu_cores": sum(metrics["avg_cpu_cores"]) / len(metrics["avg_cpu_cores"]) if metrics["avg_cpu_cores"] else 0.0,
                "mean_memory_gb": sum(metrics["avg_memory_gb"]) / len(metrics["avg_memory_gb"]) if metrics["avg_memory_gb"] else 0.0,
            }


        if "statistical" in analysis and "predictive" in analysis:
            baseline_cpu = analysis["statistical"]["mean_cpu_hours"]
            mora_cpu = analysis["predictive"]["mean_cpu_hours"]

            if baseline_cpu > 0:
                cpu_savings = ((baseline_cpu - mora_cpu) / baseline_cpu) * 100
            else:
                cpu_savings = 0.0

            baseline_memory = analysis["statistical"]["mean_memory_hours"]
            mora_memory = analysis["predictive"]["mean_memory_hours"]

            if baseline_memory > 0:
                memory_savings = ((baseline_memory - mora_memory) / baseline_memory) * 100
            else:
                memory_savings = 0.0

            analysis["cost_savings"] = {
                "cpu_savings_percent": cpu_savings,
                "memory_savings_percent": memory_savings,
                "overall_savings_percent": (cpu_savings + memory_savings) / 2.0,
            }

        return analysis

    def analyze_performance_integrity(self, results: Dict[str, Any]) -> Dict[str, Any]:

        experiments = results.get("experiments", [])

        if not experiments:
            return {}


        strategy_metrics = {}

        for exp in experiments:
            if exp.get("status") != "success":
                continue

            strategy = exp.get("strategy")
            summary = exp.get("summary", {})
            perf = summary.get("performance_integrity", {})

            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    "avg_p95_latency": [],
                    "max_p95_latency": [],
                    "avg_error_rate": [],
                    "avg_throughput": [],
                    "sla_compliance": [],
                }

            strategy_metrics[strategy]["avg_p95_latency"].append(
                perf.get("avg_p95_latency", 0.0)
            )
            strategy_metrics[strategy]["max_p95_latency"].append(
                perf.get("max_p95_latency", 0.0)
            )
            strategy_metrics[strategy]["avg_error_rate"].append(
                perf.get("avg_error_rate", 0.0)
            )
            strategy_metrics[strategy]["avg_throughput"].append(
                perf.get("avg_throughput", 0.0)
            )


        analysis = {}

        for strategy, metrics in strategy_metrics.items():
            analysis[strategy] = {
                "mean_p95_latency": sum(metrics["avg_p95_latency"]) / len(metrics["avg_p95_latency"]) if metrics["avg_p95_latency"] else 0.0,
                "max_p95_latency": max(metrics["max_p95_latency"]) if metrics["max_p95_latency"] else 0.0,
                "mean_error_rate": sum(metrics["avg_error_rate"]) / len(metrics["avg_error_rate"]) if metrics["avg_error_rate"] else 0.0,
                "mean_throughput": sum(metrics["avg_throughput"]) / len(metrics["avg_throughput"]) if metrics["avg_throughput"] else 0.0,
            }

        return analysis

    def analyze_stability(self, results: Dict[str, Any]) -> Dict[str, Any]:

        experiments = results.get("experiments", [])

        if not experiments:
            return {}


        strategy_metrics = {}

        for exp in experiments:
            if exp.get("status") != "success":
                continue

            strategy = exp.get("strategy")
            summary = exp.get("summary", {})
            stability = summary.get("stability", {})

            if strategy not in strategy_metrics:
                strategy_metrics[strategy] = {
                    "scaling_events": [],
                    "avg_volatility": [],
                    "max_volatility": [],
                }

            strategy_metrics[strategy]["scaling_events"].append(
                stability.get("scaling_events", 0)
            )
            strategy_metrics[strategy]["avg_volatility"].append(
                stability.get("avg_volatility", 0.0)
            )
            strategy_metrics[strategy]["max_volatility"].append(
                stability.get("max_volatility", 0.0)
            )


        analysis = {}

        for strategy, metrics in strategy_metrics.items():
            analysis[strategy] = {
                "mean_scaling_events": sum(metrics["scaling_events"]) / len(metrics["scaling_events"]) if metrics["scaling_events"] else 0.0,
                "total_scaling_events": sum(metrics["scaling_events"]),
                "mean_volatility": sum(metrics["avg_volatility"]) / len(metrics["avg_volatility"]) if metrics["avg_volatility"] else 0.0,
                "max_volatility": max(metrics["max_volatility"]) if metrics["max_volatility"] else 0.0,
            }

        return analysis

    def generate_comparative_report(
        self, service_name: str, output_format: str = "markdown"
    ) -> str:

        results = self.load_experiment_results(service_name)

        if not results:
            return f"No results found for {service_name}"


        cost_analysis = self.analyze_cost_efficiency(results)
        perf_analysis = self.analyze_performance_integrity(results)
        stability_analysis = self.analyze_stability(results)

        if output_format == "markdown":
            return self._generate_markdown_report(
                service_name, cost_analysis, perf_analysis, stability_analysis
            )
        elif output_format == "json":
            return json.dumps(
                {
                    "service_name": service_name,
                    "cost_efficiency": cost_analysis,
                    "performance_integrity": perf_analysis,
                    "stability": stability_analysis,
                },
                indent=2,
                default=str,
            )
        else:
            return "Unsupported output format"

    def _generate_markdown_report(
        self,
        service_name: str,
        cost_analysis: Dict[str, Any],
        perf_analysis: Dict[str, Any],
        stability_analysis: Dict[str, Any],
    ) -> str:

        report = f"# Comparative Evaluation Report: {service_name}\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"


        report += "## Cost Efficiency Analysis\n\n"
        if cost_analysis:
            report += "| Strategy | Mean CPU Hours | Mean Memory Hours | Mean CPU Cores | Mean Memory GB |\n"
            report += "|----------|----------------|-------------------|----------------|---------------|\n"

            for strategy, metrics in cost_analysis.items():
                if strategy != "cost_savings":
                    report += (
                        f"| {strategy} | "
                        f"{metrics.get('mean_cpu_hours', 0):.4f} | "
                        f"{metrics.get('mean_memory_hours', 0):.4f} | "
                        f"{metrics.get('mean_cpu_cores', 0):.2f} | "
                        f"{metrics.get('mean_memory_gb', 0):.2f} |\n"
                    )

            if "cost_savings" in cost_analysis:
                savings = cost_analysis["cost_savings"]
                report += f"\n### Cost Savings (MOrA vs Baseline)\n\n"
                report += f"- CPU Savings: {savings.get('cpu_savings_percent', 0):.2f}%\n"
                report += f"- Memory Savings: {savings.get('memory_savings_percent', 0):.2f}%\n"
                report += f"- Overall Savings: {savings.get('overall_savings_percent', 0):.2f}%\n\n"
        else:
            report += "No cost efficiency data available.\n\n"


        report += "## Performance Integrity Analysis\n\n"
        if perf_analysis:
            report += "| Strategy | Mean P95 Latency | Max P95 Latency | Mean Error Rate | Mean Throughput |\n"
            report += "|----------|------------------|-----------------|-----------------|-----------------|\n"

            for strategy, metrics in perf_analysis.items():
                report += (
                    f"| {strategy} | "
                    f"{metrics.get('mean_p95_latency', 0):.4f}s | "
                    f"{metrics.get('max_p95_latency', 0):.4f}s | "
                    f"{metrics.get('mean_error_rate', 0):.4f} | "
                    f"{metrics.get('mean_throughput', 0):.2f} req/s |\n"
                )
        else:
            report += "No performance integrity data available.\n\n"


        report += "## Stability Analysis\n\n"
        if stability_analysis:
            report += "| Strategy | Mean Scaling Events | Total Scaling Events | Mean Volatility | Max Volatility |\n"
            report += "|----------|---------------------|----------------------|-----------------|----------------|\n"

            for strategy, metrics in stability_analysis.items():
                report += (
                    f"| {strategy} | "
                    f"{metrics.get('mean_scaling_events', 0):.2f} | "
                    f"{metrics.get('total_scaling_events', 0)} | "
                    f"{metrics.get('mean_volatility', 0):.4f} | "
                    f"{metrics.get('max_volatility', 0):.4f} |\n"
                )
        else:
            report += "No stability data available.\n\n"

        return report

    def save_report(self, service_name: str, report_content: str, format: str = "markdown"):

        reports_dir = self.results_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        extension = "md" if format == "markdown" else format
        report_file = reports_dir / f"{service_name}_comparative_report.{extension}"

        with open(report_file, "w") as f:
            f.write(report_content)

        logger.info(f"Report saved to {report_file}")
        return report_file
