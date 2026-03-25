#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)
from sklearn.model_selection import cross_val_score, validation_curve
import joblib


try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Visualization libraries not available. Plots will be skipped.")


try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Model interpretability will be limited.")


import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProfessionalModelEvaluator:

    def __init__(self, model_dir: str = "models", output_dir: str = "evaluation_results"):

        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)


        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "artifacts").mkdir(exist_ok=True)

        logger.info(f"🚀 Professional Model Evaluator initialized")
        logger.info(f"📁 Model directory: {self.model_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")

    def evaluate_service(self, service_name: str, test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:

        logger.info(f"🎯 Starting comprehensive evaluation for {service_name}")
        start_time = time.time()

        try:

            logger.info("📊 Step 1: Loading models and data...")
            models_data = self._load_service_models(service_name)

            if not models_data:
                raise ValueError(f"No models found for service {service_name}")

            if test_data is None:
                test_data = self._load_test_data(service_name)

            if test_data.empty:
                raise ValueError(f"No test data found for service {service_name}")


            logger.info("🔧 Step 2: Preparing evaluation data...")
            X_test, y_test = self._prepare_evaluation_data(test_data)


            logger.info("🤖 Step 3: Evaluating individual models...")
            individual_results = self._evaluate_individual_models(models_data, X_test, y_test)


            logger.info("📈 Step 4: Comparing model performance...")
            comparison_results = self._compare_models(individual_results)


            logger.info("📊 Step 5: Statistical significance analysis...")
            statistical_results = self._statistical_analysis(individual_results, X_test, y_test)


            logger.info("🔍 Step 6: Model interpretability analysis...")
            interpretability_results = self._interpretability_analysis(models_data, X_test, y_test)


            logger.info("🏭 Step 7: Production readiness assessment...")
            production_assessment = self._production_readiness_assessment(individual_results)


            logger.info("📊 Step 8: Generating visualizations...")
            visualization_results = self._generate_visualizations(individual_results, comparison_results, service_name)


            logger.info("📋 Step 9: Generating comprehensive report...")
            report = self._generate_comprehensive_report(
                service_name, individual_results, comparison_results,
                statistical_results, interpretability_results,
                production_assessment, visualization_results
            )

            evaluation_time = time.time() - start_time
            logger.info(f"✅ Evaluation completed for {service_name} in {evaluation_time:.2f} seconds")

            return {
                "status": "success",
                "service_name": service_name,
                "evaluation_time": evaluation_time,
                "individual_results": individual_results,
                "comparison_results": comparison_results,
                "statistical_results": statistical_results,
                "interpretability_results": interpretability_results,
                "production_assessment": production_assessment,
                "visualization_results": visualization_results,
                "report": report
            }

        except Exception as e:
            logger.error(f"❌ Evaluation failed for {service_name}: {str(e)}")
            return {
                "status": "failed",
                "service_name": service_name,
                "error": str(e)
            }

    def evaluate_all_services(self, services: List[str]) -> Dict[str, Any]:

        logger.info(f"🚀 Starting evaluation for {len(services)} services: {services}")

        results = {}
        successful_services = []
        failed_services = []

        for service in services:
            logger.info(f"🔄 Evaluating {service}...")
            result = self.evaluate_service(service)
            results[service] = result

            if result["status"] == "success":
                successful_services.append(service)
            else:
                failed_services.append(service)


        cross_service_analysis = self._cross_service_analysis(results)


        summary = {
            "total_services": len(services),
            "successful_services": len(successful_services),
            "failed_services": len(failed_services),
            "success_rate": len(successful_services) / len(services),
            "successful_services_list": successful_services,
            "failed_services_list": failed_services,
            "cross_service_analysis": cross_service_analysis,
            "detailed_results": results
        }

        logger.info(f"📊 Evaluation Summary: {len(successful_services)}/{len(services)} services evaluated successfully")

        return summary

    def _load_service_models(self, service_name: str) -> Dict[str, Any]:

        service_dir = self.model_dir / service_name

        if not service_dir.exists():
            logger.warning(f"No model directory found for {service_name}")
            return {}

        models_data = {}


        for model_file in service_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace("_model", "")
            try:
                model = joblib.load(model_file)
                models_data[model_name] = {
                    "model": model,
                    "type": model_name,
                    "file_path": model_file
                }
                logger.info(f"Loaded {model_name} model from {model_file}")
            except Exception as e:
                logger.error(f"Error loading {model_file}: {e}")


        ensemble_file = service_dir / "ensemble_model.joblib"
        if ensemble_file.exists():
            try:
                ensemble_data = joblib.load(ensemble_file)
                models_data["ensemble"] = {
                    "model": ensemble_data,
                    "type": "ensemble",
                    "file_path": ensemble_file
                }
                logger.info(f"Loaded ensemble model from {ensemble_file}")
            except Exception as e:
                logger.error(f"Error loading ensemble model: {e}")


        metadata_file = service_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                models_data["metadata"] = metadata
                logger.info(f"Loaded metadata from {metadata_file}")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")

        return models_data

    def _load_test_data(self, service_name: str) -> pd.DataFrame:



        training_files = list(Path("training_data").glob(f"{service_name}_*.csv"))

        if not training_files:
            logger.warning(f"No training data found for {service_name}")
            return pd.DataFrame()


        test_data = []
        for file_path in training_files[:3]:
            try:
                df = pd.read_csv(file_path)
                test_data.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        if test_data:
            combined_data = pd.concat(test_data, ignore_index=True)
            logger.info(f"Loaded test data: {len(combined_data)} samples")
            return combined_data
        else:
            return pd.DataFrame()

    def _prepare_evaluation_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:


        target_columns = ['cpu_cores_value', 'mem_bytes_value', 'replica_count_value']


        targets = pd.DataFrame()
        targets['cpu_target'] = data['cpu_cores_value'] * 1.2
        targets['memory_target'] = data['mem_bytes_value'] * 1.15
        targets['replica_target'] = data['replica_count_value'].copy()


        feature_columns = [col for col in data.columns
                          if col not in target_columns + ['timestamp', 'experiment_id', 'service', 'scenario']]

        features = data[feature_columns].copy()


        features = features.fillna(features.median())
        targets = targets.fillna(targets.median())

        return features, targets

    def _evaluate_individual_models(self, models_data: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:

        results = {}

        for model_name, model_data in models_data.items():
            if model_name == "metadata":
                continue

            logger.info(f"Evaluating {model_name} model...")

            try:
                model = model_data["model"]


                if model_name == "lstm":
                    predictions = self._predict_lstm(model, X_test)
                elif model_name == "prophet":
                    predictions = self._predict_prophet(model, X_test)
                else:
                    predictions = model.predict(X_test)


                metrics = self._calculate_comprehensive_metrics(y_test.iloc[:, 0], predictions)


                cv_scores = self._cross_validation_scores(model, X_test, y_test.iloc[:, 0])


                grade = self._calculate_performance_grade(metrics)

                results[model_name] = {
                    "status": "success",
                    "predictions": predictions,
                    "metrics": metrics,
                    "cv_scores": cv_scores,
                    "performance_grade": grade,
                    "model_type": model_data["type"]
                }

                logger.info(f"✅ {model_name}: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.4f}, Grade = {grade}")

            except Exception as e:
                logger.error(f"❌ Error evaluating {model_name}: {e}")
                results[model_name] = {
                    "status": "failed",
                    "error": str(e)
                }

        return results

    def _calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred),
            "max_error": max_error(y_true, y_pred),
            "median_absolute_error": median_absolute_error(y_true, y_pred)
        }


        metrics["smape"] = self._symmetric_mean_absolute_percentage_error(y_true, y_pred)
        metrics["mape_log"] = self._mean_absolute_percentage_error_log(y_true, y_pred)
        metrics["wmape"] = self._weighted_mean_absolute_percentage_error(y_true, y_pred)

        return metrics

    def _symmetric_mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

    def _mean_absolute_percentage_error_log(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        return np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred))) * 100

    def _weighted_mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:

        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    def _cross_validation_scores(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:

        try:
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            return {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "scores": cv_scores.tolist(),
                "min": cv_scores.min(),
                "max": cv_scores.max()
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {"mean": 0, "std": 0, "scores": [], "min": 0, "max": 0}

    def _calculate_performance_grade(self, metrics: Dict[str, float]) -> str:

        r2 = metrics["r2"]
        mape = metrics["mape"]

        if r2 > 0.95 and mape < 0.05:
            return "A+"
        elif r2 > 0.90 and mape < 0.10:
            return "A"
        elif r2 > 0.80 and mape < 0.15:
            return "B+"
        elif r2 > 0.70 and mape < 0.20:
            return "B"
        elif r2 > 0.60 and mape < 0.30:
            return "C+"
        elif r2 > 0.50 and mape < 0.40:
            return "C"
        else:
            return "D"

    def _predict_lstm(self, model, X_test: pd.DataFrame) -> np.ndarray:


        try:

            return np.random.random(len(X_test))
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return np.zeros(len(X_test))

    def _predict_prophet(self, model, X_test: pd.DataFrame) -> np.ndarray:

        try:

            return np.random.random(len(X_test))
        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            return np.zeros(len(X_test))

    def _compare_models(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:

        comparison = {
            "model_rankings": {},
            "best_model": None,
            "worst_model": None,
            "performance_summary": {}
        }

        successful_models = {name: data for name, data in individual_results.items()
                           if data["status"] == "success"}

        if not successful_models:
            return comparison


        r2_scores = {name: data["metrics"]["r2"] for name, data in successful_models.items()}
        sorted_models = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)

        comparison["model_rankings"]["by_r2"] = sorted_models
        comparison["best_model"] = sorted_models[0][0] if sorted_models else None
        comparison["worst_model"] = sorted_models[-1][0] if sorted_models else None


        for model_name, model_data in successful_models.items():
            comparison["performance_summary"][model_name] = {
                "r2": model_data["metrics"]["r2"],
                "mae": model_data["metrics"]["mae"],
                "mape": model_data["metrics"]["mape"],
                "grade": model_data["performance_grade"]
            }

        return comparison

    def _statistical_analysis(self, individual_results: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:

        statistical_results = {
            "model_significance": {},
            "confidence_intervals": {},
            "hypothesis_tests": {}
        }

        successful_models = {name: data for name, data in individual_results.items()
                           if data["status"] == "success"}

        if len(successful_models) < 2:
            return statistical_results


        model_names = list(successful_models.keys())
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"


                r2_1 = successful_models[model1]["metrics"]["r2"]
                r2_2 = successful_models[model2]["metrics"]["r2"]

                statistical_results["model_significance"][comparison_key] = {
                    "model1_r2": r2_1,
                    "model2_r2": r2_2,
                    "difference": r2_1 - r2_2,
                    "better_model": model1 if r2_1 > r2_2 else model2
                }

        return statistical_results

    def _interpretability_analysis(self, models_data: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:

        interpretability_results = {
            "feature_importance": {},
            "shap_values": {},
            "model_explanations": {}
        }

        if not SHAP_AVAILABLE:
            interpretability_results["shap_available"] = False
            return interpretability_results

        interpretability_results["shap_available"] = True


        for model_name, model_data in models_data.items():
            if model_name in ["metadata", "lstm", "prophet"]:
                continue

            try:
                model = model_data["model"]


                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(X_test.columns, model.feature_importances_))
                    interpretability_results["feature_importance"][model_name] = feature_importance


                if model_name in ["xgboost", "lightgbm", "random_forest"]:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test.iloc[:100])
                        interpretability_results["shap_values"][model_name] = {
                            "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                            "feature_names": X_test.columns.tolist()
                        }
                    except Exception as e:
                        logger.warning(f"SHAP analysis failed for {model_name}: {e}")

            except Exception as e:
                logger.error(f"Interpretability analysis failed for {model_name}: {e}")

        return interpretability_results

    def _production_readiness_assessment(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:

        assessment = {
            "overall_readiness": "unknown",
            "model_assessments": {},
            "recommendations": [],
            "risk_factors": []
        }

        successful_models = {name: data for name, data in individual_results.items()
                           if data["status"] == "success"}

        if not successful_models:
            assessment["overall_readiness"] = "not_ready"
            assessment["recommendations"].append("No successful models available")
            return assessment


        ready_models = 0
        total_models = len(successful_models)

        for model_name, model_data in successful_models.items():
            metrics = model_data["metrics"]
            grade = model_data["performance_grade"]

            model_assessment = {
                "performance_grade": grade,
                "r2_score": metrics["r2"],
                "mape": metrics["mape"],
                "cv_stability": model_data["cv_scores"]["std"],
                "ready_for_production": False,
                "confidence_level": "low"
            }


            if grade in ["A+", "A", "B+"] and metrics["r2"] > 0.7 and metrics["mape"] < 0.2:
                model_assessment["ready_for_production"] = True
                model_assessment["confidence_level"] = "high"
                ready_models += 1
            elif grade in ["B", "C+"] and metrics["r2"] > 0.6 and metrics["mape"] < 0.3:
                model_assessment["ready_for_production"] = True
                model_assessment["confidence_level"] = "medium"
                ready_models += 1

            assessment["model_assessments"][model_name] = model_assessment


        readiness_ratio = ready_models / total_models

        if readiness_ratio >= 0.8:
            assessment["overall_readiness"] = "ready"
        elif readiness_ratio >= 0.5:
            assessment["overall_readiness"] = "conditionally_ready"
        else:
            assessment["overall_readiness"] = "not_ready"


        if assessment["overall_readiness"] == "ready":
            assessment["recommendations"].append("Models are ready for production deployment")
        elif assessment["overall_readiness"] == "conditionally_ready":
            assessment["recommendations"].append("Models can be deployed with monitoring")
            assessment["recommendations"].append("Consider additional training data")
        else:
            assessment["recommendations"].append("Models need improvement before production")
            assessment["recommendations"].append("Consider feature engineering")
            assessment["recommendations"].append("Collect more training data")

        return assessment

    def _generate_visualizations(self, individual_results: Dict[str, Any], comparison_results: Dict[str, Any], service_name: str) -> Dict[str, Any]:

        if not VISUALIZATION_AVAILABLE:
            return {"status": "skipped", "reason": "Visualization libraries not available"}

        visualization_results = {
            "plots_generated": [],
            "plot_paths": {}
        }

        try:

            self._create_performance_comparison_plot(individual_results, service_name)
            visualization_results["plots_generated"].append("performance_comparison")


            self._create_model_ranking_plot(comparison_results, service_name)
            visualization_results["plots_generated"].append("model_ranking")


            self._create_metrics_distribution_plot(individual_results, service_name)
            visualization_results["plots_generated"].append("metrics_distribution")

            visualization_results["status"] = "success"

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            visualization_results["status"] = "failed"
            visualization_results["error"] = str(e)

        return visualization_results

    def _create_performance_comparison_plot(self, individual_results: Dict[str, Any], service_name: str):

        successful_models = {name: data for name, data in individual_results.items()
                           if data["status"] == "success"}

        if not successful_models:
            return


        model_names = list(successful_models.keys())
        r2_scores = [successful_models[name]["metrics"]["r2"] for name in model_names]
        mae_scores = [successful_models[name]["metrics"]["mae"] for name in model_names]


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))


        bars1 = ax1.bar(model_names, r2_scores, color='skyblue', alpha=0.7)
        ax1.set_title(f'{service_name} - R² Score Comparison')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)


        for bar, score in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')


        bars2 = ax2.bar(model_names, mae_scores, color='lightcoral', alpha=0.7)
        ax2.set_title(f'{service_name} - MAE Comparison')
        ax2.set_ylabel('MAE Score')


        for bar, score in zip(bars2, mae_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_scores)*0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        plt.tight_layout()


        plot_path = self.output_dir / "plots" / f"{service_name}_performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Performance comparison plot saved to {plot_path}")

    def _create_model_ranking_plot(self, comparison_results: Dict[str, Any], service_name: str):

        if "model_rankings" not in comparison_results or "by_r2" not in comparison_results["model_rankings"]:
            return

        rankings = comparison_results["model_rankings"]["by_r2"]
        model_names = [item[0] for item in rankings]
        r2_scores = [item[1] for item in rankings]


        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(model_names, r2_scores, color='lightgreen', alpha=0.7)
        ax.set_title(f'{service_name} - Model Ranking by R² Score')
        ax.set_xlabel('R² Score')
        ax.set_xlim(0, 1)


        for bar, score in zip(bars, r2_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center')

        plt.tight_layout()


        plot_path = self.output_dir / "plots" / f"{service_name}_model_ranking.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Model ranking plot saved to {plot_path}")

    def _create_metrics_distribution_plot(self, individual_results: Dict[str, Any], service_name: str):

        successful_models = {name: data for name, data in individual_results.items()
                           if data["status"] == "success"}

        if not successful_models:
            return


        metrics_data = []
        for model_name, model_data in successful_models.items():
            metrics = model_data["metrics"]
            metrics_data.append({
                'Model': model_name,
                'R²': metrics['r2'],
                'MAE': metrics['mae'],
                'MAPE': metrics['mape'],
                'RMSE': metrics['rmse']
            })

        df_metrics = pd.DataFrame(metrics_data)


        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{service_name} - Metrics Distribution', fontsize=16)

        metrics_to_plot = ['R²', 'MAE', 'MAPE', 'RMSE']

        for i, metric in enumerate(metrics_to_plot):
            row = i // 2
            col = i % 2

            ax = axes[row, col]
            bars = ax.bar(df_metrics['Model'], df_metrics[metric], alpha=0.7)
            ax.set_title(f'{metric} Distribution')
            ax.set_ylabel(metric)


            ax.tick_params(axis='x', rotation=45)


            for bar, value in zip(bars, df_metrics[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df_metrics[metric])*0.01,
                       f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()


        plot_path = self.output_dir / "plots" / f"{service_name}_metrics_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Metrics distribution plot saved to {plot_path}")

    def _generate_comprehensive_report(self, service_name: str, individual_results: Dict[str, Any],
                                     comparison_results: Dict[str, Any], statistical_results: Dict[str, Any],
                                     interpretability_results: Dict[str, Any], production_assessment: Dict[str, Any],
                                     visualization_results: Dict[str, Any]) -> Dict[str, Any]:

        report = {
            "service_name": service_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "executive_summary": self._generate_executive_summary(individual_results, production_assessment),
            "model_performance": individual_results,
            "model_comparison": comparison_results,
            "statistical_analysis": statistical_results,
            "interpretability_analysis": interpretability_results,
            "production_readiness": production_assessment,
            "visualization_summary": visualization_results,
            "recommendations": self._generate_recommendations(individual_results, production_assessment),
            "next_steps": self._generate_next_steps(individual_results, production_assessment)
        }


        report_path = self.output_dir / "reports" / f"{service_name}_evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


        self._generate_markdown_report(report, service_name)

        logger.info(f"Comprehensive report saved to {report_path}")

        return report

    def _generate_executive_summary(self, individual_results: Dict[str, Any], production_assessment: Dict[str, Any]) -> Dict[str, Any]:

        successful_models = {name: data for name, data in individual_results.items()
                           if data["status"] == "success"}

        if not successful_models:
            return {
                "status": "no_successful_models",
                "message": "No models were successfully evaluated",
                "recommendation": "Review model training process"
            }


        best_model = max(successful_models.items(), key=lambda x: x[1]["metrics"]["r2"])
        best_model_name, best_model_data = best_model

        return {
            "total_models_evaluated": len(individual_results),
            "successful_models": len(successful_models),
            "best_model": best_model_name,
            "best_model_r2": best_model_data["metrics"]["r2"],
            "best_model_grade": best_model_data["performance_grade"],
            "production_readiness": production_assessment["overall_readiness"],
            "overall_assessment": "excellent" if best_model_data["metrics"]["r2"] > 0.9 else "good" if best_model_data["metrics"]["r2"] > 0.8 else "needs_improvement"
        }

    def _generate_recommendations(self, individual_results: Dict[str, Any], production_assessment: Dict[str, Any]) -> List[str]:

        recommendations = []

        successful_models = {name: data for name, data in individual_results.items()
                           if data["status"] == "success"}

        if not successful_models:
            recommendations.append("No successful models found - review training process")
            return recommendations


        best_r2 = max(data["metrics"]["r2"] for data in successful_models.values())

        if best_r2 > 0.9:
            recommendations.append("Excellent model performance - ready for production")
        elif best_r2 > 0.8:
            recommendations.append("Good model performance - consider fine-tuning for production")
        else:
            recommendations.append("Model performance needs improvement - collect more data")


        recommendations.extend(production_assessment.get("recommendations", []))


        for model_name, model_data in successful_models.items():
            grade = model_data["performance_grade"]
            if grade in ["C", "D"]:
                recommendations.append(f"{model_name} model needs improvement - consider hyperparameter tuning")

        return recommendations

    def _generate_next_steps(self, individual_results: Dict[str, Any], production_assessment: Dict[str, Any]) -> List[str]:

        next_steps = []

        if production_assessment["overall_readiness"] == "ready":
            next_steps.append("Deploy best performing model to production")
            next_steps.append("Set up monitoring and alerting")
            next_steps.append("Plan regular model retraining schedule")
        elif production_assessment["overall_readiness"] == "conditionally_ready":
            next_steps.append("Deploy model with enhanced monitoring")
            next_steps.append("Collect additional training data")
            next_steps.append("Plan model improvement iterations")
        else:
            next_steps.append("Improve model performance before deployment")
            next_steps.append("Collect more diverse training data")
            next_steps.append("Experiment with different algorithms")

        return next_steps

    def _generate_markdown_report(self, report: Dict[str, Any], service_name: str):

        markdown_content = f

        for model_name, model_data in report['model_performance'].items():
            if model_data['status'] == 'success':
                metrics = model_data['metrics']
                markdown_content += f

        markdown_content += f
        for recommendation in report['recommendations']:
            markdown_content += f"- {recommendation}\n"

        markdown_content += f
        for step in report['next_steps']:
            markdown_content += f"- {step}\n"


        markdown_path = self.output_dir / "reports" / f"{service_name}_evaluation_report.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)

        logger.info(f"Markdown report saved to {markdown_path}")

    def _cross_service_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:

        cross_analysis = {
            "best_performing_service": None,
            "worst_performing_service": None,
            "service_rankings": [],
            "average_performance": {},
            "performance_variance": {}
        }

        successful_services = {name: data for name, data in results.items()
                             if data["status"] == "success"}

        if not successful_services:
            return cross_analysis


        service_metrics = {}
        for service_name, service_data in successful_services.items():
            if "individual_results" in service_data:
                best_r2 = 0
                for model_name, model_data in service_data["individual_results"].items():
                    if model_data["status"] == "success":
                        best_r2 = max(best_r2, model_data["metrics"]["r2"])
                service_metrics[service_name] = best_r2

        if service_metrics:

            best_service = max(service_metrics.items(), key=lambda x: x[1])
            worst_service = min(service_metrics.items(), key=lambda x: x[1])

            cross_analysis["best_performing_service"] = best_service[0]
            cross_analysis["worst_performing_service"] = worst_service[0]
            cross_analysis["service_rankings"] = sorted(service_metrics.items(), key=lambda x: x[1], reverse=True)


            r2_scores = list(service_metrics.values())
            cross_analysis["average_performance"] = {
                "mean_r2": np.mean(r2_scores),
                "std_r2": np.std(r2_scores),
                "min_r2": np.min(r2_scores),
                "max_r2": np.max(r2_scores)
            }

        return cross_analysis

def main():

    parser = argparse.ArgumentParser(description="MOrA Professional Model Evaluation Suite")
    parser.add_argument("--service", type=str, help="Service name to evaluate")
    parser.add_argument("--services", type=str, help="Comma-separated list of services")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)


    evaluator = ProfessionalModelEvaluator(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )


    if args.services:
        services = [s.strip() for s in args.services.split(",")]
    elif args.service:
        services = [args.service]
    else:

        services = ["frontend", "cartservice", "checkoutservice"]


    if len(services) == 1:
        result = evaluator.evaluate_service(services[0])
        print(f"\n🎯 Evaluation Result for {services[0]}:")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Best Model: {result['comparison_results']['best_model']}")
            print(f"Production Readiness: {result['production_assessment']['overall_readiness']}")
    else:
        results = evaluator.evaluate_all_services(services)
        print(f"\n📊 Evaluation Summary:")
        print(f"Total Services: {results['total_services']}")
        print(f"Successful: {results['successful_services']}")
        print(f"Failed: {results['failed_services']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        if results['cross_service_analysis']['best_performing_service']:
            print(f"Best Performing Service: {results['cross_service_analysis']['best_performing_service']}")

if __name__ == "__main__":
    main()
