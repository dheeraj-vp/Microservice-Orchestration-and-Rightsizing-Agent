#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import warnings
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import traceback


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib


try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.utils import plot_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available. LSTM models will be skipped.")


try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not available. Prophet models will be skipped.")


try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    print("⚠️  Advanced ML libraries not available. Using basic models only.")


try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Visualization libraries not available. Plots will be skipped.")


warnings.filterwarnings('ignore')
if TENSORFLOW_AVAILABLE:
    tf.get_logger().setLevel('ERROR')

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProfessionalMLPipeline:

    def __init__(self,
                 data_dir: str = "training_data",
                 model_dir: str = "models",
                 config: Optional[Dict[str, Any]] = None):

        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)


        self.config = self._load_config(config)


        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        self.validation_results = {}


        self.feature_engineering = FeatureEngineering(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.validator = ModelValidator(self.config)
        self.ensemble = ModelEnsemble(self.config)

        logger.info(f"🚀 Professional ML Pipeline initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📁 Model directory: {self.model_dir}")

    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:

        default_config = {

            "data": {
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_state": 42,
                "time_series_split": 5,
                "feature_scaling": "robust",
                "handle_outliers": True,
                "outlier_threshold": 3.0
            },


            "features": {
                "lag_features": [1, 2, 3, 5, 10],
                "rolling_windows": [3, 5, 10, 15],
                "statistical_features": ["mean", "std", "min", "max", "median"],
                "interaction_features": True,
                "polynomial_features": 2,
                "feature_selection": True,
                "max_features": 50
            },


            "lstm": {
                "sequence_length": 10,
                "hidden_units": [32, 16],
                "dropout_rate": 0.2,
                "recurrent_dropout": 0.2,
                "epochs": 20,
                "batch_size": 32,
                "validation_split": 0.2,
                "early_stopping_patience": 5,
                "learning_rate": 0.001,
                "optimizer": "adam"
            },


            "prophet": {
                "yearly_seasonality": False,
                "weekly_seasonality": True,
                "daily_seasonality": True,
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "interval_width": 0.90,
                "uncertainty_samples": 1000
            },


            "xgboost": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "early_stopping_rounds": 10
            },

            "lightgbm": {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "early_stopping_rounds": 10
            },

            "random_forest": {
                "n_estimators": 100,
                "max_depth": 6,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1
            },


            "ensemble": {
                "methods": ["voting", "stacking", "blending"],
                "weights": {"lstm": 0.3, "prophet": 0.2, "xgboost": 0.25, "lightgbm": 0.25},
                "stacking_cv": 5,
                "blending_ratio": 0.7
            },


            "validation": {
                "cv_folds": 5,
                "scoring_metrics": ["mse", "mae", "r2", "mape"],
                "time_series_cv": True,
                "walk_forward_validation": True
            },


            "production": {
                "model_versioning": True,
                "model_monitoring": True,
                "performance_thresholds": {
                    "mse": 0.1,
                    "mae": 0.05,
                    "r2": 0.8,
                    "mape": 0.1
                },
                "save_artifacts": True,
                "generate_report": True
            }
        }

        if config:
            default_config.update(config)

        return default_config

    def train_service(self, service_name: str) -> Dict[str, Any]:

        logger.info(f"🎯 Starting comprehensive training for {service_name}")
        start_time = time.time()

        try:

            logger.info("📊 Step 1: Loading and preparing data...")
            data = self._load_service_data(service_name)

            if data.empty:
                raise ValueError(f"No data found for service {service_name}")


            logger.info("🔧 Step 2: Advanced feature engineering...")
            X, y = self.feature_engineering.process_data(data, service_name)


            logger.info("✂️  Step 3: Splitting data for training and validation...")
            X_train, X_test, y_train, y_test = self._split_data(X, y)


            logger.info("🤖 Step 4: Training multiple ML models...")
            models_results = self._train_multiple_models(X_train, X_test, y_train, y_test, service_name)


            logger.info("✅ Step 5: Comprehensive model validation...")
            validation_results = self.validator.validate_models(models_results, X_test, y_test)


            logger.info("🎭 Step 6: Creating model ensemble...")
            ensemble_results = self.ensemble.create_ensemble(models_results, X_test, y_test)


            logger.info("📈 Step 7: Final model evaluation...")
            final_results = self._evaluate_final_models(ensemble_results, X_test, y_test)


            logger.info("💾 Step 8: Saving models and artifacts...")
            self._save_models_and_artifacts(service_name, models_results, ensemble_results, final_results)


            logger.info("📋 Step 9: Generating comprehensive report...")
            report = self._generate_training_report(service_name, final_results, validation_results)

            training_time = time.time() - start_time
            logger.info(f"✅ Training completed for {service_name} in {training_time:.2f} seconds")

            return {
                "status": "success",
                "service_name": service_name,
                "training_time": training_time,
                "models_trained": len(models_results),
                "best_model": final_results["best_model"],
                "performance": final_results["performance"],
                "validation_results": validation_results,
                "ensemble_results": ensemble_results,
                "report": report
            }

        except Exception as e:
            logger.error(f"❌ Training failed for {service_name}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "failed",
                "service_name": service_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def train_all_services(self, services: List[str]) -> Dict[str, Any]:

        logger.info(f"🚀 Starting training for {len(services)} services: {services}")

        results = {}
        successful_services = []
        failed_services = []

        for service in services:
            logger.info(f"🔄 Training {service}...")
            result = self.train_service(service)
            results[service] = result

            if result["status"] == "success":
                successful_services.append(service)
            else:
                failed_services.append(service)


        summary = {
            "total_services": len(services),
            "successful_services": len(successful_services),
            "failed_services": len(failed_services),
            "success_rate": len(successful_services) / len(services),
            "successful_services_list": successful_services,
            "failed_services_list": failed_services,
            "detailed_results": results
        }

        logger.info(f"📊 Training Summary: {len(successful_services)}/{len(services)} services trained successfully")

        return summary

    def _load_service_data(self, service_name: str) -> pd.DataFrame:

        csv_files = list(self.data_dir.glob(f"{service_name}_*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found for {service_name}")
            return pd.DataFrame()

        logger.info(f"Found {len(csv_files)} data files for {service_name}")

        combined_data = []
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                combined_data.append(df)
                logger.debug(f"Loaded {len(df)} rows from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined data: {len(combined_df)} total rows")
            return combined_df
        else:
            return pd.DataFrame()

    def _split_data(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if self.config["validation"]["time_series_cv"]:

            split_point = int(len(X) * (1 - self.config["data"]["test_size"]))
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
        else:

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config["data"]["test_size"],
                random_state=self.config["data"]["random_state"]
            )

        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    def _train_multiple_models(self, X_train, X_test, y_train, y_test, service_name):

        models_results = {}


        if TENSORFLOW_AVAILABLE:
            logger.info("🧠 Training LSTM model...")
            lstm_result = self.model_trainer.train_lstm(X_train, X_test, y_train, y_test, service_name)
            models_results["lstm"] = lstm_result


        if PROPHET_AVAILABLE:
            logger.info("📈 Training Prophet model...")
            prophet_result = self.model_trainer.train_prophet(X_train, X_test, y_train, y_test, service_name)
            models_results["prophet"] = prophet_result


        if ADVANCED_ML_AVAILABLE:
            logger.info("🚀 Training XGBoost model...")
            xgb_result = self.model_trainer.train_xgboost(X_train, X_test, y_train, y_test, service_name)
            models_results["xgboost"] = xgb_result

            logger.info("💡 Training LightGBM model...")
            lgb_result = self.model_trainer.train_lightgbm(X_train, X_test, y_train, y_test, service_name)
            models_results["lightgbm"] = lgb_result


        logger.info("🌲 Training RandomForest model...")
        rf_result = self.model_trainer.train_random_forest(X_train, X_test, y_train, y_test, service_name)
        models_results["random_forest"] = rf_result

        return models_results

    def _evaluate_final_models(self, ensemble_results, X_test, y_test):

        best_model = None
        best_score = float('inf')

        for model_name, model_data in ensemble_results.items():
            if model_data["status"] == "success":
                score = model_data["performance"]["mse"]
                if score < best_score:
                    best_score = score
                    best_model = model_name

        return {
            "best_model": best_model,
            "best_score": best_score,
            "performance": ensemble_results[best_model]["performance"] if best_model else {}
        }

    def _save_models_and_artifacts(self, service_name, models_results, ensemble_results, final_results):

        service_dir = self.model_dir / service_name
        service_dir.mkdir(exist_ok=True)


        for model_name, model_data in models_results.items():
            if model_data["status"] == "success":
                model_path = service_dir / f"{model_name}_model.joblib"
                joblib.dump(model_data["model"], model_path)
                logger.info(f"Saved {model_name} model to {model_path}")


        ensemble_path = service_dir / "ensemble_model.joblib"
        joblib.dump(ensemble_results, ensemble_path)


        metadata = {
            "service_name": service_name,
            "trained_at": datetime.now().isoformat(),
            "config": self.config,
            "final_results": final_results,
            "model_count": len(models_results)
        }

        metadata_path = service_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _generate_training_report(self, service_name, final_results, validation_results):

        report = {
            "service_name": service_name,
            "timestamp": datetime.now().isoformat(),
            "best_model": final_results["best_model"],
            "performance_metrics": final_results["performance"],
            "validation_summary": validation_results,
            "model_comparison": self._compare_models(validation_results),
            "recommendations": self._generate_recommendations(final_results)
        }


        report_path = self.model_dir / service_name / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _compare_models(self, validation_results):

        comparison = {}
        for model_name, results in validation_results.items():
            if results["status"] == "success":
                comparison[model_name] = {
                    "mse": results["mse"],
                    "mae": results["mae"],
                    "r2": results["r2"],
                    "mape": results["mape"]
                }
        return comparison

    def _generate_recommendations(self, final_results):

        recommendations = []

        if final_results["performance"]["r2"] > 0.9:
            recommendations.append("Excellent model performance - ready for production")
        elif final_results["performance"]["r2"] > 0.8:
            recommendations.append("Good model performance - consider fine-tuning")
        else:
            recommendations.append("Model needs improvement - consider more data or feature engineering")

        if final_results["performance"]["mape"] < 0.1:
            recommendations.append("Low prediction error - high confidence in recommendations")

        return recommendations

class FeatureEngineering:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}

    def process_data(self, data: pd.DataFrame, service_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

        logger.info(f"🔧 Processing features for {service_name}")


        data = self._create_time_features(data)


        data = self._create_lag_features(data)


        data = self._create_rolling_features(data)


        data = self._create_statistical_features(data)


        if self.config["features"]["interaction_features"]:
            data = self._create_interaction_features(data)


        if self.config["data"]["handle_outliers"]:
            data = self._handle_outliers(data)


        X, y = self._prepare_features_and_targets(data)


        X = self._scale_features(X, service_name)

        logger.info(f"✅ Feature engineering complete: {X.shape[1]} features, {X.shape[0]} samples")

        return X, y

    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:

        if 'timestamp' in data.columns:

            if data['timestamp'].dtype == 'object' and data['timestamp'].str.startswith('t_').all():

                data['time_index'] = data['timestamp'].str.extract(r't_(\d+)').astype(int)
                data['hour'] = data['time_index'] % 24
                data['day_of_week'] = (data['time_index'] // 24) % 7
                data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
            else:

                try:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data['hour'] = data['timestamp'].dt.hour
                    data['day_of_week'] = data['timestamp'].dt.dayofweek
                    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
                except Exception as e:
                    logger.warning(f"Could not parse timestamps: {e}. Skipping time features.")

        return data

    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for lag in self.config["features"]["lag_features"]:
            for col in numeric_columns:
                if col not in ['replica_count', 'load_users']:
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)

        return data

    def _create_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for window in self.config["features"]["rolling_windows"]:
            for col in numeric_columns:
                if col not in ['replica_count', 'load_users']:
                    for stat in self.config["features"]["statistical_features"]:
                        if stat == "mean":
                            data[f'{col}_rolling_{window}_mean'] = data[col].rolling(window).mean()
                        elif stat == "std":
                            data[f'{col}_rolling_{window}_std'] = data[col].rolling(window).std()
                        elif stat == "min":
                            data[f'{col}_rolling_{window}_min'] = data[col].rolling(window).min()
                        elif stat == "max":
                            data[f'{col}_rolling_{window}_max'] = data[col].rolling(window).max()
                        elif stat == "median":
                            data[f'{col}_rolling_{window}_median'] = data[col].rolling(window).median()

        return data

    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col not in ['replica_count', 'load_users']:
                data[f'{col}_zscore'] = (data[col] - data[col].mean()) / data[col].std()
                data[f'{col}_percentile'] = data[col].rank(pct=True)

        return data

    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:

        numeric_columns = data.select_dtypes(include=[np.number]).columns


        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                if col1 not in ['replica_count', 'load_users'] and col2 not in ['replica_count', 'load_users']:
                    data[f'{col1}_x_{col2}'] = data[col1] * data[col2]

        return data

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col not in ['replica_count', 'load_users']:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config["data"]["outlier_threshold"] * IQR
                upper_bound = Q3 + self.config["data"]["outlier_threshold"] * IQR

                data[col] = data[col].clip(lower_bound, upper_bound)

        return data

    def _prepare_features_and_targets(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:


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

    def _scale_features(self, X: pd.DataFrame, service_name: str) -> pd.DataFrame:

        scaler_type = self.config["data"]["feature_scaling"]

        if scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )


        self.scalers[service_name] = scaler

        return X_scaled

class ModelTrainer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def train_lstm(self, X_train, X_test, y_train, y_test, service_name):

        if not TENSORFLOW_AVAILABLE:
            return {"status": "skipped", "reason": "TensorFlow not available"}

        try:

            sequence_length = self.config["lstm"]["sequence_length"]
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)


            model = self._build_lstm_model(X_train_seq.shape[2])


            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config["lstm"]["learning_rate"]),
                loss='mse',
                metrics=['mae']
            )


            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config["lstm"]["early_stopping_patience"],
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7
                )
            ]


            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=self.config["lstm"]["epochs"],
                batch_size=self.config["lstm"]["batch_size"],
                validation_split=self.config["lstm"]["validation_split"],
                callbacks=callbacks_list,
                verbose=0
            )


            predictions = model.predict(X_test_seq, verbose=0)
            mse = mean_squared_error(y_test_seq, predictions)
            mae = mean_absolute_error(y_test_seq, predictions)
            r2 = r2_score(y_test_seq, predictions)

            return {
                "status": "success",
                "model": model,
                "history": history.history,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "predictions": predictions
            }

        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return {"status": "failed", "error": str(e)}

    def train_prophet(self, X_train, X_test, y_train, y_test, service_name):

        if not PROPHET_AVAILABLE:
            return {"status": "skipped", "reason": "Prophet not available"}

        try:

            prophet_data = pd.DataFrame({
                'ds': pd.date_range(start='2024-01-01', periods=len(y_train), freq='min'),
                'y': y_train.iloc[:, 0]
            })


            model = Prophet(
                yearly_seasonality=self.config["prophet"]["yearly_seasonality"],
                weekly_seasonality=self.config["prophet"]["weekly_seasonality"],
                daily_seasonality=self.config["prophet"]["daily_seasonality"],
                changepoint_prior_scale=self.config["prophet"]["changepoint_prior_scale"],
                seasonality_prior_scale=self.config["prophet"]["seasonality_prior_scale"],
                interval_width=self.config["prophet"]["interval_width"]
            )


            model.fit(prophet_data)


            future = model.make_future_dataframe(periods=len(y_test), freq='min')
            forecast = model.predict(future)


            predictions = forecast['yhat'].iloc[-len(y_test):].values


            mse = mean_squared_error(y_test.iloc[:, 0], predictions)
            mae = mean_absolute_error(y_test.iloc[:, 0], predictions)
            r2 = r2_score(y_test.iloc[:, 0], predictions)

            return {
                "status": "success",
                "model": model,
                "forecast": forecast,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "predictions": predictions
            }

        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            return {"status": "failed", "error": str(e)}

    def train_xgboost(self, X_train, X_test, y_train, y_test, service_name):

        if not ADVANCED_ML_AVAILABLE:
            return {"status": "skipped", "reason": "XGBoost not available"}

        try:

            model = xgb.XGBRegressor(**self.config["xgboost"])
            model.fit(X_train, y_train.iloc[:, 0])


            predictions = model.predict(X_test)


            mse = mean_squared_error(y_test.iloc[:, 0], predictions)
            mae = mean_absolute_error(y_test.iloc[:, 0], predictions)
            r2 = r2_score(y_test.iloc[:, 0], predictions)


            feature_importance = dict(zip(X_train.columns, model.feature_importances_))

            return {
                "status": "success",
                "model": model,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "predictions": predictions,
                "feature_importance": feature_importance
            }

        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return {"status": "failed", "error": str(e)}

    def train_lightgbm(self, X_train, X_test, y_train, y_test, service_name):

        if not ADVANCED_ML_AVAILABLE:
            return {"status": "skipped", "reason": "LightGBM not available"}

        try:

            model = lgb.LGBMRegressor(**self.config["lightgbm"])
            model.fit(X_train, y_train.iloc[:, 0])


            predictions = model.predict(X_test)


            mse = mean_squared_error(y_test.iloc[:, 0], predictions)
            mae = mean_absolute_error(y_test.iloc[:, 0], predictions)
            r2 = r2_score(y_test.iloc[:, 0], predictions)


            feature_importance = dict(zip(X_train.columns, model.feature_importances_))

            return {
                "status": "success",
                "model": model,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "predictions": predictions,
                "feature_importance": feature_importance
            }

        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
            return {"status": "failed", "error": str(e)}

    def train_random_forest(self, X_train, X_test, y_train, y_test, service_name):

        try:

            model = RandomForestRegressor(**self.config["random_forest"])
            model.fit(X_train, y_train.iloc[:, 0])


            predictions = model.predict(X_test)


            mse = mean_squared_error(y_test.iloc[:, 0], predictions)
            mae = mean_absolute_error(y_test.iloc[:, 0], predictions)
            r2 = r2_score(y_test.iloc[:, 0], predictions)


            feature_importance = dict(zip(X_train.columns, model.feature_importances_))

            return {
                "status": "success",
                "model": model,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "predictions": predictions,
                "feature_importance": feature_importance
            }

        except Exception as e:
            logger.error(f"RandomForest training failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _create_sequences(self, X, y, sequence_length):

        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i].values[0])

        return np.array(X_seq), np.array(y_seq)

    def _build_lstm_model(self, input_dim):

        model = models.Sequential()


        model.add(layers.LSTM(
            self.config["lstm"]["hidden_units"][0],
            return_sequences=True,
            input_shape=(self.config["lstm"]["sequence_length"], input_dim),
            dropout=self.config["lstm"]["dropout_rate"],
            recurrent_dropout=self.config["lstm"]["recurrent_dropout"]
        ))


        model.add(layers.LSTM(
            self.config["lstm"]["hidden_units"][1],
            return_sequences=False,
            dropout=self.config["lstm"]["dropout_rate"],
            recurrent_dropout=self.config["lstm"]["recurrent_dropout"]
        ))


        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.config["lstm"]["dropout_rate"]))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))

        return model

class ModelValidator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def validate_models(self, models_results, X_test, y_test):

        validation_results = {}

        for model_name, model_data in models_results.items():
            if model_data["status"] == "success":
                validation_results[model_name] = self._validate_single_model(model_data, X_test, y_test)
            else:
                validation_results[model_name] = model_data

        return validation_results

    def _validate_single_model(self, model_data, X_test, y_test):

        try:
            predictions = model_data["predictions"]


            mse = mean_squared_error(y_test.iloc[:, 0], predictions)
            mae = mean_absolute_error(y_test.iloc[:, 0], predictions)
            r2 = r2_score(y_test.iloc[:, 0], predictions)
            mape = mean_absolute_percentage_error(y_test.iloc[:, 0], predictions)


            cv_scores = self._cross_validate_model(model_data["model"], X_test, y_test)

            return {
                "status": "success",
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "mape": mape,
                "cv_scores": cv_scores,
                "performance_grade": self._grade_performance(r2, mape)
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _cross_validate_model(self, model, X, y):

        try:

            if hasattr(model, 'predict'):
                cv_scores = cross_val_score(model, X, y.iloc[:, 0], cv=self.config["validation"]["cv_folds"], scoring='r2')
                return {
                    "mean": cv_scores.mean(),
                    "std": cv_scores.std(),
                    "scores": cv_scores.tolist()
                }
            else:
                return {"mean": 0, "std": 0, "scores": []}
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")
            return {"mean": 0, "std": 0, "scores": []}

    def _grade_performance(self, r2, mape):

        if r2 > 0.9 and mape < 0.1:
            return "A+"
        elif r2 > 0.8 and mape < 0.2:
            return "A"
        elif r2 > 0.7 and mape < 0.3:
            return "B"
        elif r2 > 0.6 and mape < 0.4:
            return "C"
        else:
            return "D"

class ModelEnsemble:

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def create_ensemble(self, models_results, X_test, y_test):

        ensemble_results = {}


        voting_result = self._create_voting_ensemble(models_results, X_test, y_test)
        ensemble_results["voting"] = voting_result


        stacking_result = self._create_stacking_ensemble(models_results, X_test, y_test)
        ensemble_results["stacking"] = stacking_result

        return ensemble_results

    def _create_voting_ensemble(self, models_results, X_test, y_test):

        try:
            successful_models = [data for data in models_results.values() if data["status"] == "success"]

            if len(successful_models) < 2:
                return {"status": "skipped", "reason": "Not enough successful models"}


            weights = []
            predictions = []

            for model_data in successful_models:
                if "predictions" in model_data:
                    predictions.append(model_data["predictions"])

                    weight = 1.0 / (model_data["mse"] + 1e-8)
                    weights.append(weight)

            if predictions:
                weights = np.array(weights)
                weights = weights / weights.sum()

                ensemble_predictions = np.average(predictions, axis=0, weights=weights)


                mse = mean_squared_error(y_test.iloc[:, 0], ensemble_predictions)
                mae = mean_absolute_error(y_test.iloc[:, 0], ensemble_predictions)
                r2 = r2_score(y_test.iloc[:, 0], ensemble_predictions)

                return {
                    "status": "success",
                    "method": "weighted_voting",
                    "weights": weights.tolist(),
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "predictions": ensemble_predictions,
                    "performance": {
                        "mse": mse,
                        "mae": mae,
                        "r2": r2
                    }
                }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _create_stacking_ensemble(self, models_results, X_test, y_test):

        try:
            successful_models = [data for data in models_results.values() if data["status"] == "success"]

            if len(successful_models) < 2:
                return {"status": "skipped", "reason": "Not enough successful models"}


            meta_features = []
            for model_data in successful_models:
                if "predictions" in model_data:
                    meta_features.append(model_data["predictions"])

            if meta_features:
                meta_features = np.column_stack(meta_features)
                meta_learner = LinearRegression()
                meta_learner.fit(meta_features, y_test.iloc[:, 0])

                ensemble_predictions = meta_learner.predict(meta_features)


                mse = mean_squared_error(y_test.iloc[:, 0], ensemble_predictions)
                mae = mean_absolute_error(y_test.iloc[:, 0], ensemble_predictions)
                r2 = r2_score(y_test.iloc[:, 0], ensemble_predictions)

                return {
                    "status": "success",
                    "method": "stacking",
                    "meta_learner": meta_learner,
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "predictions": ensemble_predictions,
                    "performance": {
                        "mse": mse,
                        "mae": mae,
                        "r2": r2
                    }
                }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

def main():

    parser = argparse.ArgumentParser(description="MOrA Professional ML Training Pipeline")
    parser.add_argument("--service", type=str, help="Service name to train")
    parser.add_argument("--services", type=str, help="Comma-separated list of services")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data-dir", type=str, default="training_data", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)


    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)


    pipeline = ProfessionalMLPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        config=config
    )


    if args.services:
        services = [s.strip() for s in args.services.split(",")]
    elif args.service:
        services = [args.service]
    else:

        services = ["frontend", "cartservice", "checkoutservice"]


    if len(services) == 1:
        result = pipeline.train_service(services[0])
        print(f"\n🎯 Training Result for {services[0]}:")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Best Model: {result['best_model']}")
            print(f"Performance: {result['performance']}")
    else:
        results = pipeline.train_all_services(services)
        print(f"\n📊 Training Summary:")
        print(f"Total Services: {results['total_services']}")
        print(f"Successful: {results['successful_services']}")
        print(f"Failed: {results['failed_services']}")
        print(f"Success Rate: {results['success_rate']:.2%}")

if __name__ == "__main__":
    main()
