#!/usr/bin/env python3

import os
import sys
import logging
import warnings
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import traceback


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
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


warnings.filterwarnings('ignore')
if TENSORFLOW_AVAILABLE:
    tf.get_logger().setLevel('ERROR')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_prophet_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class LightweightLSTMProphetPipeline:

    def __init__(self, data_dir: str = "training_data", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)


        self.config = {
            "lstm": {
                "sequence_length": 10,
                "hidden_units": [32, 16],
                "dropout_rate": 0.2,
                "recurrent_dropout": 0.2,
                "epochs": 20,
                "batch_size": 16,
                "validation_split": 0.2,
                "early_stopping_patience": 5,
                "learning_rate": 0.001,
                "verbose": 0
            },
            "prophet": {
                "yearly_seasonality": False,
                "weekly_seasonality": True,
                "daily_seasonality": True,
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "interval_width": 0.9
            },
            "fusion": {
                "prophet_weight": 0.4,
                "lstm_weight": 0.6,
                "base_confidence": 0.7
            }
        }


        self.targets = {
            "cpu_target": {"buffer": 1.2, "description": "CPU cores with 20% buffer"},
            "memory_target": {"buffer": 1.15, "description": "Memory with 15% buffer"},
            "replica_target": {"buffer": 1.0, "description": "Replica count (no buffer)"}
        }

        logger.info("🚀 Lightweight LSTM + Prophet Pipeline initialized")
        logger.info(f"📁 Data directory: {self.data_dir}")
        logger.info(f"📁 Model directory: {self.model_dir}")

    def load_training_data(self, service_name: str) -> pd.DataFrame:

        logger.info(f"📊 Loading training data for {service_name}")


        csv_files = [f for f in os.listdir(self.data_dir)
                     if f.startswith(service_name) and f.endswith('.csv')]

        if not csv_files:
            raise ValueError(f"No training data found for service {service_name}")

        logger.info(f"Found {len(csv_files)} data files for {service_name}")


        all_data = []
        for file in csv_files:
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)
            all_data.append(df)
            logger.info(f"Loaded {len(df)} rows from {file}")


        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total combined data: {len(combined_df)} rows")

        return combined_df

    def prepare_time_series_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series], List[str]]:

        logger.info("🔧 Preparing time series data")


        for target_name, target_config in self.targets.items():
            if target_name == "cpu_target":
                df[target_name] = df['cpu_cores_value'] * target_config["buffer"]
            elif target_name == "memory_target":
                df[target_name] = df['mem_bytes_value'] * target_config["buffer"]
            elif target_name == "replica_target":
                df[target_name] = df['replica_count_value'] * target_config["buffer"]


        df = self._create_time_features(df)


        feature_columns = [
            'cpu_cores_value', 'mem_bytes_value', 'net_rx_bytes_value', 'net_tx_bytes_value',
            'pod_restarts_value', 'replica_count_value', 'load_users',
            'hour', 'day_of_week', 'is_weekend'
        ]


        available_features = [col for col in feature_columns if col in df.columns]
        logger.info(f"Using features: {available_features}")


        targets = {}
        for target_name in self.targets.keys():
            targets[target_name] = df[target_name]

        return df, targets, available_features

    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:

        if 'timestamp' in data.columns:


            if data['timestamp'].dtype == 'object':

                timestamp_str = data['timestamp'].astype(str)
                t_pattern_mask = timestamp_str.str.startswith('t_', na=False)

                if t_pattern_mask.sum() > len(data) * 0.8:


                    extracted = timestamp_str.str.extract(r't_(\d+)')

                    extracted = extracted.fillna(0)

                    extracted_col = extracted.iloc[:, 0] if extracted.shape[1] > 0 else pd.Series([0] * len(data))

                    data['time_index'] = pd.to_numeric(extracted_col, errors='coerce').fillna(0).astype(int)
                    data['hour'] = data['time_index'] % 24
                    data['day_of_week'] = (data['time_index'] // 24) % 7
                    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
                else:

                    try:
                        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

                        data['timestamp'] = data['timestamp'].fillna(pd.Timestamp.now())
                        data['hour'] = data['timestamp'].dt.hour
                        data['day_of_week'] = data['timestamp'].dt.dayofweek
                        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

                        data['time_index'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds() / 60
                    except Exception as e:
                        logger.warning(f"Could not parse timestamps: {e}. Skipping time features.")

                        data['time_index'] = range(len(data))
                        data['hour'] = data['time_index'] % 24
                        data['day_of_week'] = (data['time_index'] // 24) % 7
                        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
            else:

                data['time_index'] = range(len(data))
                data['hour'] = data['time_index'] % 24
                data['day_of_week'] = (data['time_index'] // 24) % 7
                data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        else:

            data['time_index'] = range(len(data))
            data['hour'] = data['time_index'] % 24
            data['day_of_week'] = (data['time_index'] // 24) % 7
            data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        return data

    def create_prophet_models(self, service_name: str, df: pd.DataFrame, targets: Dict[str, pd.Series]) -> Dict[str, Any]:

        logger.info("📈 Creating Prophet models")

        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, skipping Prophet models")
            return {}

        prophet_results = {}

        for target_name, target_series in targets.items():
            logger.info(f"Training Prophet model for {target_name}")

            try:

                prophet_df = pd.DataFrame({
                    'ds': pd.date_range(start='2024-01-01', periods=len(target_series), freq='15min'),
                    'y': target_series.values
                })


                model = Prophet(
                    yearly_seasonality=self.config["prophet"]["yearly_seasonality"],
                    weekly_seasonality=self.config["prophet"]["weekly_seasonality"],
                    daily_seasonality=self.config["prophet"]["daily_seasonality"],
                    changepoint_prior_scale=self.config["prophet"]["changepoint_prior_scale"],
                    seasonality_prior_scale=self.config["prophet"]["seasonality_prior_scale"],
                    interval_width=self.config["prophet"]["interval_width"]
                )


                model.fit(prophet_df)


                future = model.make_future_dataframe(periods=30, freq='15min')
                forecast = model.predict(future)

                prophet_results[target_name] = {
                    'model': model,
                    'forecast': forecast,
                    'status': 'success'
                }

                logger.info(f"✅ Prophet model for {target_name} trained successfully")

            except Exception as e:
                logger.error(f"❌ Prophet model for {target_name} failed: {e}")
                prophet_results[target_name] = {
                    'model': None,
                    'forecast': None,
                    'status': 'failed',
                    'error': str(e)
                }

        return prophet_results

    def create_lstm_models(self, service_name: str, df: pd.DataFrame, targets: Dict[str, pd.Series], features: List[str]) -> Dict[str, Any]:

        logger.info("🧠 Creating LSTM models")

        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping LSTM models")
            return {}

        lstm_results = {}


        X = df[features].fillna(0)


        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        for target_name, target_series in targets.items():
            logger.info(f"Training LSTM model for {target_name}")

            try:
                y = target_series.fillna(target_series.median())


                sequence_length = self.config["lstm"]["sequence_length"]
                X_seq, y_seq = self._create_sequences(X_scaled, y, sequence_length)

                if len(X_seq) == 0:
                    logger.warning(f"No sequences created for {target_name}, skipping")
                    continue


                X_train, X_test, y_train, y_test = train_test_split(
                    X_seq, y_seq, test_size=0.2, random_state=42
                )


                model = self._build_lstm_model(X_train.shape[2])


                history = model.fit(
                    X_train, y_train,
                    epochs=self.config["lstm"]["epochs"],
                    batch_size=self.config["lstm"]["batch_size"],
                    validation_split=self.config["lstm"]["validation_split"],
                    callbacks=[
                        callbacks.EarlyStopping(
                            patience=self.config["lstm"]["early_stopping_patience"],
                            restore_best_weights=True
                        ),
                        callbacks.ReduceLROnPlateau(
                            factor=0.5,
                            patience=3,
                            min_lr=1e-6
                        )
                    ],
                    verbose=self.config["lstm"]["verbose"]
                )


                y_pred = model.predict(X_test, verbose=0)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                lstm_results[target_name] = {
                    'model': model,
                    'scaler': scaler,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'history': history,
                    'status': 'success'
                }

                logger.info(f"✅ LSTM model for {target_name} trained successfully")
                logger.info(f"   MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")

            except Exception as e:
                logger.error(f"❌ LSTM model for {target_name} failed: {e}")
                lstm_results[target_name] = {
                    'model': None,
                    'scaler': None,
                    'status': 'failed',
                    'error': str(e)
                }

        return lstm_results

    def _create_sequences(self, X, y, sequence_length):

        X_seq = []
        y_seq = []

        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])

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


        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1))


        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config["lstm"]["learning_rate"]),
            loss='mse',
            metrics=['mae']
        )

        return model

    def create_fusion_predictions(self, prophet_results: Dict[str, Any], lstm_results: Dict[str, Any]) -> Dict[str, Any]:

        logger.info("🔗 Creating fusion predictions")

        fusion_results = {}

        for target_name in self.targets.keys():
            logger.info(f"Creating fusion prediction for {target_name}")

            try:

                prophet_pred = None
                if target_name in prophet_results and prophet_results[target_name]['status'] == 'success':
                    prophet_forecast = prophet_results[target_name]['forecast']
                    prophet_pred = float(prophet_forecast.iloc[-1]['yhat'])


                lstm_pred = None
                if target_name in lstm_results and lstm_results[target_name]['status'] == 'success':

                    lstm_model = lstm_results[target_name]['model']
                    lstm_scaler = lstm_results[target_name]['scaler']


                    lstm_pred = 0.0


                if prophet_pred is not None and lstm_pred is not None:

                    fused_pred = (
                        self.config["fusion"]["prophet_weight"] * prophet_pred +
                        self.config["fusion"]["lstm_weight"] * lstm_pred
                    )
                    confidence = self._calculate_confidence(prophet_results.get(target_name), lstm_results.get(target_name))
                elif prophet_pred is not None:

                    fused_pred = prophet_pred
                    confidence = 0.6
                elif lstm_pred is not None:

                    fused_pred = lstm_pred
                    confidence = 0.5
                else:

                    fused_pred = 0.0
                    confidence = 0.3

                fusion_results[target_name] = {
                    'prediction': fused_pred,
                    'confidence': confidence,
                    'prophet_prediction': prophet_pred,
                    'lstm_prediction': lstm_pred,
                    'status': 'success'
                }

                logger.info(f"✅ Fusion prediction for {target_name}: {fused_pred:.6f} (confidence: {confidence:.2f})")

            except Exception as e:
                logger.error(f"❌ Fusion prediction for {target_name} failed: {e}")
                fusion_results[target_name] = {
                    'prediction': 0.0,
                    'confidence': 0.0,
                    'status': 'failed',
                    'error': str(e)
                }

        return fusion_results

    def _calculate_confidence(self, prophet_result, lstm_result):

        confidence = self.config["fusion"]["base_confidence"]


        if lstm_result and lstm_result.get('status') == 'success':
            mse = lstm_result.get('mse', 0)
            if mse > 0:
                confidence = max(0.5, min(0.9, 1.0 - (mse / 1000.0)))

        return confidence

    def save_pipeline(self, service_name: str, prophet_results: Dict[str, Any],
                     lstm_results: Dict[str, Any], fusion_results: Dict[str, Any]) -> str:

        logger.info(f"💾 Saving pipeline for {service_name}")

        model_path = os.path.join(self.model_dir, f"{service_name}_lstm_prophet_pipeline.joblib")

        pipeline_data = {
            'prophet_models': prophet_results,
            'lstm_models': lstm_results,
            'fusion_results': fusion_results,
            'config': self.config,
            'targets': self.targets,
            'trained_at': datetime.now().isoformat(),
            'service_name': service_name,
            'pipeline_type': 'lightweight_lstm_prophet',
            'version': '1.0'
        }

        joblib.dump(pipeline_data, model_path)


        file_size = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"✅ Pipeline saved to: {model_path} ({file_size:.2f} MB)")

        return model_path

    def train_pipeline(self, service_name: str) -> Dict[str, Any]:

        logger.info(f"🚀 Starting lightweight LSTM + Prophet training for {service_name}")
        logger.info("=" * 60)

        start_time = time.time()

        try:

            logger.info("📊 Step 1: Loading training data")
            df = self.load_training_data(service_name)
            df, targets, features = self.prepare_time_series_data(df)

            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Features: {features}")
            logger.info(f"Targets: {list(targets.keys())}")


            logger.info("📈 Step 2: Training Prophet models")
            prophet_results = self.create_prophet_models(service_name, df, targets)


            logger.info("🧠 Step 3: Training LSTM models")
            lstm_results = self.create_lstm_models(service_name, df, targets, features)


            logger.info("🔗 Step 4: Creating fusion predictions")
            fusion_results = self.create_fusion_predictions(prophet_results, lstm_results)


            logger.info("💾 Step 5: Saving pipeline")
            model_path = self.save_pipeline(service_name, prophet_results, lstm_results, fusion_results)


            training_time = time.time() - start_time

            logger.info("=" * 60)
            logger.info("✅ LIGHTWEIGHT LSTM + PROPHET TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"⏱️  Training time: {training_time:.2f} seconds")
            logger.info(f"📁 Model saved to: {model_path}")

            return {
                'status': 'success',
                'service_name': service_name,
                'model_path': model_path,
                'training_time': training_time,
                'prophet_results': prophet_results,
                'lstm_results': lstm_results,
                'fusion_results': fusion_results
            }

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(traceback.format_exc())

            return {
                'status': 'failed',
                'service_name': service_name,
                'error': str(e),
                'training_time': time.time() - start_time
            }

def main():

    import argparse

    parser = argparse.ArgumentParser(description='Lightweight LSTM + Prophet Pipeline')
    parser.add_argument('--service', required=True, help='Service name to train')
    parser.add_argument('--data-dir', default='training_data', help='Data directory')
    parser.add_argument('--model-dir', default='models', help='Model directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)


    pipeline = LightweightLSTMProphetPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir
    )


    result = pipeline.train_pipeline(args.service)

    if result['status'] == 'success':
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {result['model_path']}")
        print(f"⏱️  Training time: {result['training_time']:.2f} seconds")
        sys.exit(0)
    else:
        print(f"\n❌ Training failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
