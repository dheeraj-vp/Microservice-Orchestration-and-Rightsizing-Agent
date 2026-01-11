#!/usr/bin/env python3
"""
Final Fixed Autoscaler Pipeline
- LSTM-CLF: Fixed class imbalance, proper activation, class weights
- LSTM-REG: Fixed sequence bounds checking
- Multiple scalers: standard, robust, power
"""

import sys
import argparse
import logging
import warnings
import json
import random
import os
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, regularizers
    TENSORFLOW_AVAILABLE = True
except:
    TENSORFLOW_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False

warnings.filterwarnings('ignore')

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if TENSORFLOW_AVAILABLE:
        tf.random.set_seed(seed)

def setup_logging(log_dir="logs"):
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("pipeline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S')
    fh, ch = logging.FileHandler(log_file), logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

@dataclass
class ClassifierMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    optimal_threshold: float = 0.5
    
    def to_dict(self):
        return {k: float(v) if isinstance(v, (np.floating, float)) else 
                [[int(x) for x in row] for row in v] if k == 'confusion_matrix' else v 
                for k, v in self.__dict__.items()}

@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float = 0.0
    directional_accuracy: float = 0.0
    
    def to_dict(self):
        return {k: float(v) for k, v in self.__dict__.items()}

@dataclass
class HybridMetrics:
    mae_3way: float
    rmse_3way: float
    r2_3way: float
    lstm_weight: float
    rf_weight: float
    prophet_weight: float
    
    def to_dict(self):
        return {k: float(v) for k, v in self.__dict__.items()}

@dataclass
class PipelineResults:
    timestamp: str
    service_name: str
    scaler_type: str = "standard"
    rf_clf_metrics: Optional[ClassifierMetrics] = None
    rf_reg_metrics: Optional[RegressionMetrics] = None
    lstm_clf_metrics: Optional[ClassifierMetrics] = None
    lstm_reg_metrics: Optional[RegressionMetrics] = None
    prophet_metrics: Optional[RegressionMetrics] = None
    hybrid_3way_metrics: Optional[HybridMetrics] = None
    lstm_seq_len: int = 60
    training_samples: int = 0
    test_samples: int = 0
    
    def to_dict(self):
        return {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in self.__dict__.items()}
    
    def print_summary(self):
        print("\n" + "="*80)
        print(f"RESULTS - {self.service_name} | Scaler: {self.scaler_type.upper()}")
        print("="*80)
        print(f"Train: {self.training_samples} | Test: {self.test_samples} | seq_len: {self.lstm_seq_len}")
        
        if self.rf_clf_metrics:
            m = self.rf_clf_metrics
            print(f"\n📊 RF-CLF: Acc={m.accuracy:.4f} Prec={m.precision:.4f} Rec={m.recall:.4f} F1={m.f1:.4f} AUC={m.roc_auc:.4f}")
        if self.rf_reg_metrics:
            m = self.rf_reg_metrics
            print(f"📊 RF-REG: MAE={m.mae:.4f} RMSE={m.rmse:.4f} R²={m.r2:.4f}")
        if self.lstm_clf_metrics:
            m = self.lstm_clf_metrics
            print(f"🧠 LSTM-CLF: Acc={m.accuracy:.4f} Prec={m.precision:.4f} Rec={m.recall:.4f} F1={m.f1:.4f} AUC={m.roc_auc:.4f}")
        if self.lstm_reg_metrics:
            m = self.lstm_reg_metrics
            print(f"🧠 LSTM-REG: MAE={m.mae:.4f} RMSE={m.rmse:.4f} R²={m.r2:.4f}")
        if self.prophet_metrics:
            m = self.prophet_metrics
            print(f"📈 Prophet: MAE={m.mae:.4f} RMSE={m.rmse:.4f} R²={m.r2:.4f}")
        if self.hybrid_3way_metrics:
            m = self.hybrid_3way_metrics
            print(f"\n🔗 3-Way Hybrid: MAE={m.mae_3way:.4f} RMSE={m.rmse_3way:.4f} R²={m.r2_3way:.4f}")
            print(f"   Weights: LSTM={m.lstm_weight:.2f} RF={m.rf_weight:.2f} Prophet={m.prophet_weight:.2f}")
        print("="*80)


class AutoscalerPipeline:
    def __init__(self, data_dir="training_data", increase_threshold=0.1, seq_len=60, 
                 clf_threshold=0.7, use_smote=True, scaler_type="standard", logger=None):
        self.data_dir = Path(data_dir)
        self.increase_threshold = increase_threshold
        self.seq_len = seq_len
        self.clf_threshold = clf_threshold
        self.use_smote = use_smote
        self.scaler_type = scaler_type
        self.logger = logger or logging.getLogger("pipeline")
        
        scalers = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "power": PowerTransformer(method='yeo-johnson'),
        }
        self.scaler = scalers.get(scaler_type, StandardScaler())
        self.models = {}
    
    def load_and_prepare_data(self, service_name):
        self.logger.info(f"Loading: {service_name}")
        csv_files = list(self.data_dir.glob(f"{service_name}_*_lstm_prophet_ready.csv"))
        if not csv_files:
            raise ValueError(f"No CSV for {service_name}")
        
        data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        self.logger.info(f"Raw: {data.shape}")
        
        data = data.drop(columns=[c for c in ['experiment_id', 'service', 'scenario', 'timestamp'] if c in data.columns])
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric_cols:
            self.logger.info(f"Dropping non-numeric: {non_numeric_cols}")
        
        timestamps = pd.date_range('2024-01-01', periods=len(data), freq='5min')
        data.index = timestamps
        data = data[numeric_cols].resample('5min').mean().interpolate(method='linear')
        timestamps = data.index
        
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.fillna(0, inplace=True)
        
        for col in ['cpu_cores_value', 'node_mem_util_value', 'replica_count', 'load_users']:
            if col in data.columns:
                data[f'{col}_delta'] = data[col].diff().fillna(0)
        
        data['sin_hour'] = np.sin(2 * np.pi * timestamps.hour / 24)
        data['cos_hour'] = np.cos(2 * np.pi * timestamps.hour / 24)
        data['sin_day'] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
        data['cos_day'] = np.cos(2 * np.pi * timestamps.dayofweek / 7)
        
        if 'cpu_cores_value' in data.columns and 'replica_count' in data.columns:
            data['cpu_per_replica'] = data['cpu_cores_value'] / np.maximum(data['replica_count'], 1)
        if 'node_mem_util_value' in data.columns and 'replica_count' in data.columns:
            data['mem_per_replica'] = data['node_mem_util_value'] / np.maximum(data['replica_count'], 1)
        
        for col in ['node_cpu_util_value', 'node_mem_util_value']:
            if col in data.columns:
                data[f'{col}_log'] = np.log1p(np.abs(data[col]))
        
        for col in ['node_cpu_util_value', 'replica_count', 'load_users']:
            if col in data.columns:
                for lag in [1, 2, 3, 5, 10]:
                    data[f'{col}_lag{lag}'] = data[col].shift(lag).fillna(0)
        
        feature_cols = [c for c in data.columns if c != 'cpu_cores_value']
        X = data[feature_cols].iloc[:-1]
        
        if 'node_cpu_util_value' in data.columns:
            X['_raw_cpu'] = data['node_cpu_util_value'].iloc[:-1].values
        
        prophet_cols = ['cpu_cores_value', 'node_mem_util_value', 'replica_count', 'load_users', 
                       'replica_change', 'cpu_regime_encoded', 'scaling_intensity']
        prophet_cols = [c for c in prophet_cols if c in data.columns]
        X['_prophet_data'] = [data[prophet_cols].iloc[i].to_dict() if i < len(data) else {} for i in range(len(X))]
        
        current_cpu = data['cpu_cores_value'].replace(0, 1e-6)
        future_cpu = data['cpu_cores_value'].shift(-1).replace(0, 1e-6)
        rel_inc = (future_cpu - current_cpu) / current_cpu
        
        y_class = (rel_inc > self.increase_threshold).astype(int).iloc[:-1].fillna(0)
        y_reg = rel_inc.iloc[:-1].fillna(0).clip(-1, 1)
        timestamps = timestamps[:-1]
        
        self.logger.info(f"X={X.shape}, Classes={np.bincount(y_class)}")
        return X, y_class, y_reg, timestamps
    
    def add_rolling(self, X_train, X_test):
        if '_raw_cpu' not in X_train.columns:
            return X_train, X_test
        
        for w in [5, 10, 20, 30, 60]:
            X_train[f'cpu_roll{w}_mean'] = X_train['_raw_cpu'].rolling(w, min_periods=1).mean()
            X_train[f'cpu_roll{w}_std'] = X_train['_raw_cpu'].rolling(w, min_periods=1).std().fillna(0)
        
        combined = pd.concat([X_train['_raw_cpu'], X_test['_raw_cpu']], ignore_index=False)
        for w in [5, 10, 20, 30, 60]:
            X_test[f'cpu_roll{w}_mean'] = combined.rolling(w, min_periods=1).mean().iloc[len(X_train):].values
            X_test[f'cpu_roll{w}_std'] = combined.rolling(w, min_periods=1).std().fillna(0).iloc[len(X_train):].values
        
        return X_train.drop(columns=['_raw_cpu']), X_test.drop(columns=['_raw_cpu'])
    
    def time_split(self, X, y_class, y_reg, timestamps, test_size=0.2):
        split_idx = int(len(X) * (1 - test_size))
        return (X.iloc[:split_idx], X.iloc[split_idx:], y_class.iloc[:split_idx], y_class.iloc[split_idx:],
                y_reg.iloc[:split_idx], y_reg.iloc[split_idx:], timestamps[:split_idx], timestamps[split_idx:])
    
    def tune_rf(self, X, y):
        param_dist = {'n_estimators': [300, 500], 'max_depth': [20, 30, None], 'min_samples_split': [2, 4],
                      'max_features': ['sqrt', 0.5], 'class_weight': ['balanced']}
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        search = RandomizedSearchCV(rf, param_dist, n_iter=12, cv=TimeSeriesSplit(3), 
                                     scoring='f1', random_state=42, n_jobs=-1, verbose=0)
        search.fit(X, y)
        return search.best_estimator_
    
    def eval_clf(self, model, X, y, threshold=0.5):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba > threshold).astype(int)
        return ClassifierMetrics(
            accuracy=float(accuracy_score(y, preds)),
            precision=float(precision_score(y, preds, zero_division=0)),
            recall=float(recall_score(y, preds, zero_division=0)),
            f1=float(f1_score(y, preds, zero_division=0)),
            roc_auc=float(roc_auc_score(y, proba)),
            confusion_matrix=confusion_matrix(y, preds),
            optimal_threshold=float(threshold)
        )
    
    def eval_reg(self, preds, y):
        mae = float(mean_absolute_error(y, preds))
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        r2 = float(r2_score(y, preds))
        dir_acc = float(accuracy_score(np.diff(y) > 0, np.diff(preds) > 0)) if len(y) > 1 else 0.0
        return RegressionMetrics(mae=mae, rmse=rmse, r2=r2, directional_accuracy=dir_acc)
    
    def create_seq(self, X, y, seq_len):
        """Fixed: bounds checking"""
        max_start = len(X) - seq_len
        if max_start <= 0:
            raise ValueError(f"Sequence length {seq_len} too long for dataset size {len(X)}")
        
        X_seq, y_seq = [], []
        for i in range(max_start):
            X_seq.append(X.iloc[i:i+seq_len].values)
            y_seq.append(y.iloc[i+seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_clf(self, shape, num_classes):
        """Fixed: proper activation for binary/multiclass"""
        m = models.Sequential([
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001)), input_shape=shape),
            layers.BatchNormalization(), layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(32, kernel_regularizer=regularizers.l2(0.0001))),
            layers.BatchNormalization(), layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes if num_classes > 2 else 1, 
                        activation='softmax' if num_classes > 2 else 'sigmoid'),
        ])
        
        loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
        m.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
        return m
    
    def build_lstm_reg(self, shape):
        m = models.Sequential([
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.0001)), input_shape=shape),
            layers.BatchNormalization(), layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(32, kernel_regularizer=regularizers.l2(0.0001))),
            layers.BatchNormalization(), layers.Dropout(0.2),
            layers.Dense(32, activation='relu'), layers.Dense(1),
        ])
        m.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['mae'])
        return m
    
    def train_lstm(self, X_train, y_train, X_val, y_val, seq_len, is_clf=True):
        if not TENSORFLOW_AVAILABLE:
            return None
        
        # Ensure int targets
        y_train = y_train.astype(int) if is_clf else y_train
        y_val = y_val.astype(int) if is_clf else y_val
        
        X_tr_seq, y_tr_seq = self.create_seq(pd.DataFrame(X_train), pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train, seq_len)
        X_val_seq, y_val_seq = self.create_seq(pd.DataFrame(X_val), pd.Series(y_val) if not isinstance(y_val, pd.Series) else y_val, seq_len)
        
        if len(X_tr_seq) < 50:
            self.logger.warning(f"Insufficient sequences: {len(X_tr_seq)}")
            return None
        
        if is_clf:
            num_classes = len(np.unique(y_tr_seq))
            self.logger.info(f"Classes: {num_classes}, Distribution: {np.bincount(y_tr_seq)}")
            
            # Class weights
            cw = compute_class_weight('balanced', classes=np.unique(y_tr_seq), y=y_tr_seq)
            cw_dict = dict(enumerate(cw))
            
            m = self.build_lstm_clf((seq_len, X_train.shape[1]), num_classes)
            m.fit(X_tr_seq, y_tr_seq, validation_data=(X_val_seq, y_val_seq), epochs=100, batch_size=32, 
                  verbose=0, callbacks=[callbacks.EarlyStopping('val_loss', 10, restore_best_weights=True)], 
                  class_weight=cw_dict)
        else:
            m = self.build_lstm_reg((seq_len, X_train.shape[1]))
            m.fit(X_tr_seq, y_tr_seq, validation_data=(X_val_seq, y_val_seq), epochs=100, batch_size=32, 
                  verbose=0, callbacks=[callbacks.EarlyStopping('val_loss', 10, restore_best_weights=True)])
        return m
    
    def eval_lstm(self, m, X, y, seq_len, threshold=0.5, is_clf=True):
        y = y.astype(int) if is_clf else y
        X_seq, y_seq = self.create_seq(pd.DataFrame(X), pd.Series(y) if not isinstance(y, pd.Series) else y, seq_len)
        preds = m.predict(X_seq, verbose=0)
        
        if is_clf:
            if preds.shape[1] > 1:  # Multi-class
                preds_bin = np.argmax(preds, axis=1)
                proba = np.max(preds, axis=1)
            else:  # Binary
                preds = preds.flatten()
                preds_bin = (preds > threshold).astype(int)
                proba = preds
            
            return ClassifierMetrics(
                accuracy=float(accuracy_score(y_seq, preds_bin)),
                precision=float(precision_score(y_seq, preds_bin, average='weighted', zero_division=0)),
                recall=float(recall_score(y_seq, preds_bin, average='weighted', zero_division=0)),
                f1=float(f1_score(y_seq, preds_bin, average='weighted', zero_division=0)),
                roc_auc=float(roc_auc_score(y_seq, proba)) if len(np.unique(y_seq)) == 2 else 0.0,
                confusion_matrix=confusion_matrix(y_seq, preds_bin),
                optimal_threshold=float(threshold)
            )
        else:
            preds = preds.flatten()
            return self.eval_reg(preds, y_seq), preds
    
    def train_prophet(self, prophet_data_train, y_train, prophet_data_test, y_test, ts_train, ts_test):
        if not PROPHET_AVAILABLE:
            return None, None
        
        df = pd.DataFrame({'ds': ts_train, 'y': y_train.values})
        regs = ['replica_change', 'cpu_regime_encoded', 'scaling_intensity']
        for c in regs:
            if len(prophet_data_train) > 0 and c in prophet_data_train[0]:
                df[c] = [d.get(c, 0) for d in prophet_data_train]
        
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        for c in regs:
            if c in df.columns:
                m.add_regressor(c)
        m.fit(df)
        
        future = m.make_future_dataframe(len(y_test), '5min')
        for c in regs:
            if c in df.columns:
                future[c] = [d.get(c, 0) for d in prophet_data_train] + [d.get(c, 0) for d in prophet_data_test]
        
        preds = m.predict(future)['yhat'].iloc[-len(y_test):].values
        return self.eval_reg(preds, y_test.values), preds
    
    def hybrid_3way(self, lstm_preds, rf_preds, prophet_preds, y_true):
        min_len = min(len(lstm_preds), len(rf_preds), len(prophet_preds))
        lstm_p, rf_p, prophet_p, y = lstm_preds[:min_len], rf_preds[:min_len], prophet_preds[:min_len], y_true[:min_len]
        
        lstm_rmse = float(np.sqrt(mean_squared_error(y, lstm_p)))
        rf_rmse = float(np.sqrt(mean_squared_error(y, rf_p)))
        prophet_rmse = float(np.sqrt(mean_squared_error(y, prophet_p)))
        
        total = (1/lstm_rmse) + (1/rf_rmse) + (1/prophet_rmse)
        lstm_w, rf_w, prophet_w = (1/lstm_rmse) / total, (1/rf_rmse) / total, (1/prophet_rmse) / total
        
        hybrid_preds = lstm_w * lstm_p + rf_w * rf_p + prophet_w * prophet_p
        mae = float(mean_absolute_error(y, hybrid_preds))
        rmse = float(np.sqrt(mean_squared_error(y, hybrid_preds)))
        r2 = float(r2_score(y, hybrid_preds))
        
        return HybridMetrics(mae, rmse, r2, lstm_w, rf_w, prophet_w)
    
    def train_all(self, service_name):
        self.logger.info(f"Training: {service_name}")
        
        X, y_class, y_reg, timestamps = self.load_and_prepare_data(service_name)
        
        prophet_data = X['_prophet_data'].tolist()
        X = X.drop(columns=['_prophet_data'])
        
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test, ts_train, ts_test = \
            self.time_split(X, y_class, y_reg, timestamps, 0.2)
        
        prophet_data_train = prophet_data[:len(X_train)]
        prophet_data_test = prophet_data[len(X_train):len(X_train)+len(X_test)]
        
        X_train, X_test = self.add_rolling(X_train, X_test)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        if self.use_smote and len(np.bincount(y_class_train)) > 1:
            try:
                X_train_s, y_class_train = SMOTE(random_state=42).fit_resample(X_train_s, y_class_train)
                self.logger.info(f"SMOTE: {np.bincount(y_class_train)}")
            except:
                pass
        
        res = PipelineResults(datetime.now().isoformat(), service_name, self.scaler_type, lstm_seq_len=self.seq_len)
        res.training_samples, res.test_samples = len(X_train), len(X_test)
        
        # RF CLF
        try:
            self.logger.info("RF-CLF...")
            rf_clf = self.tune_rf(X_train_s, y_class_train.values if hasattr(y_class_train, 'values') else y_class_train)
            res.rf_clf_metrics = self.eval_clf(rf_clf, X_test_s, y_class_test.values, self.clf_threshold)
            self.models['rf_clf'] = rf_clf
            self.logger.info(f"✓ RF-CLF F1={res.rf_clf_metrics.f1:.4f}")
        except Exception as e:
            self.logger.error(f"RF-CLF: {e}")
        
        # RF REG
        rf_reg_preds = None
        try:
            self.logger.info("RF-REG...")
            rf_reg = RandomForestRegressor(n_estimators=300, max_depth=30, random_state=42, n_jobs=-1)
            rf_reg.fit(X_train_s, y_reg_train.values)
            rf_reg_preds = rf_reg.predict(X_test_s)
            res.rf_reg_metrics = self.eval_reg(rf_reg_preds, y_reg_test.values)
            self.models['rf_reg'] = rf_reg
            self.logger.info(f"✓ RF-REG MAE={res.rf_reg_metrics.mae:.4f}")
        except Exception as e:
            self.logger.error(f"RF-REG: {e}")
        
        # LSTM CLF
        if TENSORFLOW_AVAILABLE:
            try:
                self.logger.info("LSTM-CLF...")
                lstm_clf = self.train_lstm(X_train_s, y_class_train, X_test_s, y_class_test, self.seq_len, is_clf=True)
                if lstm_clf:
                    res.lstm_clf_metrics = self.eval_lstm(lstm_clf, X_test_s, y_class_test, self.seq_len, self.clf_threshold, is_clf=True)
                    self.models['lstm_clf'] = lstm_clf
                    self.logger.info(f"✓ LSTM-CLF F1={res.lstm_clf_metrics.f1:.4f}")
            except Exception as e:
                self.logger.error(f"LSTM-CLF: {e}")
        
        # LSTM REG
        lstm_reg_preds = None
        if TENSORFLOW_AVAILABLE:
            try:
                self.logger.info("LSTM-REG...")
                lstm_reg = self.train_lstm(X_train_s, y_reg_train, X_test_s, y_reg_test, self.seq_len, is_clf=False)
                if lstm_reg:
                    res.lstm_reg_metrics, lstm_reg_preds = self.eval_lstm(lstm_reg, X_test_s, y_reg_test, self.seq_len, is_clf=False)
                    self.models['lstm_reg'] = lstm_reg
                    self.logger.info(f"✓ LSTM-REG MAE={res.lstm_reg_metrics.mae:.4f}")
            except Exception as e:
                self.logger.error(f"LSTM-REG: {e}")
        
        # Prophet
        prophet_preds = None
        if PROPHET_AVAILABLE:
            try:
                self.logger.info("Prophet...")
                res.prophet_metrics, prophet_preds = self.train_prophet(prophet_data_train, y_reg_train, prophet_data_test, y_reg_test, ts_train, ts_test)
                self.logger.info(f"✓ Prophet MAE={res.prophet_metrics.mae:.4f}")
            except Exception as e:
                self.logger.error(f"Prophet: {e}")
        
        # 3-Way Hybrid
        if lstm_reg_preds is not None and rf_reg_preds is not None and prophet_preds is not None:
            try:
                y_test_aligned = y_reg_test.values[self.seq_len:]
                res.hybrid_3way_metrics = self.hybrid_3way(lstm_reg_preds, rf_reg_preds[self.seq_len:], prophet_preds[self.seq_len:], y_test_aligned)
                self.logger.info(f"✓ 3-Way MAE={res.hybrid_3way_metrics.mae_3way:.4f}")
            except Exception as e:
                self.logger.error(f"Hybrid: {e}")
        
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", default="frontend")
    parser.add_argument("--data-dir", default="training_data")
    parser.add_argument("--increase-threshold", type=float, default=0.1)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--clf-threshold", type=float, default=0.7)
    parser.add_argument("--scaler", default="standard", choices=["standard", "robust", "power"])
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seeds(args.seed)
    logger = setup_logging(args.log_dir)
    
    try:
        pipeline = AutoscalerPipeline(args.data_dir, args.increase_threshold, args.seq_len, 
                                      args.clf_threshold, not args.no_smote, args.scaler, logger)
        results = pipeline.train_all(args.service)
        results.print_summary()
        
        with open(Path(args.log_dir) / f"results_{args.service}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        if args.save_models:
            models_dir = Path("trained_models") / args.service
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline.scaler, models_dir / "scaler.pkl")
            for name, model in pipeline.models.items():
                if 'lstm' in name:
                    model.save(models_dir / f"{name}.keras")
                else:
                    joblib.dump(model, models_dir / f"{name}.pkl")
        
        logger.info("Done!")
    except Exception as e:
        logger.exception(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
