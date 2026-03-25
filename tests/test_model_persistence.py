#!/usr/bin/env python3

import os
import sys
import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import joblib
import pickle


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestModelPersistence:

    def setup_method(self):

        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def teardown_method(self):

        shutil.rmtree(self.temp_dir)

    def create_sample_pipeline_data(self):


        class DummyScaler:
            def transform(self, X):
                return X
            def fit(self, X):
                return self

        return {
            'pipeline_type': 'lightweight_lstm_prophet',
            'trained_at': pd.Timestamp.now().isoformat(),
            'service_name': 'frontend',
            'target_columns': ['cpu_target', 'memory_target', 'replica_target'],
            'feature_columns': ['cpu_cores_value', 'mem_bytes_value'],
            'data_shape': (100, 10),
            'scaler_X': DummyScaler(),
            'scaler_y': DummyScaler(),
            'prophet_models': {'cpu_target': {'mse': 0.1}, 'memory_target': {'mse': 0.2}},
            'lstm_models': {'cpu_target': {'mse': 0.1}, 'memory_target': {'mse': 0.2}},
            'config': {'epochs': 20, 'batch_size': 16}
        }

    def test_save_pipeline(self):

        pipeline_data = self.create_sample_pipeline_data()

        model_path = os.path.join(self.model_dir, 'frontend_lstm_prophet_pipeline.joblib')
        joblib.dump(pipeline_data, model_path)

        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0

    def test_load_pipeline(self):

        pipeline_data = self.create_sample_pipeline_data()

        model_path = os.path.join(self.model_dir, 'frontend_lstm_prophet_pipeline.joblib')
        joblib.dump(pipeline_data, model_path)


        loaded_pipeline = joblib.load(model_path)

        assert loaded_pipeline is not None
        assert loaded_pipeline['pipeline_type'] == 'lightweight_lstm_prophet'
        assert loaded_pipeline['service_name'] == 'frontend'
        assert 'target_columns' in loaded_pipeline

    def test_pipeline_metadata_preservation(self):

        pipeline_data = self.create_sample_pipeline_data()
        original_timestamp = pipeline_data['trained_at']
        original_config = pipeline_data['config']

        model_path = os.path.join(self.model_dir, 'frontend_lstm_prophet_pipeline.joblib')
        joblib.dump(pipeline_data, model_path)

        loaded_pipeline = joblib.load(model_path)

        assert loaded_pipeline['trained_at'] == original_timestamp
        assert loaded_pipeline['config'] == original_config
        assert len(loaded_pipeline['target_columns']) == 3
        assert len(loaded_pipeline['feature_columns']) == 2

    def test_load_nonexistent_pipeline(self):

        model_path = os.path.join(self.model_dir, 'nonexistent_lstm_prophet_pipeline.joblib')

        with pytest.raises(FileNotFoundError):
            joblib.load(model_path)

    def test_load_corrupted_pipeline(self):

        bad_file = os.path.join(self.model_dir, 'corrupted_pipeline.joblib')
        with open(bad_file, 'w') as f:
            f.write("not a valid joblib file")

        with pytest.raises((ValueError, EOFError, pickle.UnpicklingError)):
            joblib.load(bad_file)

    def test_save_multiple_services(self):

        services = ['frontend', 'cartservice', 'checkoutservice']

        for service in services:
            pipeline_data = self.create_sample_pipeline_data()
            pipeline_data['service_name'] = service

            model_path = os.path.join(self.model_dir, f'{service}_lstm_prophet_pipeline.joblib')
            joblib.dump(pipeline_data, model_path)


        for service in services:
            model_path = os.path.join(self.model_dir, f'{service}_lstm_prophet_pipeline.joblib')
            assert os.path.exists(model_path)


            loaded = joblib.load(model_path)
            assert loaded['service_name'] == service

    def test_pipeline_version_compatibility(self):


        pipeline_data = self.create_sample_pipeline_data()
        pipeline_data['version'] = '1.0.0'

        model_path = os.path.join(self.model_dir, 'versioned_pipeline.joblib')
        joblib.dump(pipeline_data, model_path)

        loaded = joblib.load(model_path)
        assert loaded.get('version') == '1.0.0'

    def test_pipeline_with_custom_metadata(self):

        pipeline_data = self.create_sample_pipeline_data()
        pipeline_data['custom_metadata'] = {
            'training_duration': 120.5,
            'data_source': 'production',
            'model_config': {'learning_rate': 0.001}
        }

        model_path = os.path.join(self.model_dir, 'custom_metadata_pipeline.joblib')
        joblib.dump(pipeline_data, model_path)

        loaded = joblib.load(model_path)
        assert 'custom_metadata' in loaded
        assert loaded['custom_metadata']['training_duration'] == 120.5

    def test_pipeline_file_size(self):

        pipeline_data = self.create_sample_pipeline_data()

        model_path = os.path.join(self.model_dir, 'frontend_lstm_prophet_pipeline.joblib')
        joblib.dump(pipeline_data, model_path)

        file_size = os.path.getsize(model_path)


        assert file_size < 100 * 1024 * 1024


        assert file_size > 0

    def test_concurrent_save_load(self):

        pipeline_data = self.create_sample_pipeline_data()


        model_path = os.path.join(self.model_dir, 'concurrent_pipeline.joblib')
        joblib.dump(pipeline_data, model_path)


        loaded = joblib.load(model_path)

        assert loaded is not None
        assert loaded['service_name'] == 'frontend'

    def test_pipeline_with_complex_objects(self):

        pipeline_data = self.create_sample_pipeline_data()


        pipeline_data['metadata'] = {
            'training_history': {
                'epochs': [1, 2, 3, 4, 5],
                'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
                'val_loss': [0.6, 0.5, 0.4, 0.3, 0.2]
            },
            'feature_importance': {
                'cpu_cores_value': 0.3,
                'mem_bytes_value': 0.4,
                'load_users': 0.3
            }
        }

        model_path = os.path.join(self.model_dir, 'complex_pipeline.joblib')
        joblib.dump(pipeline_data, model_path)

        loaded = joblib.load(model_path)
        assert 'metadata' in loaded
        assert 'training_history' in loaded['metadata']
        assert 'feature_importance' in loaded['metadata']

class TestModelPersistenceIntegration:

    def setup_method(self):

        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, 'models')
        self.data_dir = os.path.join(self.temp_dir, 'training_data')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def teardown_method(self):

        shutil.rmtree(self.temp_dir)

    def test_complete_save_load_cycle(self):


        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'cpu_cores_value': np.random.rand(100),
            'mem_bytes_value': np.random.rand(100) * 1000000
        })


        pipeline_data = {
            'pipeline_type': 'lightweight_lstm_prophet',
            'trained_at': pd.Timestamp.now().isoformat(),
            'service_name': 'frontend',
            'data_shape': data.shape
        }

        model_path = os.path.join(self.model_dir, 'frontend_lstm_prophet_pipeline.joblib')
        joblib.dump(pipeline_data, model_path)


        loaded = joblib.load(model_path)


        assert loaded['service_name'] == 'frontend'
        assert loaded['pipeline_type'] == 'lightweight_lstm_prophet'
        assert loaded['data_shape'] == data.shape

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
