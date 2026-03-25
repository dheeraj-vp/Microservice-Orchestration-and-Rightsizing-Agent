#!/usr/bin/env python3

import os
import sys
import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


from mora.cli.main import main
from click.testing import CliRunner

class TestProfessionalMLPipeline:

    def setup_method(self):

        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'training_data')
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


        self.create_sample_data()

    def teardown_method(self):

        shutil.rmtree(self.temp_dir)

    def create_sample_data(self):


        sample_data = {
            'timestamp': [f't_{i}' for i in range(100)],
            'experiment_id': ['frontend_browsing_replicas_1_users_5'] * 100,
            'service': ['frontend'] * 100,
            'scenario': ['browsing'] * 100,
            'replica_count': [1] * 100,
            'load_users': [5] * 100,
            'cpu_cores_value': np.random.uniform(0.1, 0.5, 100),
            'mem_bytes_value': np.random.uniform(1000000, 5000000, 100),
            'net_rx_bytes_value': np.random.uniform(1000, 10000, 100),
            'net_tx_bytes_value': np.random.uniform(1000, 10000, 100),
            'pod_restarts_value': np.random.uniform(0, 2, 100),
            'replica_count_value': [1] * 100,
            'node_cpu_util_value': np.random.uniform(0.1, 0.8, 100),
            'node_mem_util_value': np.random.uniform(0.1, 0.8, 100),
            'network_activity_rate_value': np.random.uniform(0.1, 0.5, 100),
            'processing_intensity_value': np.random.uniform(0.1, 0.5, 100),
            'service_stability_value': np.random.uniform(0.8, 1.0, 100),
            'resource_pressure_value': np.random.uniform(0.1, 0.5, 100),
        }

        df = pd.DataFrame(sample_data)
        csv_path = os.path.join(self.data_dir, 'frontend_browsing_replicas_1_users_5.csv')
        df.to_csv(csv_path, index=False)

    @patch('train_models.train_professional_ml_pipeline.ProfessionalMLPipeline')
    def test_models_training_command(self, mock_pipeline_class):


        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.train_service.return_value = {
            'status': 'success',
            'service_name': 'frontend',
            'training_time': 120.5,
            'models_trained': 3,
            'best_model': 'XGBoost',
            'performance': {
                'r2': 0.95,
                'mae': 0.05,
                'mape': 0.08
            }
        }

        runner = CliRunner()
        result = runner.invoke(main, [
            'train', 'models',
            '--service', 'frontend',
            '--data-dir', self.data_dir,
            '--model-dir', self.model_dir
        ])

        assert result.exit_code == 0
        assert 'Training completed successfully!' in result.output
        assert 'frontend' in result.output
        assert 'XGBoost' in result.output

    @patch('train_models.train_professional_ml_pipeline.ProfessionalMLPipeline')
    def test_models_training_multiple_services(self, mock_pipeline_class):


        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.train_all_services.return_value = {
            'total_services': 2,
            'successful_services': 2,
            'failed_services': 0,
            'success_rate': 1.0,
            'successful_services_list': ['frontend', 'cartservice'],
            'failed_services_list': []
        }

        runner = CliRunner()
        result = runner.invoke(main, [
            'train', 'models',
            '--services', 'frontend,cartservice',
            '--data-dir', self.data_dir,
            '--model-dir', self.model_dir
        ])

        assert result.exit_code == 0
        assert 'Training Summary' in result.output
        assert 'frontend' in result.output
        assert 'cartservice' in result.output

    @patch('evaluate_models.evaluate_professional_models.ProfessionalModelEvaluator')
    def test_professional_evaluation_command(self, mock_evaluator_class):


        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_evaluator.evaluate_service.return_value = {
            'status': 'success',
            'service_name': 'frontend',
            'evaluation_time': 45.2,
            'comparison_results': {
                'best_model': 'XGBoost'
            },
            'production_assessment': {
                'overall_readiness': 'Production Ready',
                'recommendations': [
                    'Model shows excellent performance',
                    'Ready for deployment',
                    'Consider monitoring drift'
                ]
            }
        }

        runner = CliRunner()
        result = runner.invoke(main, [
            'train', 'evaluate',
            '--service', 'frontend',
            '--model-dir', self.model_dir,
            '--output-dir', os.path.join(self.temp_dir, 'evaluation_results')
        ])

        assert result.exit_code == 0
        assert 'Evaluation completed successfully!' in result.output
        assert 'Production Ready' in result.output

    def test_models_command_help(self):

        runner = CliRunner()
        result = runner.invoke(main, ['train', 'models', '--help'])

        assert result.exit_code == 0
        assert 'Train ML models using advanced algorithms' in result.output
        assert '--service' in result.output
        assert '--services' in result.output
        assert '--config' in result.output

    def test_evaluation_command_help(self):

        runner = CliRunner()
        result = runner.invoke(main, ['train', 'evaluate', '--help'])

        assert result.exit_code == 0
        assert 'Evaluate trained models using comprehensive analysis' in result.output
        assert '--service' in result.output
        assert '--services' in result.output
        assert '--model-dir' in result.output

    def test_models_training_with_config(self):


        config_data = {
            'algorithms': ['RandomForest', 'XGBoost'],
            'hyperparameters': {
                'RandomForest': {'n_estimators': 100},
                'XGBoost': {'n_estimators': 100}
            },
            'evaluation': {
                'cv_folds': 5,
                'test_size': 0.2
            }
        }

        config_path = os.path.join(self.temp_dir, 'test_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        with patch('train_models.train_professional_ml_pipeline.ProfessionalMLPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.train_service.return_value = {
                'status': 'success',
                'service_name': 'frontend',
                'training_time': 120.5,
                'models_trained': 2,
                'best_model': 'XGBoost'
            }

            runner = CliRunner()
            result = runner.invoke(main, [
                'train', 'models',
                '--service', 'frontend',
                '--config', config_path,
                '--data-dir', self.data_dir,
                '--model-dir', self.model_dir
            ])

            assert result.exit_code == 0
            assert 'Training completed successfully!' in result.output

class TestProfessionalMLIntegration:

    def setup_method(self):

        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'training_data')
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def teardown_method(self):

        shutil.rmtree(self.temp_dir)

    def test_cli_command_structure(self):

        runner = CliRunner()


        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'train' in result.output


        result = runner.invoke(main, ['train', '--help'])
        assert result.exit_code == 0
        assert 'models' in result.output
        assert 'evaluate' in result.output
        assert 'collect-data' in result.output

    def test_professional_ml_files_exist(self):

        pipeline_file = Path("train_models/train_professional_ml_pipeline.py")
        evaluator_file = Path("evaluate_models/evaluate_professional_models.py")
        config_file = Path("config/professional_ml_config.json")

        assert pipeline_file.exists(), f"Professional ML pipeline file not found: {pipeline_file}"
        assert evaluator_file.exists(), f"Professional evaluator file not found: {evaluator_file}"
        assert config_file.exists(), f"Professional ML config file not found: {config_file}"

    def test_professional_ml_imports(self):

        try:
            from train_models.train_professional_ml_pipeline import ProfessionalMLPipeline
            from evaluate_models.evaluate_professional_models import ProfessionalModelEvaluator
        except ImportError as e:
            pytest.fail(f"Failed to import professional ML components: {e}")

    def test_professional_ml_config_loading(self):

        config_file = Path("config/professional_ml_config.json")

        import json
        with open(config_file, 'r') as f:
            config = json.load(f)

        assert isinstance(config, dict), "Config should be a dictionary"
        assert 'algorithms' in config, "Config should contain algorithms section"
        assert 'hyperparameters' in config, "Config should contain hyperparameters section"
        assert 'evaluation' in config, "Config should contain evaluation section"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
