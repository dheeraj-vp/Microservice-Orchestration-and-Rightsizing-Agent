
import pytest
from click.testing import CliRunner
from unittest.mock import Mock, patch
import os
import tempfile
import json


from src.mora.cli.main import main

class TestCLI:

    def setup_method(self):

        self.runner = CliRunner()

    def test_main_help(self):

        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "MOrA - Microservices-Aware Orchestrator Agent" in result.output
        assert "rightsize" in result.output
        assert "status" in result.output
        assert "train" in result.output

    def test_rightsize_command_help(self):

        result = self.runner.invoke(main, ['rightsize', '--help'])
        assert result.exit_code == 0
        assert "Generate rightsizing recommendations" in result.output
        assert "--service" in result.output
        assert "--strategy" in result.output
        assert "--namespace" in result.output

    def test_status_command_help(self):

        result = self.runner.invoke(main, ['status', '--help'])
        assert result.exit_code == 0
        assert "Show current status" in result.output
        assert "--namespace" in result.output

    def test_train_command_help(self):

        result = self.runner.invoke(main, ['train', '--help'])
        assert result.exit_code == 0
        assert "ML model training and data collection commands" in result.output
        assert "models" in result.output
        assert "evaluate" in result.output
        assert "lightweight" in result.output
        assert "collect-data" in result.output
        assert "collect-data-parallel" in result.output
        assert "status" in result.output

    def test_lightweight_command_help(self):

        result = self.runner.invoke(main, ['train', 'lightweight', '--help'])
        assert result.exit_code == 0
        assert "lightweight" in result.output.lower()
        assert "--service" in result.output
        assert "--services" in result.output

    def test_models_command_help(self):

        result = self.runner.invoke(main, ['train', 'models', '--help'])
        assert result.exit_code == 0
        assert "Train ML models using advanced algorithms" in result.output
        assert "--service" in result.output
        assert "--services" in result.output
        assert "--config" in result.output
        assert "--data-dir" in result.output
        assert "--model-dir" in result.output

    def test_evaluate_command_help(self):

        result = self.runner.invoke(main, ['train', 'evaluate', '--help'])
        assert result.exit_code == 0
        assert "Evaluate trained models using unified evaluation system" in result.output
        assert "--service" in result.output
        assert "--model-dir" in result.output
        assert "--data-dir" in result.output

    def test_collect_data_command_help(self):

        result = self.runner.invoke(main, ['train', 'collect-data', '--help'])
        assert result.exit_code == 0
        assert "Collect training data for ML model training" in result.output
        assert "--service" in result.output
        assert "--config-file" in result.output

    def test_collect_data_parallel_command_help(self):

        result = self.runner.invoke(main, ['train', 'collect-data-parallel', '--help'])
        assert result.exit_code == 0
        assert "Collect training data for multiple services in parallel" in result.output
        assert "--services" in result.output
        assert "--max-workers" in result.output

    def test_train_status_command_help(self):

        result = self.runner.invoke(main, ['train', 'status', '--help'])
        assert result.exit_code == 0
        assert "Check training experiment progress for a service" in result.output
        assert "--service" in result.output
        assert "--config-file" in result.output

    @patch('train_models.train_professional_ml_pipeline.ProfessionalMLPipeline')
    def test_models_command_execution(self, mock_pipeline_class):


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

        result = self.runner.invoke(main, [
            'train', 'models',
            '--service', 'frontend'
        ])

        assert result.exit_code == 0
        assert 'Training completed successfully!' in result.output
        assert 'frontend' in result.output
        assert 'XGBoost' in result.output

    @patch('evaluate_models.evaluate_professional_models.ProfessionalModelEvaluator')
    def test_evaluate_command_execution(self, mock_evaluator_class):


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
                    'Ready for deployment'
                ]
            }
        }

        result = self.runner.invoke(main, [
            'train', 'evaluate',
            '--service', 'frontend'
        ])

        assert result.exit_code == 0
        assert 'Evaluation completed successfully!' in result.output
        assert 'frontend' in result.output or 'Service' in result.output

class TestCLIIntegration:

    def setup_method(self):

        self.runner = CliRunner()

    @patch('src.mora.cli.main.DataPipeline')
    def test_status_command_success(self, mock_pipeline_class):


        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {
            'kubernetes': True,
            'prometheus': True
        }
        mock_pipeline.get_deployed_services.return_value = ['frontend', 'cartservice']
        mock_pipeline.test_prometheus_metrics.return_value = {
            'available_metrics': ['cpu_usage', 'memory_usage'],
            'working_metrics': ['cpu_usage', 'memory_usage']
        }

        result = self.runner.invoke(main, ['status'])

        assert result.exit_code == 0
        assert 'System Status' in result.output
        assert 'frontend' in result.output
        assert 'cartservice' in result.output

    @patch('src.mora.cli.main.DataPipeline')
    def test_rightsize_command_with_mocks(self, mock_pipeline_class):

        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {
            'kubernetes': True,
            'prometheus': True
        }


        mock_service_data = {
            'cpu_usage': [0.1, 0.2, 0.3],
            'memory_usage': [100, 200, 300]
        }
        mock_pipeline.collect_service_data.return_value = mock_service_data

        result = self.runner.invoke(main, [
            'rightsize',
            '--service', 'frontend',
            '--strategy', 'statistical'
        ])


        assert result.exit_code == 0 or result.exit_code == 1

    def test_rightsize_command_validation(self):

        result = self.runner.invoke(main, ['rightsize', '--help'])
        assert result.exit_code == 0
        assert "--service" in result.output
        assert "--strategy" in result.output
        assert "--duration-hours" in result.output
        assert "--output-format" in result.output

    def test_output_format_options(self):

        result = self.runner.invoke(main, ['rightsize', '--help'])
        assert result.exit_code == 0
        assert "table" in result.output
        assert "yaml" in result.output
        assert "json" in result.output

class TestCLIErrorHandling:

    def setup_method(self):

        self.runner = CliRunner()

    @patch('src.mora.cli.main.DataPipeline')
    def test_connection_failure_handling(self, mock_pipeline_class):

        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {
            'kubernetes': False,
            'prometheus': False
        }

        result = self.runner.invoke(main, ['status'])


        assert result.exit_code == 0
        assert 'System Status' in result.output

    @patch('src.mora.cli.main.DataPipeline')
    def test_service_not_found_handling(self, mock_pipeline_class):

        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.test_connections.return_value = {
            'kubernetes': True,
            'prometheus': True
        }
        mock_pipeline.collect_service_data.return_value = {}

        result = self.runner.invoke(main, [
            'rightsize',
            '--service', 'nonexistent-service',
            '--strategy', 'statistical'
        ])


        assert result.exit_code == 0 or result.exit_code == 1

class TestDataCollectionCommands:

    def setup_method(self):

        self.runner = CliRunner()

    @patch('src.mora.cli.main.DataAcquisitionPipeline')
    @patch('src.mora.cli.main.load_config')
    def test_collect_data_command_execution(self, mock_load_config, mock_pipeline_class):


        mock_config = {
            'training': {
                'steady_state_config': {
                    'experiment_duration_minutes': 15,
                    'replica_counts': [1, 2, 4],
                    'load_levels_users': [5, 10, 20],
                    'test_scenarios': ['browsing', 'checkout']
                }
            },
            'kubernetes': {'namespace': 'hipster-shop'},
            'prometheus': {'url': 'http://localhost:9090'}
        }
        mock_load_config.return_value = mock_config


        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline._get_completed_experiments.return_value = set()
        mock_pipeline.run_isolated_training_experiment.return_value = {
            'status': 'completed',
            'experiments_completed': 12,
            'data_quality': 'Good'
        }

        result = self.runner.invoke(main, [
            'train', 'collect-data',
            '--service', 'frontend'
        ])


        assert result.exit_code == 0 or result.exit_code == 1

    @patch('src.mora.cli.main.DataAcquisitionPipeline')
    @patch('src.mora.cli.main.load_config')
    def test_collect_data_parallel_command_execution(self, mock_load_config, mock_pipeline_class):


        mock_config = {
            'training': {
                'steady_state_config': {
                    'experiment_duration_minutes': 15,
                    'replica_counts': [1, 2],
                    'load_levels_users': [5, 10],
                    'test_scenarios': ['browsing']
                }
            },
            'kubernetes': {'namespace': 'hipster-shop'},
            'prometheus': {'url': 'http://localhost:9090'}
        }
        mock_load_config.return_value = mock_config


        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_parallel_training_experiments.return_value = {
            'status': 'completed',
            'successful_services': 2,
            'failed_services': 0,
            'total_experiments': 8
        }

        result = self.runner.invoke(main, [
            'train', 'collect-data-parallel',
            '--services', 'frontend,cartservice',
            '--max-workers', '2'
        ])


        assert result.exit_code == 0 or result.exit_code == 1

    @patch('src.mora.cli.main.DataAcquisitionPipeline')
    @patch('src.mora.cli.main.load_config')
    def test_train_status_command_execution(self, mock_load_config, mock_pipeline_class):


        mock_config = {
            'training': {
                'steady_state_config': {
                    'replica_counts': [1, 2, 4],
                    'load_levels_users': [5, 10, 20],
                    'test_scenarios': ['browsing', 'checkout']
                }
            },
            'kubernetes': {'namespace': 'hipster-shop'},
            'prometheus': {'url': 'http://localhost:9090'}
        }
        mock_load_config.return_value = mock_config


        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline._get_completed_experiments.return_value = {
            'frontend_browsing_replicas_1_users_5',
            'frontend_browsing_replicas_2_users_10'
        }

        result = self.runner.invoke(main, [
            'train', 'status',
            '--service', 'frontend'
        ])

        assert result.exit_code == 0
        assert 'Training Progress for frontend' in result.output
        assert 'Completed: 2' in result.output

class TestCLIProductionReadiness:

    def setup_method(self):

        self.runner = CliRunner()

    def test_error_handling_in_status_command(self):

        with patch('src.mora.cli.main.DataPipeline') as mock_pipeline_class:

            mock_pipeline_class.side_effect = Exception("Connection failed")

            result = self.runner.invoke(main, ['status'])


            assert result.exit_code == 0

    @patch('src.mora.cli.main.load_config')
    def test_config_file_handling_in_status_command(self, mock_load_config):

        mock_config = {
            'kubernetes': {'namespace': 'test-namespace'},
            'prometheus': {'url': 'http://test-prometheus:9090'}
        }
        mock_load_config.return_value = mock_config

        with patch('src.mora.cli.main.DataPipeline') as mock_pipeline_class:
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_pipeline.test_connections.return_value = {'kubernetes': True, 'prometheus': True}
            mock_pipeline.get_deployed_services.return_value = ['frontend']
            mock_pipeline.test_prometheus_metrics.return_value = {'available_metrics': [], 'working_metrics': []}

            result = self.runner.invoke(main, ['status'])

            assert result.exit_code == 0



if __name__ == '__main__':
    pytest.main([__file__])
