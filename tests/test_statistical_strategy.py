
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.mora.core.statistical_strategy import StatisticalRightsizer

class TestStatisticalRightsizer:

    def setup_method(self):

        self.rightsizer = StatisticalRightsizer(cpu_percentile=95.0, memory_buffer_percentage=15.0)

    def test_initialization(self):

        assert self.rightsizer.cpu_percentile == 95.0
        assert self.rightsizer.memory_buffer_percentage == 15.0

    def test_parse_cpu_value(self):


        assert self.rightsizer.parse_cpu_value("100m") == 0.1
        assert self.rightsizer.parse_cpu_value("500m") == 0.5


        assert self.rightsizer.parse_cpu_value("1") == 1.0
        assert self.rightsizer.parse_cpu_value("2.5") == 2.5


        assert self.rightsizer.parse_cpu_value("Unknown") == 0.0
        assert self.rightsizer.parse_cpu_value("") == 0.0
        assert self.rightsizer.parse_cpu_value(None) == 0.0

    def test_parse_memory_value(self):


        assert self.rightsizer.parse_memory_value("1024Ki") == 1024 * 1024


        assert self.rightsizer.parse_memory_value("128Mi") == 128 * 1024 * 1024


        assert self.rightsizer.parse_memory_value("1Gi") == 1024 * 1024 * 1024


        assert self.rightsizer.parse_memory_value("Unknown") == 0
        assert self.rightsizer.parse_memory_value("") == 0
        assert self.rightsizer.parse_memory_value(None) == 0

    def test_format_cpu_value(self):


        assert self.rightsizer.format_cpu_value(0.1) == "100m"
        assert self.rightsizer.format_cpu_value(0.5) == "500m"


        result = self.rightsizer.format_cpu_value(1.0)
        assert result in ["1", "1.0"]

        result = self.rightsizer.format_cpu_value(2.5)
        assert result in ["2.5", "2.500"]

    def test_format_memory_value(self):


        result = self.rightsizer.format_memory_value(1024 * 1024 * 1024)
        assert "Gi" in result


        result = self.rightsizer.format_memory_value(128 * 1024 * 1024)
        assert "Mi" in result


        result = self.rightsizer.format_memory_value(1024)
        assert "Ki" in result

    def test_analyze_cpu_usage_with_data(self):


        timestamps = pd.date_range('2023-01-01', periods=100, freq='1min')
        cpu_values = np.random.uniform(0.1, 0.8, 100)

        cpu_df = pd.DataFrame({
            'timestamp': timestamps,
            '__value': cpu_values
        })

        result = self.rightsizer.analyze_cpu_usage(cpu_df)

        assert result['has_data'] == True
        assert 'recommended_requests' in result
        assert 'current_usage_stats' in result
        assert result['current_usage_stats']['count'] == 100

    def test_analyze_cpu_usage_empty_data(self):

        empty_df = pd.DataFrame()

        result = self.rightsizer.analyze_cpu_usage(empty_df)

        assert result['has_data'] == False
        assert result['recommended_requests'] == "100m"
        assert result['current_usage_stats'] == {}

    def test_analyze_memory_usage_with_data(self):


        timestamps = pd.date_range('2023-01-01', periods=100, freq='1min')
        memory_values = np.random.uniform(100 * 1024 * 1024, 500 * 1024 * 1024, 100)

        memory_df = pd.DataFrame({
            'timestamp': timestamps,
            '__value': memory_values
        })

        result = self.rightsizer.analyze_memory_usage(memory_df)

        assert result['has_data'] == True
        assert 'recommended_requests' in result
        assert 'max_usage_bytes' in result
        assert result['current_usage_stats']['count'] == 100
        assert result['buffer_percentage'] == 15.0

    def test_analyze_memory_usage_empty_data(self):

        empty_df = pd.DataFrame()

        result = self.rightsizer.analyze_memory_usage(empty_df)

        assert result['has_data'] == False
        assert "Mi" in result['recommended_requests']

    def test_generate_recommendations(self):


        service_data = {
            'service_name': 'test-service',
            'namespace': 'default',
            'deployment': {
                'name': 'test-service',
                'containers': [{
                    'name': 'test-container',
                    'resources': {
                        'requests': {'cpu': '100m', 'memory': '128Mi'}
                    }
                }]
            },
            'metrics': {
                'cpu': pd.DataFrame({'__value': [0.1, 0.2, 0.3]}),
                'memory': pd.DataFrame({'__value': [100 * 1024 * 1024, 150 * 1024 * 1024]})
            }
        }

        recommendations = self.rightsizer.generate_recommendations(service_data)

        assert len(recommendations) == 1
        rec = recommendations[0]

        assert rec['service_name'] == 'test-service'
        assert rec['container_name'] == 'test-container'
        assert rec['strategy'] == 'statistical'
        assert 'current_requests' in rec
        assert 'recommended_requests' in rec
        assert 'analysis' in rec

    def test_generate_recommendations_no_deployment(self):

        service_data = {
            'service_name': 'test-service',
            'namespace': 'default'
        }

        recommendations = self.rightsizer.generate_recommendations(service_data)

        assert recommendations == []

    def test_validate_recommendations(self):

        recommendations = [
            {
                'container_name': 'test-container',
                'recommended_requests': {
                    'cpu': '500m',
                    'memory': '256Mi'
                }
            }
        ]

        validation = self.rightsizer.validate_recommendations(recommendations)

        assert validation['is_valid'] == True
        assert 'warnings' in validation
        assert 'errors' in validation

    def test_validate_recommendations_low_cpu_warning(self):

        recommendations = [
            {
                'container_name': 'test-container',
                'recommended_requests': {
                    'cpu': '5m',
                    'memory': '256Mi'
                }
            }
        ]

        validation = self.rightsizer.validate_recommendations(recommendations)

        assert validation['is_valid'] == True
        assert len(validation['warnings']) > 0
        assert 'low CPU recommendation' in validation['warnings'][0]

class TestStatisticalRightsizerIntegration:

    def setup_method(self):

        self.rightsizer = StatisticalRightsizer()

    def test_end_to_end_analysis(self):


        np.random.seed(42)


        cpu_data = np.random.normal(0.3, 0.1, 1440)
        cpu_data = np.clip(cpu_data, 0.05, 0.9)

        cpu_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1440, freq='1min'),
            '__value': cpu_data
        })


        memory_data = np.random.normal(200 * 1024 * 1024, 50 * 1024 * 1024, 1440)
        memory_data = np.clip(memory_data, 100 * 1024 * 1024, 500 * 1024 * 1024)

        memory_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1440, freq='1min'),
            '__value': memory_data
        })


        cpu_result = self.rightsizer.analyze_cpu_usage(cpu_df)
        memory_result = self.rightsizer.analyze_memory_usage(memory_df)


        assert cpu_result['has_data'] == True
        assert memory_result['has_data'] == True


        cpu_recommended = self.rightsizer.parse_cpu_value(cpu_result['recommended_requests'])
        assert 0.1 <= cpu_recommended <= 1.0


        memory_recommended = self.rightsizer.parse_memory_value(memory_result['recommended_requests'])
        max_usage = memory_result['max_usage_bytes']
        buffer_factor = 1.15
        expected_min = max_usage * buffer_factor
        assert memory_recommended >= expected_min * 0.9
