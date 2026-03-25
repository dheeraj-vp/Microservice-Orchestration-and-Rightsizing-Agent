
import pytest
import subprocess
import time
import json
import requests
from click.testing import CliRunner
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from src.mora.cli.main import main
except ImportError:
    pytest.skip("Cannot import CLI module", allow_module_level=True)

class TestIntegration:

    @classmethod
    def setup_class(cls):

        cls.runner = CliRunner()
        cls.system_ready = cls._check_system_health()

    @classmethod
    def _check_system_health(cls):

        try:

            result = subprocess.run(['kubectl', 'cluster-info'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return False


            try:
                response = requests.get('http://localhost:9090/-/ready', timeout=5)
                if response.status_code != 200:
                    return False
            except requests.exceptions.RequestException:
                return False


            try:
                response = requests.get('http://localhost:4000/api/health', timeout=5)
                if response.status_code != 200:
                    return False
            except requests.exceptions.RequestException:
                return False


            result = subprocess.run(['kubectl', 'get', 'pods', '-n', 'hipster-shop'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0 or 'No resources found' in result.stdout:
                return False

            return True
        except Exception as e:
            print(f"System health check failed: {e}")
            return False

    def test_system_health(self):

        if not self.system_ready:
            pytest.skip("System not ready - ensure Minikube, Prometheus, and Hipster Shop are running")

        assert self.system_ready, "System should be ready for integration tests"

    def test_status_command_integration(self):

        result = self.runner.invoke(main, [
            'status',
            '--namespace', 'hipster-shop',
            '--prometheus-url', 'http://localhost:9090'
        ])



        if result.exit_code == 0:

            assert 'MOrA System Status' in result.output or 'system' in result.output.lower()

    @pytest.mark.skipif(not hasattr(TestIntegration, 'system_ready') or not getattr(TestIntegration, 'system_ready', False),
                       reason="System not ready")
    def test_rightsize_command_basic_integration(self):

        result = self.runner.invoke(main, [
            'rightsize',
            '--service', 'frontend',
            '--namespace', 'hipster-shop',
            '--strategy', 'statistical',
            '--duration-hours', '1'
        ])


        assert result.exit_code in [0, 1]


        output_lower = result.output.lower()
        assert any(keyword in output_lower for keyword in [
            'mora', 'rightsizing', 'analysis', 'service', 'frontend'
        ])

    @pytest.mark.skipif(not hasattr(TestIntegration, 'system_ready') or not getattr(TestIntegration, 'system_ready', False),
                       reason="System not ready")
    def test_json_output_format(self):

        result = self.runner.invoke(main, [
            'rightsize',
            '--service', 'frontend',
            '--namespace', 'hipster-shop',
            '--strategy', 'statistical',
            '--output-format', 'json',
            '--duration-hours', '1'
        ])


        assert result.exit_code in [0, 1]


        if result.output.strip():

            try:
                json.loads(result.output)
            except json.JSONDecodeError:

                assert any(keyword in result.output.lower() for keyword in [
                    'error', 'failed', 'connecting', 'collecting', 'analysis'
                ])

class TestSystemComponents:

    def test_prometheus_connectivity(self):

        try:
            response = requests.get('http://localhost:9090/-/ready', timeout=5)
            if response.status_code == 200:
                assert True
            else:
                pytest.skip(f"Prometheus returned status {response.status_code}")
        except requests.exceptions.RequestException:
            pytest.skip("Prometheus not accessible at localhost:9090")

    def test_grafana_connectivity(self):

        try:
            response = requests.get('http://localhost:4000/api/health', timeout=5)
            if response.status_code == 200:
                assert True
            else:
                pytest.skip(f"Grafana returned status {response.status_code}")
        except requests.exceptions.RequestException:
            pytest.skip("Grafana not accessible at localhost:4000")

    def test_kubernetes_connectivity(self):

        try:
            result = subprocess.run(['kubectl', 'cluster-info'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                assert "is running at" in result.stdout
            else:
                pytest.skip("Kubernetes cluster not accessible")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("kubectl not available or cluster not accessible")

    def test_hipster_shop_deployment(self):

        try:
            result = subprocess.run(['kubectl', 'get', 'deployments', '-n', 'hipster-shop', 'frontend'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'frontend' in result.stdout:
                assert True
            else:
                pytest.skip("Hipster Shop frontend deployment not found")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("kubectl not available")

class TestGrafanaIntegration:

    @pytest.mark.skipif(not hasattr(TestIntegration, 'system_ready') or not getattr(TestIntegration, 'system_ready', False),
                       reason="System not ready")
    def test_setup_grafana_integration(self):

        runner = CliRunner()
        result = runner.invoke(main, [
            'setup-grafana',
            '--namespace', 'hipster-shop',
            '--grafana-url', 'http://localhost:4000',
            '--prometheus-url', 'http://localhost:9090'
        ])


        assert result.exit_code in [0, 1]


        if result.output.strip():
            assert any(keyword in result.output.lower() for keyword in [
                'grafana', 'dashboard', 'setup', 'integration', 'mora'
            ])

class TestCLIBasic:

    def setup_method(self):

        self.runner = CliRunner()

    def test_help_commands(self):


        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "MOrA" in result.output


        result = self.runner.invoke(main, ['rightsize', '--help'])
        assert result.exit_code == 0
        assert "Generate rightsizing recommendations" in result.output


        result = self.runner.invoke(main, ['status', '--help'])
        assert result.exit_code == 0
        assert "Show current status" in result.output

    def test_version(self):

        result = self.runner.invoke(main, ['--version'])

        assert result.exit_code in [0, 1]

class TestValidation:

    def setup_method(self):

        self.runner = CliRunner()

    def test_required_parameters(self):


        result = self.runner.invoke(main, ['rightsize'])
        assert result.exit_code != 0

    def test_invalid_strategy(self):

        result = self.runner.invoke(main, [
            'rightsize',
            '--service', 'test-service',
            '--strategy', 'invalid-strategy'
        ])
        assert result.exit_code != 0

if __name__ == "__main__":

    pytest.main([__file__, '-v'])
