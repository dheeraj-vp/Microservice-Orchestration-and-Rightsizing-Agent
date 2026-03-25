
import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:

    if config_path is None:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        config_path = os.path.join(project_root, "config", "default.yaml")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config or {}
    except FileNotFoundError:

        return {
            'kubernetes': {
                'namespace': 'hipster-shop',
                'context': 'minikube'
            },
            'prometheus': {
                'url': 'http://localhost:9090',
                'timeout': 30,
                'retry_attempts': 3
            },
            'grafana': {
                'url': 'http://localhost:4000',
                'admin_user': 'admin',
                'timeout': 30
            }
        }
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}
