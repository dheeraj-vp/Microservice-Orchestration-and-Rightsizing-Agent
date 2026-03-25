#!/usr/bin/env python3

import json
import requests
from typing import Dict, Any

GRAFANA_URL = "http://localhost:4000"
GRAFANA_USER = "admin"
GRAFANA_PASSWORD = "prom-operator"
NAMESPACE = "hipster-shop"

def create_comprehensive_dashboard() -> str:

    dashboard = {
        "dashboard": {
            "id": None,
            "uid": "hipster-shop-auto",
            "title": "Hipster Shop - Auto Metrics Dashboard",
            "tags": ["hipster-shop", "auto", "prometheus", "kubernetes"],
            "timezone": "browser",
            "refresh": "30s",
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": [

                {
                    "id": 1,
                    "title": "CPU Usage by Pod (5m avg)",
                    "type": "timeseries",
                    "targets": [{
                        "expr": f'avg(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}", container!="POD", container!=""}}[5m])) by (pod)',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "{{pod}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "short",
                            "color": {"mode": "palette-classic"},
                            "custom": {"drawStyle": "line", "fillOpacity": 10}
                        }
                    }
                },
                {
                    "id": 2,
                    "title": "CPU Usage by Service",
                    "type": "timeseries",
                    "targets": [{
                        "expr": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}", container!="POD", container!=""}}[5m])) by (container)',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "{{container}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "short",
                            "color": {"mode": "palette-classic"}
                        }
                    }
                },

                {
                    "id": 3,
                    "title": "Memory Usage by Pod",
                    "type": "timeseries",
                    "targets": [{
                        "expr": f'avg(container_memory_working_set_bytes{{namespace="{NAMESPACE}", container!="POD", container!=""}}) by (pod)',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "{{pod}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "bytes",
                            "color": {"mode": "palette-classic"}
                        }
                    }
                },
                {
                    "id": 4,
                    "title": "Memory Usage by Service",
                    "type": "timeseries",
                    "targets": [{
                        "expr": f'sum(container_memory_working_set_bytes{{namespace="{NAMESPACE}", container!="POD", container!=""}}) by (container)',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "{{container}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "bytes",
                            "color": {"mode": "palette-classic"}
                        }
                    }
                },

                {
                    "id": 5,
                    "title": "Network Receive Bytes",
                    "type": "timeseries",
                    "targets": [{
                        "expr": f'rate(container_network_receive_bytes_total{{namespace="{NAMESPACE}"}}[5m])',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "RX {{pod}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "binBps",
                            "color": {"mode": "palette-classic"}
                        }
                    }
                },
                {
                    "id": 6,
                    "title": "Network Transmit Bytes",
                    "type": "timeseries",
                    "targets": [{
                        "expr": f'rate(container_network_transmit_bytes_total{{namespace="{NAMESPACE}"}}[5m])',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "TX {{pod}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "binBps",
                            "color": {"mode": "palette-classic"}
                        }
                    }
                },

                {
                    "id": 7,
                    "title": "Pod Status",
                    "type": "stat",
                    "targets": [{
                        "expr": f'kube_pod_status_phase{{namespace="{NAMESPACE}"}}',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "{{pod}} - {{phase}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 6, "w": 8, "x": 0, "y": 24},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "short",
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"value": 0, "color": "red"},
                                    {"value": 1, "color": "green"}
                                ]
                            }
                        }
                    }
                },
                {
                    "id": 8,
                    "title": "Pod Replicas by Deployment",
                    "type": "stat",
                    "targets": [{
                        "expr": f'kube_deployment_status_replicas{{namespace="{NAMESPACE}"}}',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "{{deployment}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 6, "w": 8, "x": 8, "y": 24},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "short",
                            "color": {"mode": "thresholds"}
                        }
                    }
                },
                {
                    "id": 9,
                    "title": "Container Restarts",
                    "type": "stat",
                    "targets": [{
                        "expr": f'kube_pod_container_status_restarts_total{{namespace="{NAMESPACE}"}}',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "{{pod}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 6, "w": 8, "x": 16, "y": 24},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "short",
                            "color": {"mode": "thresholds"}
                        }
                    }
                },

                {
                    "id": 10,
                    "title": "File System Reads",
                    "type": "timeseries",
                    "targets": [{
                        "expr": f'rate(container_fs_reads_bytes_total{{namespace="{NAMESPACE}"}}[5m])',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "{{pod}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 30},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "binBps",
                            "color": {"mode": "palette-classic"}
                        }
                    }
                },
                {
                    "id": 11,
                    "title": "File System Writes",
                    "type": "timeseries",
                    "targets": [{
                        "expr": f'rate(container_fs_writes_bytes_total{{namespace="{NAMESPACE}"}}[5m])',
                        "datasource": {"type": "prometheus", "uid": "prometheus"},
                        "legendFormat": "{{pod}}",
                        "refId": "A"
                    }],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 30},
                    "fieldConfig": {
                        "defaults": {
                            "unit": "binBps",
                            "color": {"mode": "palette-classic"}
                        }
                    }
                }
            ]
        },
        "overwrite": True
    }


    session = requests.Session()
    session.auth = (GRAFANA_USER, GRAFANA_PASSWORD)

    try:

        ds_response = session.get(f"{GRAFANA_URL}/api/datasources/name/Prometheus")
        if ds_response.status_code == 200:
            prom_uid = ds_response.json().get("uid", "prometheus")

            for panel in dashboard["dashboard"]["panels"]:
                for target in panel.get("targets", []):
                    if "datasource" in target:
                        target["datasource"]["uid"] = prom_uid


        response = session.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            json=dashboard,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        return result.get("uid", "unknown")
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        return None

if __name__ == "__main__":
    print("Creating comprehensive Hipster Shop dashboard...")
    uid = create_comprehensive_dashboard()
    if uid:
        print(f"✅ Dashboard created successfully!")
        print(f"   UID: {uid}")
        print(f"   URL: http://localhost:4000/d/{uid}")
        print(f"\n📊 Dashboard automatically displays:")
        print(f"   - CPU usage by pod and service")
        print(f"   - Memory usage by pod and service")
        print(f"   - Network I/O (receive/transmit)")
        print(f"   - Pod status and replicas")
        print(f"   - Container restarts")
        print(f"   - File system I/O")
        print(f"\n🔄 Auto-refresh: Every 30 seconds")
        print(f"📈 Time range: Last 1 hour (adjustable)")
    else:
        print("❌ Failed to create dashboard")
