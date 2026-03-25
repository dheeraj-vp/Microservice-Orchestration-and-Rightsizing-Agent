#!/usr/bin/env bash


set -e

NAMESPACE="monitoring"
DASHBOARD_NAME="hipster-shop-dashboard"
DASHBOARD_FILE="/home/dheeraj/KO/config/grafana/dashboards/hipster-shop-dashboard.json"

echo "=== Setting up Auto Grafana Dashboard ==="
echo


if [ ! -f "$DASHBOARD_FILE" ]; then
    echo "❌ Dashboard file not found: $DASHBOARD_FILE"
    exit 1
fi


echo "📊 Creating ConfigMap for dashboard..."
kubectl create configmap "$DASHBOARD_NAME" \
    --from-file="$DASHBOARD_FILE" \
    -n "$NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -


echo "🏷️  Labeling ConfigMap for Grafana auto-discovery..."
kubectl label configmap "$DASHBOARD_NAME" \
    grafana_dashboard=1 \
    -n "$NAMESPACE" \
    --overwrite


echo "⏳ Waiting for Grafana to discover dashboard..."
sleep 5


echo
echo "✅ Dashboard setup complete!"
echo
echo "📋 Next steps:"
echo "   1. Access Grafana at: http://localhost:4000"
echo "   2. Login with: admin / prom-operator"
echo "   3. Go to Dashboards → Browse"
echo "   4. Look for 'Hipster Shop - Auto Metrics Dashboard'"
echo
echo "🔄 The dashboard will automatically:"
echo "   - Display all Prometheus metrics from hipster-shop namespace"
echo "   - Auto-refresh every 30 seconds"
echo "   - Show CPU, Memory, Network I/O, Pod status, and more"
echo
echo "💡 Alternative: Use existing Kubernetes dashboards:"
echo "   - Kubernetes / Compute Resources / Namespace (Pods)"
echo "   - Kubernetes / Compute Resources / Namespace (Workloads)"
echo "   - Filter by namespace: hipster-shop"

