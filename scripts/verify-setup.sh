#!/usr/bin/env bash
set -euo pipefail






PROM_NAMESPACE="monitoring"
HIPSTER_NAMESPACE="hipster-shop"
PROM_LOCAL_PORT=9090
GRAFANA_LOCAL_PORT=4000

echo
echo "=== MOrA Setup Verification ==="
echo


echo "1. Checking kubectl connectivity..."
if kubectl cluster-info >/dev/null 2>&1; then
    echo "✅ kubectl can access the cluster"
    kubectl config current-context
else
    echo "❌ kubectl cannot access cluster - check minikube status"
    exit 1
fi


echo
echo "2. Checking Prometheus stack..."
if kubectl get namespace "${PROM_NAMESPACE}" >/dev/null 2>&1; then
    echo "✅ Monitoring namespace exists"
    
    PROM_PODS=$(kubectl -n "${PROM_NAMESPACE}" get pods --no-headers | wc -l)
    echo "   Found ${PROM_PODS} pods in monitoring namespace"
    
    if kubectl -n "${PROM_NAMESPACE}" get pods -l app.kubernetes.io/name=prometheus --no-headers | grep -q Running; then
        echo "✅ Prometheus server is running"
    else
        echo "⚠️  Prometheus server may not be ready - check with: kubectl -n ${PROM_NAMESPACE} get pods"
    fi
else
    echo "❌ Monitoring namespace not found - run setup script first"
    exit 1
fi


echo
echo "3. Checking Hipster Shop deployment..."
if kubectl get namespace "${HIPSTER_NAMESPACE}" >/dev/null 2>&1; then
    echo "✅ Hipster Shop namespace exists"
    
    FRONTEND_READY=$(kubectl -n "${HIPSTER_NAMESPACE}" get deployment frontend -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    FRONTEND_DESIRED=$(kubectl -n "${HIPSTER_NAMESPACE}" get deployment frontend -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    
    if [[ "${FRONTEND_READY}" == "${FRONTEND_DESIRED}" && "${FRONTEND_READY}" != "0" ]]; then
        echo "✅ Frontend service is ready (${FRONTEND_READY}/${FRONTEND_DESIRED} replicas)"
    else
        echo "⚠️  Frontend service may not be ready (${FRONTEND_READY}/${FRONTEND_DESIRED} replicas)"
    fi
    
    TOTAL_PODS=$(kubectl -n "${HIPSTER_NAMESPACE}" get pods --no-headers | wc -l)
    echo "   Found ${TOTAL_PODS} pods in hipster-shop namespace"
else
    echo "❌ Hipster Shop namespace not found - run setup script first"
    exit 1
fi


echo
echo "4. Checking Prometheus port-forward..."
if [[ -f ".prom_portforward.pid" ]]; then
    PID=$(cat .prom_portforward.pid)
    if kill -0 "${PID}" 2>/dev/null; then
        echo "✅ Prometheus port-forward is running (PID: ${PID})"
        

        if curl -fsS "http://localhost:${PROM_LOCAL_PORT}/-/ready" >/dev/null 2>&1; then
            echo "✅ Prometheus is accessible at http://localhost:${PROM_LOCAL_PORT}"
        else
            echo "⚠️  Prometheus endpoint not responding - check .prom_portforward.log"
        fi
    else
        echo "❌ Port-forward PID file exists but process not running"
    fi
else
    echo "❌ No port-forward detected - Prometheus may not be accessible locally"
fi


echo
echo "5. Checking Grafana port-forward..."
if [[ -f ".grafana_portforward.pid" ]]; then
    PID=$(cat .grafana_portforward.pid)
    if kill -0 "${PID}" 2>/dev/null; then
        echo "✅ Grafana port-forward is running (PID: ${PID})"
        

        if curl -fsS "http://localhost:${GRAFANA_LOCAL_PORT}/api/health" >/dev/null 2>&1; then
            echo "✅ Grafana is accessible at http://localhost:${GRAFANA_LOCAL_PORT}"
        else
            echo "⚠️  Grafana endpoint not responding - check .grafana_portforward.log"
        fi
    else
        echo "❌ Grafana port-forward PID file exists but process not running"
    fi
else
    echo "❌ No Grafana port-forward detected"
fi


echo
echo "=== Verification Summary ==="
echo "If all checks passed, you can now:"
echo "1. Access Prometheus at: http://localhost:${PROM_LOCAL_PORT}"
echo "2. Access Grafana at: http://localhost:${GRAFANA_LOCAL_PORT}"
echo "3. Run MOrA CLI commands:"
echo "   mora status --namespace ${HIPSTER_NAMESPACE}"
echo "   mora rightsize --service frontend --namespace ${HIPSTER_NAMESPACE}"
echo

exit 0

