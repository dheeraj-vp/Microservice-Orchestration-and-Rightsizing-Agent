#!/usr/bin/env bash








set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PROM_NAMESPACE="monitoring"
HIPSTER_NAMESPACE="hipster-shop"
PROM_LOCAL_PORT=9090
GRAFANA_LOCAL_PORT=4000


GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo
echo -e "${BLUE}🚀 MOrA Infrastructure Restore${NC}"
echo "=================================="
echo


check_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo -e "${RED}❌ ERROR: Required command '$1' not found${NC}"
        echo "Please install it and re-run."
        exit 1
    fi
}


check_cmd minikube
check_cmd kubectl


kill_portforward() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "   Stopped old port-forward (PID: $pid)"
        fi
        rm -f "$pid_file"
    fi
}


check_portforward() {
    local pid_file="$1"
    local port="$2"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then

            if lsof -i ":$port" >/dev/null 2>&1 || netstat -tlnp 2>/dev/null | grep -q ":$port" || ss -tlnp 2>/dev/null | grep -q ":$port"; then
                return 0
            fi
        fi
    fi
    return 1
}


echo -e "${BLUE}📋 Checking infrastructure status...${NC}"
echo

MINIKUBE_RUNNING=false
MONITORING_EXISTS=false
HIPSTER_EXISTS=false
PROMETHEUS_READY=false
GRAFANA_READY=false


if minikube status &>/dev/null; then
    MINIKUBE_STATUS=$(minikube status --format='{{.Host}} {{.Kubelet}} {{.APIServer}}' 2>/dev/null || echo "")
    if echo "$MINIKUBE_STATUS" | grep -q "Running"; then
        MINIKUBE_RUNNING=true
        echo -e "${GREEN}✅ Minikube is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Minikube exists but not fully running${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Minikube not found or not running${NC}"
fi


if kubectl get namespace "$PROM_NAMESPACE" &>/dev/null 2>&1; then
    MONITORING_EXISTS=true
    echo -e "${GREEN}✅ Monitoring namespace exists${NC}"
else
    echo -e "${YELLOW}⚠️  Monitoring namespace not found${NC}"
fi

if kubectl get namespace "$HIPSTER_NAMESPACE" &>/dev/null 2>&1; then
    HIPSTER_EXISTS=true
    echo -e "${GREEN}✅ Hipster Shop namespace exists${NC}"
else
    echo -e "${YELLOW}⚠️  Hipster Shop namespace not found${NC}"
fi


if [ "$MONITORING_EXISTS" = true ]; then
    if kubectl -n "$PROM_NAMESPACE" get pod -l app.kubernetes.io/name=prometheus --no-headers 2>/dev/null | grep -q Running; then
        PROMETHEUS_READY=true
        echo -e "${GREEN}✅ Prometheus pod is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Prometheus pod not ready${NC}"
    fi
    
    if kubectl -n "$PROM_NAMESPACE" get pod -l app.kubernetes.io/name=grafana --no-headers 2>/dev/null | grep -q Running; then
        GRAFANA_READY=true
        echo -e "${GREEN}✅ Grafana pod is running${NC}"
    else
        echo -e "${YELLOW}⚠️  Grafana pod not ready${NC}"
    fi
fi

echo


QUICK_RESTORE=false

if [ "$MINIKUBE_RUNNING" = true ] && [ "$MONITORING_EXISTS" = true ] && [ "$HIPSTER_EXISTS" = true ] && [ "$PROMETHEUS_READY" = true ] && [ "$GRAFANA_READY" = true ]; then
    QUICK_RESTORE=true
    echo -e "${GREEN}✨ All components detected! Running quick restore...${NC}"
    echo
else
    echo -e "${YELLOW}📦 Some components missing. Running full setup...${NC}"
    echo
fi


if [ "$QUICK_RESTORE" = true ]; then
    echo -e "${BLUE}🔄 Restarting port-forwards...${NC}"
    

    kill_portforward "$SCRIPT_DIR/.prom_portforward.pid"
    kill_portforward "$SCRIPT_DIR/.grafana_portforward.pid"
    

    sleep 1
    

    echo "   Starting Prometheus port-forward..."
    kubectl -n "$PROM_NAMESPACE" port-forward svc/prometheus-kube-prometheus-prometheus \
        "$PROM_LOCAL_PORT:$PROM_LOCAL_PORT" > "$SCRIPT_DIR/.prom_portforward.log" 2>&1 &
    echo $! > "$SCRIPT_DIR/.prom_portforward.pid"
    

    echo "   Starting Grafana port-forward..."
    kubectl -n "$PROM_NAMESPACE" port-forward svc/prometheus-grafana \
        "$GRAFANA_LOCAL_PORT:80" > "$SCRIPT_DIR/.grafana_portforward.log" 2>&1 &
    echo $! > "$SCRIPT_DIR/.grafana_portforward.pid"
    

    echo "   Waiting for connections..."
    sleep 3
    

    PROM_OK=false
    GRAFANA_OK=false
    
    echo "   Verifying connections..."
    for i in {1..5}; do
        if curl -fsS "http://localhost:$PROM_LOCAL_PORT/-/ready" >/dev/null 2>&1; then
            PROM_OK=true
            break
        fi
        sleep 1
    done
    
    for i in {1..5}; do
        if curl -fsS "http://localhost:$GRAFANA_LOCAL_PORT/api/health" >/dev/null 2>&1; then
            GRAFANA_OK=true
            break
        fi
        sleep 1
    done
    
    echo
    if [ "$PROM_OK" = true ] && [ "$GRAFANA_OK" = true ]; then
        echo -e "${GREEN}✅ Quick restore complete!${NC}"
        echo
        echo -e "${BLUE}📊 Access Information:${NC}"
        echo "   Prometheus: http://localhost:$PROM_LOCAL_PORT"
        echo "   Grafana: http://localhost:$GRAFANA_LOCAL_PORT"
        

        GRAFANA_PASS=$(kubectl -n "$PROM_NAMESPACE" get secret prometheus-grafana \
            -o jsonpath="{.data.admin-password}" 2>/dev/null | base64 -d 2>/dev/null || echo "admin")
        echo "   Grafana login: admin / $GRAFANA_PASS"
        echo
        

        if ! kubectl -n "$PROM_NAMESPACE" get configmap hipster-shop-dashboard &>/dev/null 2>&1; then
            echo -e "${BLUE}📊 Setting up Grafana dashboard...${NC}"
            "$SCRIPT_DIR/setup_auto_grafana_dashboard.sh" 2>/dev/null || true
        fi
        
        echo
        echo -e "${GREEN}✅ All done! Infrastructure is ready.${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Port-forwards started but verification failed${NC}"
        echo "   Falling back to full setup to ensure everything works..."
        echo
    fi
fi


echo -e "${BLUE}📦 Running full infrastructure setup...${NC}"
echo "   This may take a few minutes..."
echo


"$SCRIPT_DIR/setup-minikube.sh"


echo
echo -e "${BLUE}📊 Setting up Grafana dashboard...${NC}"
"$SCRIPT_DIR/setup_auto_grafana_dashboard.sh" 2>/dev/null || echo "   Dashboard will be auto-discovered by Grafana"

echo
echo -e "${GREEN}✅ Infrastructure restore complete!${NC}"
echo
echo -e "${BLUE}📊 Quick Access:${NC}"
echo "   Prometheus: http://localhost:$PROM_LOCAL_PORT"
echo "   Grafana: http://localhost:$GRAFANA_LOCAL_PORT"
echo
echo -e "${BLUE}💡 Tips:${NC}"
echo "   - Run this script anytime after shutdown: ./scripts/restore-infra.sh"
echo "   - Check status: ./scripts/verify-setup.sh"
echo "   - Stop port-forwards: kill \$(cat scripts/.prom_portforward.pid scripts/.grafana_portforward.pid)"
echo

exit 0

