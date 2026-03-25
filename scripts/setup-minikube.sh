#!/usr/bin/env bash
set -euo pipefail









PROM_RELEASE_NAME="prometheus"
PROM_NAMESPACE="monitoring"
HIPSTER_NAMESPACE="hipster-shop"
PROM_LOCAL_PORT=9090
GRAFANA_LOCAL_PORT=4000
HIPSTER_MANIFEST="https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/master/release/kubernetes-manifests.yaml"
PORTFORWARD_PID_FILE=".prom_portforward.pid"
GRAFANA_PID_FILE=".grafana_portforward.pid"
HELM_REPO_NAME="prometheus-community"
HELM_REPO_URL="https://prometheus-community.github.io/helm-charts"
HELM_CHART="kube-prometheus-stack"
HELM_TIMEOUT="10m"

echo
echo "=== MOrA quick setup ==="
echo


check_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command '$1' not found in PATH. Install it and re-run."
    exit 2
  fi
}

check_cmd minikube
check_cmd kubectl
check_cmd helm
check_cmd curl || check_cmd wget


echo
echo "=> Checking Minikube status..."
MINI_STATUS="$(minikube status --format='{{.Host}} {{.Kubelet}} {{.APIServer}} {{.KubeConfig}}' 2>/dev/null || true)"
if [[ -z "$MINI_STATUS" ]]; then
  echo "Minikube not running. Starting minikube with sufficient resources (this may take a few minutes)..."
  minikube start --memory=8192 --cpus=4 --disk-size=20GB
else
  echo "Minikube status: $MINI_STATUS"

  if echo "$MINI_STATUS" | grep -q "Stopped"; then
    echo "Minikube API server appears stopped — restarting minikube..."
    minikube stop || true
    minikube start --memory=8192 --cpus=4 --disk-size=20GB
  fi
fi


echo
echo "=> Ensuring kubectl is pointed at minikube..."
kubectl cluster-info >/dev/null 2>&1 || {
  echo "kubectl can't access cluster. Ensure minikube started successfully."
  exit 3
}
kubectl config use-context minikube >/dev/null 2>&1 || true


echo
echo "=> Enabling required addons..."
minikube addons enable ingress >/dev/null 2>&1 || true
minikube addons enable metrics-server >/dev/null 2>&1 || true


echo
echo "=> Adding/updating Helm repo ${HELM_REPO_NAME}..."
helm repo add "${HELM_REPO_NAME}" "${HELM_REPO_URL}" >/dev/null 2>&1 || true
helm repo update >/dev/null

echo
echo "=> Installing/upgrading Prometheus (Helm chart: ${HELM_REPO_NAME}/${HELM_CHART}) into namespace '${PROM_NAMESPACE}'..."
helm upgrade --install "${PROM_RELEASE_NAME}" "${HELM_REPO_NAME}/${HELM_CHART}" \
  --namespace "${PROM_NAMESPACE}" --create-namespace \
  --wait --timeout "${HELM_TIMEOUT}" || {
    echo "Helm install/upgrade failed. Check 'kubectl -n ${PROM_NAMESPACE} get pods' for details."
    exit 4
}

echo "Prometheus helm chart installed/upgraded."


echo
echo "=> Creating namespace for Hipster Shop..."
kubectl create namespace "${HIPSTER_NAMESPACE}" >/dev/null 2>&1 || true

echo
echo "=> Deploying sample app (microservices-demo / Hipster Shop) to namespace '${HIPSTER_NAMESPACE}'..."
if kubectl get deployment frontend -n "${HIPSTER_NAMESPACE}" >/dev/null 2>&1; then
  echo "Found existing 'frontend' deployment in namespace '${HIPSTER_NAMESPACE}' — skipping sample app apply."
else
  echo "Applying hipster shop manifest (this may take a few minutes to pull images)..."

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${HIPSTER_MANIFEST}" | kubectl apply -f - -n "${HIPSTER_NAMESPACE}"
  else
    wget -qO- "${HIPSTER_MANIFEST}" | kubectl apply -f - -n "${HIPSTER_NAMESPACE}"
  fi
fi


echo
echo "=> Waiting for Prometheus server pod to be ready in namespace '${PROM_NAMESPACE}'..."

kubectl -n "${PROM_NAMESPACE}" wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus --timeout=5m || {
  echo "Warning: Prometheus server pods did not become ready in 5m. You can check with: kubectl -n ${PROM_NAMESPACE} get pods"
}

echo
echo "=> Waiting for 'frontend' deployment in '${HIPSTER_NAMESPACE}' namespace to be ready..."
if kubectl -n "${HIPSTER_NAMESPACE}" get deploy frontend >/dev/null 2>&1; then
  kubectl -n "${HIPSTER_NAMESPACE}" rollout status deploy/frontend --timeout=5m || {
    echo "Warning: frontend rollout did not finish within 5m. Check pods with 'kubectl -n ${HIPSTER_NAMESPACE} get pods'."
  }
else
  echo "No 'frontend' deployment found in ${HIPSTER_NAMESPACE} namespace (the manifest may have a different name)."
fi


echo
echo "=> Locating Prometheus service in namespace '${PROM_NAMESPACE}'..."
PROM_SVC="$(kubectl -n "${PROM_NAMESPACE}" get svc -l app.kubernetes.io/name=prometheus -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -z "${PROM_SVC}" ]]; then

  PROM_SVC="${PROM_RELEASE_NAME}-kube-prometheus-prometheus"
  echo "Couldn't find service by label; falling back to service name '${PROM_SVC}'."
fi

echo "Prometheus service: ${PROM_SVC}"


echo
if [[ -f "${PORTFORWARD_PID_FILE}" ]]; then
  OLDPID="$(cat "${PORTFORWARD_PID_FILE}" || true)"
  if [[ -n "${OLDPID}" ]] && kill -0 "${OLDPID}" 2>/dev/null; then
    echo "Prometheus port-forward already running with PID ${OLDPID}."
  else
    echo "Found stale PID file, removing."
    rm -f "${PORTFORWARD_PID_FILE}"
  fi
fi

if [[ ! -f "${PORTFORWARD_PID_FILE}" ]]; then
  echo "Starting port-forward: localhost:${PROM_LOCAL_PORT} -> svc/${PROM_SVC}:${PROM_LOCAL_PORT} (namespace ${PROM_NAMESPACE})"

  nohup kubectl -n "${PROM_NAMESPACE}" port-forward "svc/${PROM_SVC}" ${PROM_LOCAL_PORT}:${PROM_LOCAL_PORT} \
    > .prom_portforward.log 2>&1 &
  PF_PID=$!
  echo "${PF_PID}" > "${PORTFORWARD_PID_FILE}"
  echo "Port-forward started (PID ${PF_PID}). Logs: .prom_portforward.log"

  sleep 2
fi


echo
echo "=> Verifying Prometheus HTTP endpoint on http://localhost:${PROM_LOCAL_PORT}/-/ready ..."
if curl -fsS "http://localhost:${PROM_LOCAL_PORT}/-/ready" >/dev/null 2>&1; then
  echo "Prometheus ready at http://localhost:${PROM_LOCAL_PORT}"
else
  echo "Prometheus /-/ready did not respond immediately. Check .prom_portforward.log and pod logs: kubectl -n ${PROM_NAMESPACE} logs -l app.kubernetes.io/name=prometheus --tail=200"
fi


echo
echo "=> Locating Grafana service in namespace '${PROM_NAMESPACE}'..."
GRAFANA_SVC="$(kubectl -n "${PROM_NAMESPACE}" get svc -l app.kubernetes.io/name=grafana -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
if [[ -z "${GRAFANA_SVC}" ]]; then

  GRAFANA_SVC="${PROM_RELEASE_NAME}-grafana"
  echo "Couldn't find Grafana service by label; falling back to service name '${GRAFANA_SVC}'."
fi

echo "Grafana service: ${GRAFANA_SVC}"


echo
if [[ -f "${GRAFANA_PID_FILE}" ]]; then
  OLDPID="$(cat "${GRAFANA_PID_FILE}" || true)"
  if [[ -n "${OLDPID}" ]] && kill -0 "${OLDPID}" 2>/dev/null; then
    echo "Grafana port-forward already running with PID ${OLDPID}."
  else
    echo "Found stale Grafana PID file, removing."
    rm -f "${GRAFANA_PID_FILE}"
  fi
fi

if [[ ! -f "${GRAFANA_PID_FILE}" ]]; then
  echo "Starting Grafana port-forward: localhost:${GRAFANA_LOCAL_PORT} -> svc/${GRAFANA_SVC}:80 (namespace ${PROM_NAMESPACE})"

  nohup kubectl -n "${PROM_NAMESPACE}" port-forward "svc/${GRAFANA_SVC}" ${GRAFANA_LOCAL_PORT}:80 \
    > .grafana_portforward.log 2>&1 &
  GF_PID=$!
  echo "${GF_PID}" > "${GRAFANA_PID_FILE}"
  echo "Grafana port-forward started (PID ${GF_PID}). Logs: .grafana_portforward.log"

  sleep 2
fi


echo
echo "=> Verifying Grafana HTTP endpoint on http://localhost:${GRAFANA_LOCAL_PORT} ..."
if curl -fsS "http://localhost:${GRAFANA_LOCAL_PORT}/api/health" >/dev/null 2>&1; then
  echo "Grafana ready at http://localhost:${GRAFANA_LOCAL_PORT}"
  echo "Grafana login: admin / $(kubectl -n ${PROM_NAMESPACE} get secrets ${GRAFANA_SVC} -o jsonpath='{.data.admin-password}' | base64 -d 2>/dev/null || echo 'admin')"
else
  echo "Grafana /api/health did not respond immediately. Check .grafana_portforward.log and pod logs: kubectl -n ${PROM_NAMESPACE} logs -l app.kubernetes.io/name=grafana --tail=200"
fi


echo
echo "=== MOrA Setup complete ==="
echo "Prometheus is reachable at: http://localhost:${PROM_LOCAL_PORT}"
echo "Grafana is reachable at: http://localhost:${GRAFANA_LOCAL_PORT}"
echo "Hipster Shop deployed in namespace: ${HIPSTER_NAMESPACE}"
echo
echo "📊 Current service status:"
kubectl get pods -n "${HIPSTER_NAMESPACE}"
echo
echo "🔧 Access Information:"
echo "  Prometheus: http://localhost:${PROM_LOCAL_PORT}"
echo "  Grafana: http://localhost:${GRAFANA_LOCAL_PORT} (admin/admin or check password above)"
echo
echo "Try this command to verify with MOrA CLI:"
echo "  mora rightsize --service frontend --namespace ${HIPSTER_NAMESPACE} --strategy statistical"
echo
echo "To stop port-forwards later:"
echo "  kill \$(cat ${PORTFORWARD_PID_FILE}) && rm -f ${PORTFORWARD_PID_FILE}"
echo "  kill \$(cat ${GRAFANA_PID_FILE}) && rm -f ${GRAFANA_PID_FILE}"
echo
echo "To remove everything (minikube + all workloads):"
echo "  minikube delete"
echo

exit 0
