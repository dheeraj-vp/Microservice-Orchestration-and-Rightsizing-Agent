#!/bin/bash

# MOrA Resource-Optimized Data Collection Starter
# This script starts data collection with resource-optimized settings

echo "=========================================="
echo "🚀 MOrA RESOURCE-OPTIMIZED DATA COLLECTION"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""

# Check system resources first
echo "🔍 Checking system resources..."
./check_system_resources.sh

echo ""
echo "📊 RESOURCE-OPTIMIZED CONFIGURATION:"
echo "  • Experiment Duration: 15 minutes (vs 45 default)"
echo "  • Replica Counts: [1, 2, 4] (vs [1,2,4,6] default)"
echo "  • Load Levels: [5, 10, 20, 30, 50, 75] users (vs [10,50,100,150,200,250] default)"
echo "  • Max Workers: 1 (vs 4 default)"
echo "  • Total Experiments: 36 (vs 96 default)"
echo "  • Estimated Time: ~18 hours (vs 72 hours default)"
echo ""

# Check if system is ready
echo "🏥 Checking system health..."
if ! kubectl cluster-info >/dev/null 2>&1; then
    echo "❌ Kubernetes cluster not accessible"
    echo "Run: ./scripts/setup-minikube.sh"
    exit 1
fi

if ! curl -fsS http://localhost:9090/-/ready >/dev/null 2>&1; then
    echo "❌ Prometheus not accessible"
    echo "Run: ./scripts/setup-minikube.sh"
    exit 1
fi

echo "✅ System is ready for data collection"
echo ""

# Start data collection
echo "🚀 Starting resource-optimized data collection..."
echo "This will run in the background and save data immediately after each experiment."
echo ""

# Start the data collection process
nohup python3 -m src.mora.cli.main train collect-data-parallel \
    --services "frontend,checkoutservice" \
    --config-file config/resource-optimized.yaml \
    --max-workers 1 \
    > data_collection.log 2>&1 &

COLLECTION_PID=$!
echo "Data collection started with PID: $COLLECTION_PID"
echo "Log file: data_collection.log"
echo ""

# Create monitoring script
cat > monitor_collection.sh << 'EOF'
#!/bin/bash
echo "=== MOrA DATA COLLECTION MONITOR ==="
echo "Timestamp: $(date)"
echo ""

# Check if process is running
if pgrep -f "collect-data-parallel" > /dev/null; then
    echo "✅ Data collection process is RUNNING"
    echo "   PID: $(pgrep -f 'collect-data-parallel')"
else
    echo "❌ Data collection process is NOT RUNNING"
fi

# Check data collection results
if [ -d "training_data" ]; then
    json_count=$(find training_data -name "*.json" 2>/dev/null | wc -l)
    csv_count=$(find training_data -name "*.csv" 2>/dev/null | wc -l)
    echo "📊 Data Collection Results:"
    echo "   JSON files: $json_count"
    echo "   CSV files: $csv_count"
    echo "   Experiments completed: $((json_count / 2))"
else
    echo "📊 No training data collected yet"
fi

# Check system resources
echo ""
echo "💻 System Resources:"
echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "   Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "   Disk: $(df -h . | tail -1 | awk '{print $5}')"

echo ""
echo "=========================================="
EOF

chmod +x monitor_collection.sh

echo "📋 MONITORING COMMANDS:"
echo "  • Check status: ./monitor_collection.sh"
echo "  • View logs: tail -f data_collection.log"
echo "  • Stop collection: pkill -f collect-data-parallel"
echo ""

echo "🎯 EXPECTED TIMELINE:"
echo "  • First experiment: ~17 minutes (15 min load + 2 min setup)"
echo "  • All 36 experiments: ~18 hours"
echo "  • Data saved immediately after each experiment"
echo ""

echo "✅ Resource-optimized data collection started!"
echo "Your laptop should remain responsive throughout the process."
echo "=========================================="
