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
