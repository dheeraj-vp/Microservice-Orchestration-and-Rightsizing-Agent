#!/bin/bash




echo "=========================================="
echo "🔍 SYSTEM RESOURCE CHECK"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""


echo "💾 MEMORY STATUS:"
total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2}')
available_mem=$(free -m | awk 'NR==2{printf "%.0f", $7}')
used_mem=$(free -m | awk 'NR==2{printf "%.0f", $3}')
mem_usage_percent=$((used_mem * 100 / total_mem))

echo "   Total Memory: ${total_mem}MB"
echo "   Available Memory: ${available_mem}MB"
echo "   Used Memory: ${used_mem}MB (${mem_usage_percent}%)"

if [ $mem_usage_percent -gt 80 ]; then
    echo "   ⚠️  WARNING: High memory usage (${mem_usage_percent}%)"
    echo "   Recommendation: Close other applications before starting data collection"
elif [ $available_mem -lt 4000 ]; then
    echo "   ⚠️  WARNING: Low available memory (${available_mem}MB)"
    echo "   Recommendation: Free up memory before starting data collection"
else
    echo "   ✅ Memory status: OK"
fi


echo ""
echo "🖥️  CPU STATUS:"
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "   Current CPU Usage: ${cpu_usage}%"

if (( $(echo "$cpu_usage > 80" | bc -l) )); then
    echo "   ⚠️  WARNING: High CPU usage (${cpu_usage}%)"
    echo "   Recommendation: Wait for CPU usage to decrease"
else
    echo "   ✅ CPU status: OK"
fi


echo ""
echo "💿 DISK STATUS:"
disk_usage=$(df -h . | tail -1 | awk '{print $5}' | cut -d'%' -f1)
available_space=$(df -h . | tail -1 | awk '{print $4}')
echo "   Disk Usage: ${disk_usage}%"
echo "   Available Space: ${available_space}"

if [ $disk_usage -gt 90 ]; then
    echo "   ⚠️  WARNING: High disk usage (${disk_usage}%)"
    echo "   Recommendation: Free up disk space"
else
    echo "   ✅ Disk status: OK"
fi


echo ""
echo "🔄 RUNNING PROCESSES:"
heavy_processes=$(ps aux --sort=-%cpu | head -10 | grep -v "USER\|%CPU" | awk '{if($3>10.0) print $2, $3, $11}' | head -5)
if [ -n "$heavy_processes" ]; then
    echo "   Heavy processes (>10% CPU):"
    echo "$heavy_processes" | while read pid cpu cmd; do
        echo "     PID $pid: ${cpu}% CPU - $cmd"
    done
    echo "   Recommendation: Consider closing heavy processes"
else
    echo "   ✅ No heavy processes detected"
fi


echo ""
echo "📋 RECOMMENDATION:"
if [ $mem_usage_percent -gt 80 ] || (( $(echo "$cpu_usage > 80" | bc -l) )) || [ $disk_usage -gt 90 ]; then
    echo "   ❌ System resources are HIGH"
    echo "   ⏸️  Wait for resources to free up before starting data collection"
    echo "   💡 Consider using resource-optimized configuration"
else
    echo "   ✅ System resources are OK"
    echo "   🚀 Safe to start data collection"
fi

echo ""
echo "=========================================="
