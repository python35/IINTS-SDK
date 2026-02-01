import time
import subprocess
import threading
import psutil
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class PerformanceMetrics:
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    temperature: Optional[float] = None
    power_consumption: Optional[float] = None

class HardwareBenchmark:
    """Hardware performance monitoring for Jetson and other platforms."""
    
    def __init__(self, sample_interval=1.0):
        self.sample_interval = sample_interval
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
        self.is_jetson = self._detect_jetson()
        
    def _detect_jetson(self) -> bool:
        """Detect if running on Jetson platform."""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                return 'jetson' in model.lower()
        except:
            return False
    
    def _get_tegrastats_metrics(self) -> Optional[Dict]:
        """Parse tegrastats output for Jetson-specific metrics."""
        if not self.is_jetson:
            return None
            
        try:
            result = subprocess.run(['tegrastats', '--interval', '100'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                # Parse tegrastats output (simplified)
                lines = result.stdout.strip().split('\n')
                if lines:
                    # Example parsing - adapt based on actual tegrastats format
                    line = lines[-1]  # Get last line
                    # This is a simplified parser - real implementation would be more robust
                    return {"raw_tegrastats": line}
        except:
            pass
        return None
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        timestamp = time.time()
        
        # CPU and Memory (cross-platform)
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics (Jetson-specific)
        gpu_usage = None
        gpu_memory = None
        temperature = None
        power_consumption = None
        
        if self.is_jetson:
            tegra_metrics = self._get_tegrastats_metrics()
            if tegra_metrics:
                # Parse tegrastats data here
                # This is platform-specific and would need proper parsing
                pass
            
            # Try to get temperature
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_millicelsius = int(f.read().strip())
                    temperature = temp_millicelsius / 1000.0
            except:
                pass
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            temperature=temperature,
            power_consumption=power_consumption
        )
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 samples to prevent memory issues
            if len(self.metrics_history) > 1000:
                self.metrics_history.pop(0)
            
            time.sleep(self.sample_interval)
    
    def start_monitoring(self):
        """Start background performance monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def benchmark_algorithm(self, algorithm, test_data, iterations=100):
        """Benchmark algorithm performance."""
        self.start_monitoring()
        
        start_time = time.time()
        start_metrics = self._collect_metrics()
        
        # Run algorithm iterations
        for i in range(iterations):
            # Simulate algorithm execution
            if hasattr(algorithm, 'calculate_insulin'):
                algorithm.calculate_insulin(
                    current_glucose=test_data.get('glucose', 120),
                    time_step=test_data.get('time_step', 5),
                    carb_intake=test_data.get('carbs', 0)
                )
        
        end_time = time.time()
        end_metrics = self._collect_metrics()
        
        self.stop_monitoring()
        
        # Calculate performance statistics
        total_time = end_time - start_time
        avg_time_per_iteration = total_time / iterations
        
        # Get metrics during benchmark
        benchmark_metrics = [m for m in self.metrics_history 
                           if start_time <= m.timestamp <= end_time]
        
        if benchmark_metrics:
            avg_cpu = sum(m.cpu_usage for m in benchmark_metrics) / len(benchmark_metrics)
            avg_memory = sum(m.memory_usage for m in benchmark_metrics) / len(benchmark_metrics)
            max_cpu = max(m.cpu_usage for m in benchmark_metrics)
            max_memory = max(m.memory_usage for m in benchmark_metrics)
        else:
            avg_cpu = avg_memory = max_cpu = max_memory = 0
        
        return {
            "algorithm_name": algorithm.__class__.__name__,
            "iterations": iterations,
            "total_time_seconds": total_time,
            "avg_time_per_iteration_ms": avg_time_per_iteration * 1000,
            "iterations_per_second": iterations / total_time,
            "cpu_usage": {
                "average": avg_cpu,
                "maximum": max_cpu
            },
            "memory_usage": {
                "average": avg_memory,
                "maximum": max_memory
            },
            "platform": "Jetson" if self.is_jetson else "Generic",
            "sample_count": len(benchmark_metrics)
        }
    
    def get_current_metrics(self) -> Dict:
        """Get current system metrics."""
        metrics = self._collect_metrics()
        return asdict(metrics)
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        cpu_values = [m.cpu_usage for m in self.metrics_history]
        memory_values = [m.memory_usage for m in self.metrics_history]
        
        return {
            "sample_count": len(self.metrics_history),
            "duration_seconds": self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp,
            "cpu_usage": {
                "average": sum(cpu_values) / len(cpu_values),
                "minimum": min(cpu_values),
                "maximum": max(cpu_values)
            },
            "memory_usage": {
                "average": sum(memory_values) / len(memory_values),
                "minimum": min(memory_values),
                "maximum": max(memory_values)
            },
            "platform": "Jetson" if self.is_jetson else "Generic"
        }
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file."""
        data = {
            "platform": "Jetson" if self.is_jetson else "Generic",
            "sample_interval": self.sample_interval,
            "metrics": [asdict(m) for m in self.metrics_history]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_metrics(self):
        """Clear collected metrics."""
        self.metrics_history.clear()