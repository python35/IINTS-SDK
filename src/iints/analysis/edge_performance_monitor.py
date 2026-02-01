#!/usr/bin/env python3
"""
Edge AI Performance Monitor - IINTS-AF
Jetson Nano performance validation for medical device standards
"""

import time
import psutil
import json
from pathlib import Path
from datetime import datetime
import numpy as np

class EdgeAIPerformanceMonitor:
    """Monitor Jetson Nano performance for medical device validation"""
    
    def __init__(self):
        self.performance_log = []
        self.baseline_metrics = None
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start performance monitoring session"""
        self.monitoring_active = True
        self.baseline_metrics = self._capture_baseline()
        
        print("Edge AI Performance Monitoring Started")
        print(f"Baseline CPU: {self.baseline_metrics['cpu_percent']:.1f}%")
        print(f"Baseline Memory: {self.baseline_metrics['memory_mb']:.1f} MB")
        print(f"Available Memory: {self.baseline_metrics['available_memory_mb']:.1f} MB")
        
    def measure_inference_latency(self, inference_function, input_data, iterations=100):
        """Measure AI inference latency with statistical analysis"""
        
        if not self.monitoring_active:
            self.start_monitoring()
        
        latencies = []
        cpu_usage = []
        memory_usage = []
        
        print(f"Measuring inference latency over {iterations} iterations...")
        
        for i in range(iterations):
            # Pre-inference metrics
            cpu_before = psutil.cpu_percent()
            memory_before = psutil.virtual_memory().used / (1024 * 1024)  # MB
            
            # Measure inference time
            start_time = time.perf_counter()
            
            try:
                result = inference_function(input_data)
            except Exception as e:
                print(f"Inference error at iteration {i}: {e}")
                continue
            
            end_time = time.perf_counter()
            
            # Post-inference metrics
            cpu_after = psutil.cpu_percent()
            memory_after = psutil.virtual_memory().used / (1024 * 1024)  # MB
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            cpu_delta = cpu_after - cpu_before
            memory_delta = memory_after - memory_before
            
            latencies.append(latency_ms)
            cpu_usage.append(cpu_delta)
            memory_usage.append(memory_delta)
            
            # Log detailed metrics every 10 iterations
            if (i + 1) % 10 == 0:
                avg_latency = np.mean(latencies[-10:])
                print(f"Iteration {i+1:3d}: Avg latency {avg_latency:.2f}ms")
        
        # Statistical analysis
        performance_stats = self._analyze_performance_stats(latencies, cpu_usage, memory_usage)
        
        # Log performance session
        self.performance_log.append({
            'timestamp': datetime.now().isoformat(),
            'test_type': 'inference_latency',
            'iterations': len(latencies),
            'statistics': performance_stats,
            'raw_latencies_ms': latencies,
            'cpu_usage_delta': cpu_usage,
            'memory_usage_delta_mb': memory_usage
        })
        
        return performance_stats
    
    def _analyze_performance_stats(self, latencies, cpu_usage, memory_usage):
        """Analyze performance statistics for medical device validation"""
        
        if not latencies:
            return {'error': 'No valid measurements'}
        
        latency_stats = {
            'mean_ms': round(np.mean(latencies), 3),
            'median_ms': round(np.median(latencies), 3),
            'std_ms': round(np.std(latencies), 3),
            'min_ms': round(np.min(latencies), 3),
            'max_ms': round(np.max(latencies), 3),
            'p95_ms': round(np.percentile(latencies, 95), 3),
            'p99_ms': round(np.percentile(latencies, 99), 3)
        }
        
        # Embedded system performance classification
        mean_latency = latency_stats['mean_ms']
        if mean_latency < 10:
            performance_class = "SUB_10MS_LATENCY - Real-time capable"
        elif mean_latency < 50:
            performance_class = "SUB_50MS_LATENCY - Near real-time suitable"
        elif mean_latency < 100:
            performance_class = "SUB_100MS_LATENCY - Batch processing suitable"
        else:
            performance_class = "OPTIMIZATION_REQUIRED - Exceeds embedded constraints"
        
        # Consistency analysis (coefficient of variation)
        cv_percent = (latency_stats['std_ms'] / latency_stats['mean_ms']) * 100
        
        if cv_percent < 5:
            consistency_rating = "HIGHLY CONSISTENT"
        elif cv_percent < 15:
            consistency_rating = "CONSISTENT"
        elif cv_percent < 30:
            consistency_rating = "MODERATELY VARIABLE"
        else:
            consistency_rating = "HIGHLY VARIABLE - Investigate"
        
        return {
            'latency_statistics': latency_stats,
            'performance_classification': performance_class,
            'consistency_rating': consistency_rating,
            'coefficient_of_variation_percent': round(cv_percent, 2),
            'cpu_impact': {
                'mean_delta_percent': round(np.mean(cpu_usage), 2),
                'max_delta_percent': round(np.max(cpu_usage), 2)
            },
            'memory_impact': {
                'mean_delta_mb': round(np.mean(memory_usage), 2),
                'max_delta_mb': round(np.max(memory_usage), 2)
            },
            'medical_device_assessment': self._assess_embedded_suitability(mean_latency, cv_percent)
        }
    
    def _assess_embedded_suitability(self, mean_latency, cv_percent):
        """Assess suitability for embedded system deployment"""
        
        # Embedded system criteria
        criteria = {
            'real_time_response': mean_latency < 100,  # < 100ms for real-time
            'consistent_performance': cv_percent < 20,  # < 20% variation
            'low_latency': mean_latency < 50,  # < 50ms preferred
            'high_reliability': cv_percent < 10  # < 10% for high reliability
        }
        
        passed_criteria = sum(criteria.values())
        total_criteria = len(criteria)
        
        if passed_criteria == total_criteria:
            suitability = "EMBEDDED_OPTIMAL - Compatible with real-time constraints"
        elif passed_criteria >= 3:
            suitability = "EMBEDDED_SUITABLE - Compatible with near real-time applications"
        elif passed_criteria >= 2:
            suitability = "RESEARCH_GRADE - Suitable for research and development"
        else:
            suitability = "OPTIMIZATION_REQUIRED - Requires performance tuning"
        
        return {
            'suitability_rating': suitability,
            'criteria_passed': f"{passed_criteria}/{total_criteria}",
            'detailed_criteria': criteria,
            'recommendations': self._generate_optimization_recommendations(criteria)
        }
    
    def _generate_optimization_recommendations(self, criteria):
        """Generate optimization recommendations based on failed criteria"""
        
        recommendations = []
        
        if not criteria['real_time_response']:
            recommendations.append("Optimize model architecture for faster inference")
        
        if not criteria['consistent_performance']:
            recommendations.append("Investigate system load variations and thermal throttling")
        
        if not criteria['low_latency']:
            recommendations.append("Consider model quantization or pruning techniques")
        
        if not criteria['high_reliability']:
            recommendations.append("Implement performance monitoring and adaptive scheduling")
        
        if not recommendations:
            recommendations.append("Performance meets embedded system constraints")
        
        return recommendations
    
    def _capture_baseline(self):
        """Capture baseline system metrics"""
        
        memory = psutil.virtual_memory()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_mb': memory.used / (1024 * 1024),
            'available_memory_mb': memory.available / (1024 * 1024),
            'memory_percent': memory.percent,
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown'
        }
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        
        if not self.performance_log:
            return "No performance data available"
        
        latest_session = self.performance_log[-1]
        stats = latest_session['statistics']
        
        report = f"""
EDGE AI PERFORMANCE VALIDATION REPORT
=====================================

Test Configuration:
- Device: Jetson Nano (Edge AI Platform)
- Test Date: {latest_session['timestamp'][:19]}
- Iterations: {latest_session['iterations']}
- Test Type: {latest_session['test_type']}

INFERENCE PERFORMANCE:
- Mean Latency: {stats['latency_statistics']['mean_ms']:.3f} ms
- Median Latency: {stats['latency_statistics']['median_ms']:.3f} ms
- 95th Percentile: {stats['latency_statistics']['p95_ms']:.3f} ms
- 99th Percentile: {stats['latency_statistics']['p99_ms']:.3f} ms
- Standard Deviation: {stats['latency_statistics']['std_ms']:.3f} ms

PERFORMANCE CLASSIFICATION:
- Overall Rating: {stats['performance_classification']}
- Consistency: {stats['consistency_rating']}
- Coefficient of Variation: {stats['coefficient_of_variation_percent']:.2f}%

EMBEDDED SYSTEM ASSESSMENT:
- Suitability: {stats['medical_device_assessment']['suitability_rating']}
- Criteria Passed: {stats['medical_device_assessment']['criteria_passed']}

SYSTEM IMPACT:
- CPU Usage Delta: {stats['cpu_impact']['mean_delta_percent']:.2f}% (avg), {stats['cpu_impact']['max_delta_percent']:.2f}% (max)
- Memory Usage Delta: {stats['memory_impact']['mean_delta_mb']:.2f} MB (avg), {stats['memory_impact']['max_delta_mb']:.2f} MB (max)

RECOMMENDATIONS:
"""
        
        for rec in stats['medical_device_assessment']['recommendations']:
            report += f"- {rec}\n"
        
        return report
    
    def export_performance_data(self, filepath):
        """Export performance data for analysis"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'baseline_metrics': self.baseline_metrics,
            'performance_sessions': self.performance_log,
            'summary_report': self.generate_performance_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filepath

# Mock inference function for testing
def mock_ai_inference(input_data):
    """Mock AI inference function for testing"""
    # Simulate neural network computation
    time.sleep(0.008)  # 8ms base latency
    
    # Add some variability
    time.sleep(np.random.exponential(0.002))  # Variable component
    
    return {"prediction": np.random.random(), "confidence": np.random.random()}

def main():
    """Test edge AI performance monitoring"""
    monitor = EdgeAIPerformanceMonitor()
    
    print("IINTS-AF Edge AI Performance Validation")
    print("=" * 45)
    
    # Test inference performance
    test_input = {"glucose": 150, "trend": [145, 148, 150]}
    
    performance_stats = monitor.measure_inference_latency(
        mock_ai_inference, test_input, iterations=50
    )
    
    # Generate and display report
    report = monitor.generate_performance_report()
    print(report)
    
    # Export data
    export_file = Path("results") / "edge_ai_performance.json"
    export_file.parent.mkdir(exist_ok=True)
    monitor.export_performance_data(export_file)
    
    print(f"\nPerformance data exported to: {export_file}")

if __name__ == "__main__":
    main()