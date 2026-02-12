import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# Suppress warnings from matplotlib about font caching
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Define the directory for results
RESULTS_DIR = "resultaten"

def plot_lifesaver_graph():
    """
    Plots the 'Lifesaver' graph, comparing AI-proposed insulin vs. final delivered dose.
    """
    try:
        # Load data
        results_file = os.path.join(RESULTS_DIR, "stress_test_results.csv")
        df = pd.read_csv(results_file)

        # Create plot
        plt.figure(figsize=(15, 7))
        
        # Plot the two insulin lines
        plt.plot(df['time_minutes'], df['algo_recommended_insulin_units'], 'r--', label='AI Proposed Dose (Unsafe)', alpha=0.7)
        plt.plot(df['time_minutes'], df['delivered_insulin_units'], 'g-', label='Final Safe Dose (Supervisor)', linewidth=2)
        
        # Shade the area where the supervisor intervened
        plt.fill_between(
            df['time_minutes'],
            df['algo_recommended_insulin_units'],
            df['delivered_insulin_units'],
            where=(df['algo_recommended_insulin_units'] > df['delivered_insulin_units']),
            color='red',
            alpha=0.2,
            interpolate=True,
            label='Supervisor Intervention'
        )

        # Formatting
        plt.title('The "Lifesaver" Graph: AI vs. Safety Supervisor', fontsize=16, fontweight='bold')
        plt.xlabel('Time (minutes)', fontsize=12)
        plt.ylabel('Insulin Dose (U)', fontsize=12)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlim(0, df['time_minutes'].max())
        plt.ylim(bottom=0)

        # Save the plot
        output_path = os.path.join(RESULTS_DIR, "lifesaver_graph.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[*] 'Lifesaver' graph saved as: {output_path}")
        plt.close()

    except FileNotFoundError:
        print(f"[!] Error: '{results_file}' not found. Run 'stress_test_analysis.py' first.")
    except Exception as e:
        print(f"[!] An unexpected error occurred while plotting the lifesaver graph: {e}")


def plot_latency_distribution():
    """
    Plots the distribution of the Safety Supervisor's latency.
    """
    try:
        results_file = os.path.join(RESULTS_DIR, "stress_test_results.csv")
        df = pd.read_csv(results_file)
        latencies_us = df['supervisor_latency_ms'] * 1000
        avg_latency = latencies_us.mean()
        p99_latency = latencies_us.quantile(0.99)

        plt.figure(figsize=(12, 6))
        plt.hist(latencies_us, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label=f'Frequency (max: {latencies_us.max():.2f} µs)')
        plt.axvline(avg_latency, color='r', linestyle='--', linewidth=2, label=f'Mean: {avg_latency:.2f} µs')
        plt.axvline(p99_latency, color='purple', linestyle=':', linewidth=2, label=f'99th Percentile: {p99_latency:.2f} µs')

        plt.title('The "Speedometer": Safety Supervisor Latency Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Latency per Check (microseconds, µs)', fontsize=12)
        plt.ylabel('Number of Checks (Logarithmic Scale)', fontsize=12)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.yscale('log')

        output_path = os.path.join(RESULTS_DIR, "latency_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[*] Latency graph saved as: {output_path}")
        plt.close()

    except FileNotFoundError:
        print(f"[!] Error: '{results_file}' not found. Run 'stress_test_analysis.py' first.")
    except Exception as e:
        print(f"[!] An unexpected error occurred while plotting the latency graph: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   IINTS-AF SDK: RESULTS VISUALIZATION")
    print("="*60 + "\n")
    plot_lifesaver_graph()
    plot_latency_distribution()
    print("\n[*] Visualization complete.")