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
        plt.plot(df['time_minutes'], df['algo_recommended_insulin_units'], 'r--', label='AI Voorgestelde Dosis (Onveilig)', alpha=0.7)
        plt.plot(df['time_minutes'], df['delivered_insulin_units'], 'g-', label='Finale Veilige Dosis (Supervisor)', linewidth=2)
        
        # Shade the area where the supervisor intervened
        plt.fill_between(
            df['time_minutes'],
            df['algo_recommended_insulin_units'],
            df['delivered_insulin_units'],
            where=(df['algo_recommended_insulin_units'] > df['delivered_insulin_units']),
            color='red',
            alpha=0.2,
            interpolate=True,
            label='Ingreep Supervisor'
        )

        # Formatting
        plt.title('De "Levensredder"-Grafiek: AI vs. Safety Supervisor', fontsize=16, fontweight='bold')
        plt.xlabel('Tijd (minuten)', fontsize=12)
        plt.ylabel('Insuline Dosis (U)', fontsize=12)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlim(0, df['time_minutes'].max())
        plt.ylim(bottom=0)

        # Save the plot
        output_path = os.path.join(RESULTS_DIR, "lifesaver_graph.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[*] 'Levensredder'-grafiek opgeslagen als: {output_path}")
        plt.close()

    except FileNotFoundError:
        print(f"[!] Fout: '{results_file}' niet gevonden. Draai eerst 'stress_test_analysis.py'.")
    except Exception as e:
        print(f"[!] Een onverwachte fout is opgetreden bij het plotten van de levensredder-grafiek: {e}")


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
        plt.hist(latencies_us, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label=f'Frequentie (max: {latencies_us.max():.2f} µs)')
        plt.axvline(avg_latency, color='r', linestyle='--', linewidth=2, label=f'Gemiddelde: {avg_latency:.2f} µs')
        plt.axvline(p99_latency, color='purple', linestyle=':', linewidth=2, label=f'99e Percentiel: {p99_latency:.2f} µs')

        plt.title('De "Snelheidsmeter": Latency Distributie van de Safety Supervisor', fontsize=16, fontweight='bold')
        plt.xlabel('Latency per Check (microseconden, µs)', fontsize=12)
        plt.ylabel('Aantal Checks (Logaritmische Schaal)', fontsize=12)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.yscale('log')

        output_path = os.path.join(RESULTS_DIR, "latency_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[*] Latency-grafiek opgeslagen als: {output_path}")
        plt.close()

    except FileNotFoundError:
        print(f"[!] Fout: '{results_file}' niet gevonden. Draai eerst 'stress_test_analysis.py'.")
    except Exception as e:
        print(f"[!] Een onverwachte fout is opgetreden bij het plotten van de latency-grafiek: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   IINTS-AF SDK: RESULTATEN VISUALISATIE")
    print("="*60 + "\n")
    plot_lifesaver_graph()
    plot_latency_distribution()
    print("\n[*] Visualisatie compleet.")