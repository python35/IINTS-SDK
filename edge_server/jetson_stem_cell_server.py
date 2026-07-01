import time
import json
import random
import threading
import sys
import subprocess
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure the 'src' directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from iints.research.stem_cell_optimizer import StemCellOptimizer
from iints_desktop.local_ai import ask_local_ai

# Global state
state = {
    "is_running": True,
    "iteration": 0,
    "config": {},
    "best_score": 0.0,
    "best_params": None,
    "latest_ai_report": None,
    "score_history": [],
}

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "jetson_config.json"
HTML_FILE = BASE_DIR / "dashboard.html"
BACKUP_FILE = BASE_DIR / "jetson_state_backup.json"

# Try to load previous state to resume after a power failure
if BACKUP_FILE.exists():
    try:
        with open(BACKUP_FILE, "r") as f:
            saved_state = json.load(f)
            # Update global state with saved values
            state.update(saved_state)
            state["is_running"] = True # Ensure it starts running again
            print(f"Resumed from backup! Iteration: {state['iteration']}")
    except Exception as e:
        print(f"Failed to load backup state: {e}")

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            try:
                with open(HTML_FILE, "rb") as f:
                    self.wfile.write(f.read())
            except Exception as e:
                self.wfile.write(f"Error loading dashboard: {e}".encode())
        elif self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(state).encode())
        elif self.path == "/api/export":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Disposition", 'attachment; filename="jetson_optimizer_data.json"')
            self.end_headers()
            self.wfile.write(json.dumps(state, indent=4).encode())
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        if self.path == "/api/action":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            command = data.get("command")
            
            if command == "stop":
                state["is_running"] = False
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "stopping"}).encode())
            elif command == "update":
                # Run SDK update in background
                subprocess.Popen([sys.executable, "-m", "pip", "install", "-U", "iints-sdk-python35"])
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "updating"}).encode())
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass # Silence logs

def run_server():
    server = HTTPServer(("0.0.0.0", 8080), DashboardHandler)
    server.serve_forever()

def run_optimizer():
    optimizer = StemCellOptimizer(duration_minutes=2880)
    
    meals = [{"start_time": 300, "value": 50, "event_type": "meal"}, {"start_time": 700, "value": 80, "event_type": "meal"}, {"start_time": 1100, "value": 60, "event_type": "meal"}]
    
    while state["is_running"]:
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except Exception:
            config = {"ai_model": "hf.co/devanshamin/PubMedDiabetes-LLM-Predictions", "ai_checkin_interval_iterations": 20}
        
        state["config"] = config
        state["iteration"] += 1
        
        # Generate random parameters
        mass = random.uniform(10.0, 150.0)
        subq = random.uniform(0.0, 1.0)
        decay = random.uniform(0.0001, 0.005)
        
        # Run simulation
        result = optimizer.evaluate_graft_configuration(
            engraftment_percent=mass,
            subq_fraction=subq,
            immune_decay=decay,
            meal_schedule=meals,
            seed=random.randint(0, 100000)
        )
        
        tir = result["tir_percent"]
        if tir > state["best_score"]:
            state["best_score"] = tir
            state["best_params"] = {
                "M_graft": mass,
                "immune_rejection_rate": decay,
                "location": f"{subq*100:.1f}% SubQ, {(1-subq)*100:.1f}% Portal Vein"
            }
            
        state["score_history"].append(tir)
        if len(state["score_history"]) > 30:
            state["score_history"].pop(0)
            
        # AI Check-in
        if state["iteration"] % config.get("ai_checkin_interval_iterations", 10) == 1:
            prompt = f"We are running an edge-computing stem cell graft optimizer. Analyze this latest simulation result (Iteration {state['iteration']}): TIR={tir:.1f}%, Hypo={result['hypo_percent']:.1f}%, Hyper={result['hyper_percent']:.1f}%. Graft Mass={mass:.1f}mg, Immune Decay={decay:.4f}, SubQ Fraction={subq:.2f}."
            try:
                ai_answer = ask_local_ai(question=prompt, model=config.get("ai_model", "hf.co/devanshamin/PubMedDiabetes-LLM-Predictions"))
                state["latest_ai_report"] = ai_answer.answer
            except Exception as e:
                state["latest_ai_report"] = f"AI Error: {e}"
                
        # Save state backup to recover from power failures
        try:
            with open(BACKUP_FILE, "w") as f:
                json.dump(state, f)
        except Exception as e:
            pass # Ignore write errors to keep running
        
        # Short sleep to prevent 100% CPU lock if needed
        time.sleep(0.01)

if __name__ == "__main__":
    t_server = threading.Thread(target=run_server, daemon=True)
    t_server.start()
    print("Jetson Stem Cell Edge Optimizer Server running on port 8080...")
    run_optimizer()
    print("Optimizer stopped safely.")
