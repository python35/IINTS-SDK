import numpy as np
from collections import deque

class SensorNoiseModel:
    """Realistic sensor noise with drift and filtering."""
    
    def __init__(self, white_noise_std=15, drift_rate=0.1, drift_amplitude=10):
        self.white_noise_std = white_noise_std
        self.drift_rate = drift_rate
        self.drift_amplitude = drift_amplitude
        self.drift_phase = 0
        
    def add_noise(self, true_glucose, time_step):
        """Add realistic sensor noise with drift."""
        white_noise = np.random.normal(0, self.white_noise_std)
        drift = self.drift_amplitude * np.sin(self.drift_phase)
        self.drift_phase += self.drift_rate * time_step / 60  # Convert to hours
        
        return true_glucose + white_noise + drift

class KalmanFilter:
    """Simple Kalman filter for glucose smoothing."""
    
    def __init__(self, process_variance=1, measurement_variance=225):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = None
        self.error_estimate = 1000
        
    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return measurement
            
        # Prediction
        prediction = self.estimate
        prediction_error = self.error_estimate + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_estimate = (1 - kalman_gain) * prediction_error
        
        return self.estimate

class MovingAverageFilter:
    """Simple moving average filter."""
    
    def __init__(self, window_size=3):
        self.window = deque(maxlen=window_size)
        
    def update(self, value):
        self.window.append(value)
        return sum(self.window) / len(self.window)