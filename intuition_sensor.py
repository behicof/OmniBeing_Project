
import random

class ExtrasensoryPatternSensor:
    def __init__(self):
        self.detected_anomalies = []

    def scan_market(self, volatility, sentiment_shift, liquidity_drop):
        signal_strength = 0
        if volatility > 5:
            signal_strength += 1
        if sentiment_shift < -0.4:
            signal_strength += 1
        if liquidity_drop > 30:
            signal_strength += 1

        if signal_strength >= 2:
            anomaly = {
                "volatility": volatility,
                "sentiment_shift": sentiment_shift,
                "liquidity_drop": liquidity_drop,
                "intuition_alert": True,
                "certainty": random.uniform(0.5, 0.9)
            }
            self.detected_anomalies.append(anomaly)
            return anomaly
        return {"intuition_alert": False}

    def get_anomaly_log(self):
        return self.detected_anomalies
