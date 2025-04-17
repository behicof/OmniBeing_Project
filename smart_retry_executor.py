
import time

class SmartRetryExecutor:
    def __init__(self, max_retries=3, cooldown=5):
        self.max_retries = max_retries
        self.cooldown = cooldown

    def execute_with_retry(self, trade_function, *args, **kwargs):
        for attempt in range(self.max_retries):
            success, message = trade_function(*args, **kwargs)
            if success:
                return {"status": "SUCCESS", "message": message, "attempt": attempt + 1}
            time.sleep(self.cooldown * (attempt + 1))  # افزایش زمان تأخیر
        return {"status": "FAILED", "message": "All retries failed"}
