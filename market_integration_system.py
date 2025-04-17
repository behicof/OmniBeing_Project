
import random

class MarketIntegrationSystem:
    def __init__(self):
        self.external_data = {}
        self.market_connections = []
        self.live_data_streams = []

    def connect_to_platform(self, platform_name):
        # شبیه‌سازی اتصال به پلتفرم معاملاتی
        self.market_connections.append(f"Connected to {platform_name} for live data.")
        return f"Connected to {platform_name}"

    def receive_live_data(self, platform_name, data):
        # شبیه‌سازی دریافت داده‌های زنده از پلتفرم‌های مختلف
        if platform_name not in self.external_data:
            self.external_data[platform_name] = []
        self.external_data[platform_name].append(data)
        self.live_data_streams.append(data)
        return f"Data from {platform_name} received."

    def process_live_data(self):
        # پردازش داده‌های زنده برای تصمیم‌گیری بهینه
        processed_data = []
        for platform, data_list in self.external_data.items():
            for data in data_list:
                decision = "hold"
                if data['market_sentiment'] > 0.7 and data['market_volatility'] < 0.5:
                    decision = "buy"
                elif data['market_sentiment'] < 0.3 and data['market_volatility'] > 0.7:
                    decision = "sell"
                processed_data.append({"platform": platform, "decision": decision})
        return processed_data

    def get_live_decisions(self):
        # دریافت تصمیمات زنده برای همه پلتفرم‌ها
        return self.process_live_data()
