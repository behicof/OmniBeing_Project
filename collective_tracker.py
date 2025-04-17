
class CollectiveSignalTracker:
    def __init__(self):
        self.mentions = {"buy": 0, "sell": 0}
        self.posts = []

    def monitor_post(self, post_text):
        text = post_text.lower()
        if "buy" in text:
            self.mentions["buy"] += 1
        if "sell" in text:
            self.mentions["sell"] += 1
        self.posts.append(post_text)

    def detect_surge(self, threshold=10):
        if self.mentions["buy"] >= threshold and self.mentions["buy"] > self.mentions["sell"]:
            return "collective_buy_surge"
        elif self.mentions["sell"] >= threshold and self.mentions["sell"] > self.mentions["buy"]:
            return "collective_sell_surge"
        return "no_surge"

    def signal_summary(self):
        return self.mentions
