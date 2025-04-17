def make_advanced_decision(self, market_data, prediction, history=None):
    """
    تصمیم‌گیری پیشرفته با استفاده از ترکیب اندیکاتورهای تکنیکال و نتایج مدل LSTM
    """
    current_price = market_data["close"].iloc[-1]
    
    signals = {
        "price_direction": 0,
        "rsi_signal": 0,
        "lstm_signal": 0
    }
    
    # بررسی جهت قیمت
    price_sma_short = market_data["close"].rolling(window=5).mean().iloc[-1]
    price_sma_long = market_data["close"].rolling(window=20).mean().iloc[-1]
    
    if price_sma_short > price_sma_long:
        signals["price_direction"] = 1
    elif price_sma_short < price_sma_long:
        signals["price_direction"] = -1
    
    # سیگنال مدل LSTM
    if prediction > current_price * 1.005:
        signals["lstm_signal"] = 1
    elif prediction < current_price * 0.995:
        signals["lstm_signal"] = -1
    
    # تصمیم نهایی
    total_score = (signals["price_direction"] * 0.3 + 
                  signals["rsi_signal"] * 0.3 + 
                  signals["lstm_signal"] * 0.4)
    
    if total_score > 0.3:
        return "BUY"
    elif total_score < -0.3:
        return "SELL"
    else:
        return "HOLD"
