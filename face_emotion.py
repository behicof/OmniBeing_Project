
class FacialEmotionDetector:
    def __init__(self):
        self.emotions_detected = []

    def detect(self, frame_info):
        # اینجا فقط شبیه‌سازی می‌کنیم، چون پردازش تصویر واقعی نیست
        simulated_emotion = "neutral"
        if "raised eyebrows" in frame_info:
            simulated_emotion = "surprise"
        elif "furrowed brow" in frame_info:
            simulated_emotion = "anger"
        elif "smile" in frame_info:
            simulated_emotion = "happy"
        elif "tight lips" in frame_info:
            simulated_emotion = "stress"
        elif "open mouth" in frame_info:
            simulated_emotion = "fear"

        self.emotions_detected.append((frame_info, simulated_emotion))
        return simulated_emotion

    def get_log(self):
        return self.emotions_detected
