
class FacialSentimentDecoder:
    def __init__(self):
        self.face_logs = []

    def decode_expression(self, mouth_curve, eyebrow_tension, gaze_direction):
        emotion = "neutral"
        if mouth_curve > 0.5 and eyebrow_tension < 0.2:
            emotion = "confident"
        elif eyebrow_tension > 0.7 and mouth_curve < 0:
            emotion = "fear"
        elif gaze_direction == "avoiding" and eyebrow_tension > 0.6:
            emotion = "hesitant"
        elif mouth_curve < -0.5 and eyebrow_tension < 0.3:
            emotion = "frustrated"

        self.face_logs.append({
            "mouth": mouth_curve,
            "brow": eyebrow_tension,
            "gaze": gaze_direction,
            "emotion": emotion
        })
        return emotion

    def get_face_log(self):
        return self.face_logs
