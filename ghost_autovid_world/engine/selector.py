
class Selector:
    def select(self, analysis_results):
        """
        Deterministically selects content roles.
        """
        valid = [v for v in analysis_results if not v["is_corrupted"] and v["duration"] > 3.0]

        if not valid:
            return None, [], []

        # Scoring for Main Content
        # Formula: Duration * AudioPresence * FacePresence * (LowMotionPreference)
        scored = []
        for v in valid:
            score = v["duration"]
            if v["avg_volume"] < 0.01: score *= 0.1 # Penalty for silence
            if v["face_score"] > 0.2: score *= 2.0  # Bonus for face
            if v["motion_score"] > 20.0: score *= 0.8 # Penalty for shaky cam in main

            scored.append((score, v))

        scored.sort(key=lambda x: x[0], reverse=True)

        main_video = scored[0][1]
        others = [x[1] for x in scored[1:]]

        b_roll = []
        discard = []

        for v in others:
            # B-Roll: High motion, short or long, audio doesn't matter (we mute it)
            # Must have visual interest (motion)
            if v["motion_score"] > 5.0 and v["duration"] > 2.0:
                b_roll.append(v)
            else:
                discard.append(v)

        return main_video, b_roll, discard
