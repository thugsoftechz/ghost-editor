
import os

class Selector:
    def select(self, analysis_results):
        """
        Classifies videos into Main, B-Roll, and Junk.
        """
        main_video = None
        b_roll = []
        junk = []

        # 1. Filter Valid
        valid_videos = [v for v in analysis_results if not v["is_corrupted"] and v["duration"] > 5.0]
        junk.extend([v for v in analysis_results if v["is_corrupted"] or v["duration"] <= 5.0])

        if not valid_videos:
            return None, [], junk

        # 2. Score for Main Video
        # Criteria: Longest, Has Audio, Face Presence
        # Score = Duration * (AudioVol > 0.01) * (FaceRatio + 0.1)

        scored_videos = []
        for v in valid_videos:
            score = v["duration"]
            if v["avg_volume"] < 0.001: score *= 0.1 # Penalty for silence
            if v["face_presence_ratio"] > 0.3: score *= 2.0 # Bonus for face

            scored_videos.append((score, v))

        # Sort desc
        scored_videos.sort(key=lambda x: x[0], reverse=True)

        main_video = scored_videos[0][1]
        remaining = [x[1] for x in scored_videos[1:]]

        # 3. Classify Remaining as B-Roll or Junk
        for v in remaining:
            # B-Roll criteria: High motion, low audio is better but high audio is okay (muted later)
            # Must not be static (low motion)
            if v["motion_score"] > 2.0: # Arbitrary threshold
                b_roll.append(v)
            else:
                # Boring static shot without being main
                junk.append(v)

        print(f"🎯 Selection Complete:")
        print(f"   Main: {os.path.basename(main_video['path'])}")
        print(f"   B-Roll: {len(b_roll)} clips")
        print(f"   Junk: {len(junk)} clips")

        return main_video, b_roll, junk
