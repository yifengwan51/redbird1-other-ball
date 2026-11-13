"""Regression checks for the bundled, openly licensed tennis examples."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


class TennisSampleTests(unittest.TestCase):
    def test_motion_fallback_finds_candidates_in_bundled_examples(self):
        from integrations import lightweight as lw

        samples = {
            "djokovic": Path("inputs/tennis_djokovic_point.webm"),
            "davis-cup": Path("inputs/tennis_davis_cup_doubles.webm"),
        }
        original = dict(lw.VIDEO_IDS)
        try:
            lw.VIDEO_IDS.update({name: lw.ROOT / path for name, path in samples.items()})
            for name, path in samples.items():
                self.assertTrue((lw.ROOT / path).is_file(), path)
                duration = lw._video_info(lw.VIDEO_IDS[name])["duration"]
                events = lw._motion_events(name, duration)
                self.assertTrue(events, name)
                self.assertTrue(all(event["event_type"] == "tennis_action_candidate" for event in events))
        finally:
            lw.VIDEO_IDS.clear()
            lw.VIDEO_IDS.update(original)


if __name__ == "__main__":
    unittest.main()
