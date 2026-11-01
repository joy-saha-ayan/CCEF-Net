import unittest
import pandas as pd

from ccef_net.ocr.tesseract import summarize_words


class TestWordSummary(unittest.TestCase):
    def make_frame(self):
        return pd.DataFrame({
            "level": [5, 5, 5, 1],
            "page_num": [1, 1, 1, 1],
            "block_num": [1, 1, 1, 0],
            "par_num": [1, 1, 1, 0],
            "line_num": [1, 1, 2, 0],
            "text": ["বাংলা", "meme", "পরীক্ষা", None],
            "conf": [80, 60, 70, -1],
        })

    def test_line_reconstruction(self):
        result = summarize_words(self.make_frame())
        self.assertEqual(result["ocr_text"], "বাংলা meme\nপরীক্ষা")
        self.assertEqual(result["recognized_words"], 3)
        self.assertEqual(result["mean_word_confidence"], 70.0)

    def test_empty_output(self):
        frame = self.make_frame()
        frame["text"] = ""
        result = summarize_words(frame)
        self.assertEqual(result["status"], "no_text")
        self.assertEqual(result["ocr_text"], "")
        self.assertIsNone(result["mean_word_confidence"])

    def test_unknown_confidence_preserves_text(self):
        frame = self.make_frame().iloc[:1].copy()
        frame["conf"] = -1
        result = summarize_words(frame)
        self.assertEqual(result["recognized_words"], 1)
        self.assertIsNone(result["mean_word_confidence"])


if __name__ == "__main__":
    unittest.main()
