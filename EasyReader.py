from easyocr import Reader
import re

class EasyReader:
    def __init__(self):
        self.reader = Reader(['en'], gpu=True, model_storage_directory=r'C:\Users\q659840\Desktop\labelAI-main\env\Lib\site-packages\easyocr\model', download_enabled=False)


    def get_high_confidence_alpha_numeric(self, image: str, confidence=0.8):
        """
        Reads text in an image and extracts item number with high confidence.

        Parameters:
        image: Path to image.

        Returns:
        matches[0]: the filtered item number
        0: no item number found
        """
        results = self.reader.readtext(image, allowlist='0123456789ABCDEF')
        numbers = []
        for detection in results:
            if detection[2] > confidence:
                numbers.append(detection[1])
        my_string = "".join(numbers)
        pattern = r"[0-9]{4}[0-9A-Z]{7}"
        matches = re.findall(pattern, my_string)
        if matches:
            return matches[0]
        else:
            return 0
        



