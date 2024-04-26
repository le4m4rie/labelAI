from easyocr import Reader
import re

class EasyReader:
    def __init__(self):
        self.reader = Reader(['en'], gpu=True, model_storage_directory=r'C:\Users\q635556\Desktop\LabelAI\env\Lib\site-packages\easyocr\model', download_enabled=False)


    def get_high_confidence_alpha_numeric(self, image: str):
        results = self.reader.readtext(image, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        numbers = []
        for detection in results:
            if detection[2] > 0.8:
                numbers.append(detection[1])
        my_string = "".join(numbers)
        for i in range(len(my_string)):
            if my_string[i] == 'O':
                my_string = my_string[:i] + '0' + my_string[i+1:]
            elif my_string[i] == 'I':
                my_string = my_string[:i] + '1' + my_string[i+1:]
        pattern = r"[0-9]{4}[0-9A-Z]{7}"
        matches = re.findall(pattern, my_string)
        if matches:
            print('    **Item number detected: ' + matches[0] +  '**\n' +
                          '                                        \n' + 
                          '                                        \n')
        else:
            print('   **Item number not or not fully detected**\n' + 
                          '                                         \n' +
                          '                                           ')
            

        