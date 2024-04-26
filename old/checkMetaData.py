from PIL import Image
from datetime import datetime, timedelta

def get_date_taken(path):
    exif = Image.open(path)._getexif()
    if not exif:
        raise Exception('Image {0} does not have EXIF data.'.format(path))
    date_string = exif[36867]
    date = datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")
    difference = datetime.now() - date

    if difference <= timedelta(weeks=4):
       print("date no older than 4 weeks")
    else:
       print("date older than 4 weeks")
    

get_date_taken('exiftest/IMG_0010.JPG')