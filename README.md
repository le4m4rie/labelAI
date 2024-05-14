# labelAI
thesis work

HAAR CASCADE STEP BY STEP

1. Negative Bilder in txt File packen
2. Positive Bilder resizen
3. Positive Bilder annotieren (Slashes aufpassen)
	opencv_annotation.exe --annotations=pos.txt --images=positive/
4. Vektor aus Positives erstellen
	opencv_createsamples.exe -info pos.txt -w 24 -h 25 -num 1000 -vec pos.vec
(zeigt Samples die erzeugt werden: opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -show -vec pos.vec)
5. Cascade Ordner leeren
6. pos.vec, neg.txt, negative, positive vorhanden
7. Training
	opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -w 25 -h 25 -numPos 1000 -numNeg 1000 -numStages 10 -maxFalseAlarmRate 0.3
------------------------------------------------------------------------------------------------

False Alarm: detetion of an object when its not present in an image
	     maximum allowed pecentage of false detections

Min Hit Rate: percentage of positive samples that need to be correctly detected during training

------------------------------------------------------------------------------------------------

DEBUGGING

Debugging: Zeile 148: positive_resized/IMG_0253.JPG 1 221 274 152 234 rausgenommen weil hier Abbruch bei Sample Erstellung
	-> bringt nichts
Bei Zeilen 148 rum: Bounding Boxes um paar Pixel verschoben wenn zu nah an 0

Zeile 132: positive_resized/IMG_0412.JPG 1 123 213 244 376 entfernt -> hier war der Fehler??

-----------------------------------------------------------------------------------------------
USE CASCADE 

scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
Basically, the scale factor is used to create your scale pyramid. More explanation, your model has a fixed size defined during training, which is visible in the XML. This means that this size of the face is detected in the image if present. However, by rescaling the input image, you can resize a larger face to a smaller one, making it detectable by the algorithm.

1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce the size by 5%, you increase the chance of a matching size with the model for detection is found. This also means that the algorithm works slower since it is more thorough. You may increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether.

minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
This parameter will affect the quality of the detected faces. Higher value results in fewer detections but with higher quality. 3~6 is a good value for it. minNeighbours (in the detectMultiScale call) is the amount of detections in about the same place nessecary to count as a valid detection

minSize – Minimum possible object size. Objects smaller than that are ignored.
This parameter determines how small size you want to detect. You decide it! Usually, [30, 30] is a good start for face detection.

maxSize – Maximum possible object size. Objects bigger than this are ignored.
This parameter determines how big size you want to detect. Again, you decide it! Usually, you don't need to set it manually, the default value assumes you want to detect without an upper limit on the size of the face.


OpenCV provides the confidence via the argument "weights" in function "detectMultiScale" from class CascadeClassifier, you need to put the flag "outputRejectLevels" to true


The precision of your cascade classifier is determined by your Acceptance Ratio of the last stage. Technically, the acceptance ratio break value tells how much your model should continue to learn and when to stop.It must ideally be around 0.0000412 or so.

If it is 4.8789e-05, it signifies that your cascade is overtrained and will not detect the objects. In this case, you will have to reduce the number of stages you set and increase the number of learning samples( give in more negative and positive images)

-----------------------------------------------------------------------------------------------
TESTING

-----------------------------------------------------------------------------------------------
TRAINING PARAMETERS

30.04.24: opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -w 50 -h 30 -numPos 1600 -numNeg 1600 -numStages 10 -maxFalseAlarmRate 0.3 -> abgebrochen weil flipped Images kontraproduktiv
14.05.24: opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -w 50 -h 30 -numPos 1600 -numNeg 1600 -numStages 10 -maxFalseAlarmRate 0.3 mit leichter Rotation und Lighting und Contrast changes
