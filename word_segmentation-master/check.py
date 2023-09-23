import page
import words
from PIL import Image
import cv2

def segment(img):
# User input page image 
 image = img

# Crop image and get bounding boxes
 new_img = page.detection(image)
 boxes = words.detection(new_img)

 lines = words.sort_words(boxes)

 i = 0
 for line in lines:
    text = new_img.copy()
    for (x1, y1, x2, y2) in line:
        # roi = text[y1:y2, x1:x2]
        save = Image.fromarray(text[y1:y2, x1:x2])
        # print(i)
        save.save("segmented/segment" + str(i) + ".png")
        i += 1
 return i  