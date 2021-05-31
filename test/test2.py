import cv2 as cv
import cv2
from mtcnn.mtcnn import MTCNN
import sys
img = cv.imread(
    r"C:\Users\Huy Hoang PC\Desktop\github\Face_Recognize_InceptionResnetv1_TripletLoss\test\WIN_20210428_18_59_41_Pro.png")

print(type(img))
# mtcnn_detector = MTCNN()
# # print(img.shape)
# # print(img.shape)
# # print(img[500])
# results = mtcnn_detector.detect_faces(img)
# # print(results.data)
# x1, y1, width, height = results[0]['box']
# # print(x1, y1, width, height)
# img = img[y1:y1+height, x1:x1+width]
# print(type(img))
# if img is None:
#     sys.exit("Could not read the image.")
# cv.imshow("Display window", img)
# k = cv.waitKey(0)
# if k == ord("s"):
#     cv.imwrite("starry_night.png", img)
