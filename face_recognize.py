from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import cv2
import os
import numpy as np
import cv2
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model, model_from_json
from API.requests_rollcall import CreateRollCall


# Load pretrained Inception-ResNet-v1 model

model_path = "Models/Inception_ResNet_v1.json"
weights_path = "Models/facenet_keras_weights.h5"
# weights_path = "enc1_model_weights.h5"

json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
print(loaded_model_json)
enc_model = model_from_json(loaded_model_json)
enc_model.load_weights(weights_path)

mtcnn_detector = MTCNN()


class Ui_Form():
    lastLabel = ""
    def setupUi(self, Form):
        Form.setObjectName("Nhan Dien Khuon Mat")
        Form.resize(1100, 768)
        self.videoCapture = QtWidgets.QLabel(Form)
        self.videoCapture.setGeometry(QtCore.QRect(300, 120, 640, 480))
        self.videoCapture.setFrameShape(QtWidgets.QFrame.Box)
        self.videoCapture.setFrameShadow(QtWidgets.QFrame.Raised)
        self.videoCapture.setLineWidth(6)
        self.videoCapture.setText("")
        self.videoCapture.setObjectName("videoCapture")

        self.recentRecognizeImage = QtWidgets.QLabel(Form)
        self.recentRecognizeImage.setGeometry(QtCore.QRect(60, 200, 130, 170))
        self.recentRecognizeImage.setFrameShape(QtWidgets.QFrame.Box)
        self.recentRecognizeImage.setFrameShadow(QtWidgets.QFrame.Raised)
        self.recentRecognizeImage.setLineWidth(1)
        self.recentRecognizeImage.setText("")
        self.recentRecognizeImage.setObjectName("recentRecognizeImage")


        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(80, 140, 111, 81))
        self.label_2.setText("")
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(550, 60, 201, 51))
        self.label_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_3.setScaledContents(False)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setWordWrap(False)
        self.label_3.setIndent(-1)
        self.label_3.setObjectName("label_3")

        self.nameEmployeeRecent = QtWidgets.QLabel(Form)
        self.nameEmployeeRecent.setGeometry(QtCore.QRect(60, 380, 130, 30))
        self.nameEmployeeRecent.setTextFormat(QtCore.Qt.AutoText)
        self.nameEmployeeRecent.setScaledContents(False)
        self.nameEmployeeRecent.setAlignment(QtCore.Qt.AlignCenter)
        self.nameEmployeeRecent.setWordWrap(False)
        self.nameEmployeeRecent.setIndent(-1)
        self.nameEmployeeRecent.setObjectName("nameEmployeeRecent")

        color = QtGui.QColor(233, 10, 150)
        alpha = 140
        values = "{r}, {g}, {b}, {a}".format(r=color.red(),
                                             g=color.green(),
                                             b=color.blue(),
                                             a=alpha
                                             )
        self.nameRecognize = QtWidgets.QLabel(Form)
        self.nameRecognize.setGeometry(QtCore.QRect(300, 120, 0, 0))
        self.nameRecognize.setTextFormat(QtCore.Qt.AutoText)
        self.nameRecognize.setAlignment(QtCore.Qt.AlignCenter)
        self.nameRecognize.setObjectName("nameRecognize")
        self.nameRecognize.setText(" ")
        self.nameRecognize.setAutoFillBackground(True)  # This is important!!
        self.nameRecognize.setStyleSheet(
            "QLabel { background-color: rgba("+values+"); }")

        self.detectFace = QtWidgets.QLabel(Form)
        self.detectFace.setGeometry(QtCore.QRect(300, 120, 0, 0))
        self.detectFace.setFrameShape(QtWidgets.QFrame.Box)
        self.detectFace.setFrameShadow(QtWidgets.QFrame.Raised)
        self.detectFace.setLineWidth(1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        # self.recentCoverImageRecognize.setTitle(
        #     _translate("Form", "Vu Tri Hau"))
        self.label_3.setText(_translate("Form", "Điểm danh khuôn mặt"))

    def openVideoCapture(self):
        self.face_recognition(None, self.known_faces_encodings,
                              self.known_faces_ids, threshold=0.75)


    # Hàm phát hiện mặt và trích xuất mặt từ ảnh đàu vào

    def detect_face(filename, required_size=(160, 160), normalize=True):

        img = Image.open(filename)

        # convert to RGB
        img = img.convert('RGB')

        # convert to array
        pixels = np.asarray(img)

        # detect faces in the image
        results = mtcnn_detector.detect_faces(pixels)

        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']

        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        # extract the face
        face = pixels[y1:y2, x1:x2]

        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)

        if normalize == True:

            mean = np.mean(face_array, axis=(0, 1, 2), keepdims=True)
            std = np.std(face_array, axis=(0, 1, 2), keepdims=True)
            std_adj = np.maximum(std, 1.0)
            return (face_array - mean) / std

        else:
            return face_array

    # Compute Face encodings and load IDs of known persons
    # Update face database path according to your working environment

    known_faces_encodings = []
    known_faces_ids = []

    known_faces_path = "Face_database/"
    # known_faces_path = "FACEDBTEST/"

    # Chạy qua toàn bộ thư mục
    # TOPO DÙNG LOAD TỪ FILE .npy cho nhanh
    for folder in os.listdir(known_faces_path):
        # Chạy qua toàn bộ ảnh trong từng thư mục
        for filename in os.listdir(known_faces_path + str(folder)+"/"):

            # Detect faces
            face = detect_face(filename=known_faces_path +
                               folder + "/"+filename, normalize=True)

            # Compute face encodings

            feature_vector = enc_model.predict(face.reshape(1, 160, 160, 3))
            feature_vector /= np.sqrt(np.sum(feature_vector**2))
            known_faces_encodings.append(feature_vector)

            # Save Person IDs
            # label = filename.split('.')[0]
            label = folder
            known_faces_ids.append(label)

    known_faces_encodings = np.array(known_faces_encodings).reshape(
        len(known_faces_encodings), 128)
    known_faces_ids = np.array(known_faces_ids)

    # np.save('known_data/known_faces_encodings.npy', known_faces_encodings)
    # np.save('known_data/known_faces_ids.npy', known_faces_ids)

    # known_faces_encodings = np.load('known_data/known_faces_encodings.npy')
    # known_faces_ids = np.load('known_data/known_faces_ids.npy')

    # print(known_faces_ids.shape[0])

    # Function to recognize a face (if it is in known_faces)

    def recognize(self, img, known_faces_encodings, known_faces_ids, threshold):

        scores = np.zeros((len(known_faces_ids), 1), dtype=float)

        enc = enc_model.predict(img.reshape(1, 160, 160, 3))
        enc /= np.sqrt(np.sum(enc**2))

        scores = np.sqrt(np.sum((enc-known_faces_encodings)**2, axis=1))

        match = np.argmin(scores)
        # print(scores[match])

        if scores[match] > threshold:

            return ("UNKNOWN", 0)

        else:

            return (known_faces_ids[match], scores[match])

    def face_recognition(self, file_path, known_faces_encodings, known_faces_ids, threshold):
        """Function to perform real-time face recognition through a webcam
        :param file_path: 
        :param known_faces_encodings: list vector 128D (Chứa đặc trưng của các face của những người đã biết)
        :param known_faces_ids: list nhãn của những người đã biết
        :param threshold: Ngưỡng để xác nhận xem có phải 1 người có trong list những người đã biết hay khong
        :param threshold:
        """
        # Mở camera với opencv
        cap = cv2.VideoCapture(0)

        while True:

            # Read the frame
            _, img = cap.read()

            # Stop if end of video file
            if _ == False:
                break
            # Dùng mtcnn detect face
            results = mtcnn_detector.detect_faces(img)

            # Display
            # Hiển thị detect + camera
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            imgCapture = QtGui.QImage(
                img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()

            faces = []

            if(len(results) > 0):
                for i in range(len(results)):

                    x, y, w, h = results[i]['box']
                    x, y = abs(x), abs(y)
                    faces.append([x, y, w, h])
            # print(faces)
            if len(faces) > 0:
                # Tìm khuôn mặt nào có kích thước lướn nhất để tiến hành nhận dạng
                # Biến lưu trữ face có độ rộng lớn nhất -> dùng để nhận dạng
                max_width_face = np.argmax(faces, axis=0)[2]
                # print(faces,max_width_face)
                # ex = Widget();
                # ex.show()
                # sys.exit(app.exec_())
                # Draw the rectangle around each face
                # for (x, y, w, h) in faces[max_width_face]:
                (x, y, w, h) = faces[max_width_face]
                image = Image.fromarray(img[y:y+h, x:x+w])
                image = image.resize((160, 160))
                face_array = asarray(image)

                # Normalize(Chuẩn hóa)
                mean = np.mean(face_array, axis=(0, 1, 2), keepdims=True)
                std = np.std(face_array, axis=(0, 1, 2), keepdims=True)
                std_adj = np.maximum(std, 1.0)
                face_array_normalized = (face_array - mean) / std

                # Recognize
                label = self.recognize(face_array_normalized,
                                       known_faces_encodings, known_faces_ids, threshold)
                #Nếu nhận diện được người có trong list database có sẵn:
                if(label[0] != "UNKNOWN"):
                    if(self.lastLabel != label[0]):
                        self.lastLabel = label[0]

                        # Hiển thị mặt và tên ở thanh điểm danh(Bên trái)
                        faceArray = img[y:y+h, x:x+w]
                        heightFace, widthFace, channelFace = faceArray.shape
                        bytesPerLineFace = 3 * widthFace
                        
                        imgFaceCapture = QtGui.QImage(
                        faceArray.data.tobytes(), widthFace, heightFace, bytesPerLineFace, QtGui.QImage.Format_RGB888).rgbSwapped()
                        
                        new_img = imgFaceCapture.scaled(130, 170)
                        self.recentRecognizeImage.setPixmap(QtGui.QPixmap(new_img))

                        self.nameEmployeeRecent.setText(str(label[0]))

                        #Gửi service thêm vào DB bằng điểm danh
                        CreateRollCall(label[0])

                    

                # Vẽ giao diện
                # Trên pyqt5
                # Khai Kháo painter -> vẽ ô vuông detect face
                self.detectFace.setGeometry(
                    QtCore.QRect(x + 300, y + 120, w, h))
                # Hiển thị tên
                # self.nameRecognize.setGeometry(QtCore.QRect(x + 300 + w/2, y + 140, 100,30))
                self.nameRecognize.setGeometry(
                    QtCore.QRect(x + 300, y + 90, w, 30))
                self.nameRecognize.setText(str(label[0]))

            else:
                # XÓA giao diện detect
                # Trên pyqt5
                self.detectFace.setGeometry(QtCore.QRect(0, 0, 0, 0))
                # Hiển thị tên
                # self.nameRecognize.setGeometry(QtCore.QRect(x + 300 + w/2, y + 140, 100,30))
                self.nameRecognize.setGeometry(QtCore.QRect(0, 0, 0, 0))
                # self.nameRecognize.setText("")

            self.videoCapture.setPixmap(QtGui.QPixmap(imgCapture))

            # Stop if escape key is pressed
            key = cv2.waitKey(25) & 0xff
            if key == 27:
                break

        # Release the VideoCapture object
        cap.release()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    ui.openVideoCapture()
    sys.exit(app.exec_())
