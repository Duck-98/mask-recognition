from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import FPS
import imutils
import numpy as np
import cv2

print("[얼굴 인식 모델 로딩]")
net = cv2.dnn.readNet('models/deploy.prototxt',
                      'models/res10_300x300_ssd_iter_140000.caffemodel')
#model = load_model('models/mask_detector.model')
print("[마스크 인식 모델 로딩]")
model = load_model('mask_detector.h5')
#cap = cv2.VideoCapture('imgs/01.mp4')
print("[웹캠 시작]")
cap = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))
#net = cv2.dnn.readNet(model, config)


def Mask_dect(img, net, model):
    h, w = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img,
                                 scalefactor=1.,
                                 size=(300, 300),
                                 mean=(104., 177., 123.))
    net.setInput(blob)
    dets = net.forward()

    result_img = img.copy()
    faces = []
    locations = []
    predicts = []

    for i in range(0, dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue

        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        (x1, y1) = (max(0, x1), max(0, y1))
        (x2, y2) = (min(w - 1, x2), min(h - 1, y2))

        face = img[y1:y2, x1:x2]
        face_in = cv2.resize(face, (224, 224))
        face_in = cv2.cvtColor(face_in, cv2.COLOR_BGR2RGB)
        face_in = img_to_array(face_in)
        face_in = preprocess_input(face_in)

        faces.append(face_in)
        locations.append((x1, y1, x2, y2))

    if len(faces) > 0:
        face_result = np.array(faces, dtype="float32")
        predicts = model.predict(face_result, batch_size=32)

    return (locations, predicts)


fps = FPS().start()

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    if not ret:
        break

    #img = imutils.resize(img, width=500)

    (loc, pre) = Mask_dect(img, net, model)

    for (box, pre) in zip(loc, pre):
        (x1, y1, x2, y2) = box
        (mask, nomask) = pre
        label = "Mask" if mask > nomask else "No Mask"

        if label == "Mask" and max(mask, nomask) * 100 >= 70:
            color = (0, 255, 0)
        elif label == "No Mask" and max(mask, nomask) * 100 >= 70:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        label = "{}: {: .2f}%".format(label, max(mask, nomask) * 100)

        cv2.putText(img,
                    label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.80,
                    color,
                    2,
                    lineType=cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

    cv2.imshow("Mask_img", img)
    if cv2.waitKey(1) == ord("q"):
        break
    fps.update()

fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

cap.release()

#     h, w = img.shape[:2]

#     blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
#     net.setInput(blob)
#     dets = net.forward()

#     result_img = img.copy()

#     for i in range(dets.shape[2]):
#         confidence = dets[0, 0, i, 2]
#         if confidence < 0.5:
#             continue

#         x1 = int(dets[0, 0, i, 3] * w)
#         y1 = int(dets[0, 0, i, 4] * h)
#         x2 = int(dets[0, 0, i, 5] * w)
#         y2 = int(dets[0, 0, i, 6] * h)

#         face = img[y1:y2, x1:x2]

#         face_input = cv2.resize(face, dsize=(224, 224))
#         face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
#         face_input = preprocess_input(face_input)
#         face_input = np.expand_dims(face_input, axis=0)

#         mask, nomask = model.predict(face_input).squeeze()

#         if mask > nomask:
#             color = (0, 255, 0)
#             label = 'Mask %d%%' % (mask * 100)
#         else:
#             color = (0, 0, 255)
#             label = 'No Mask %d%%' % (nomask * 100)

#         cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
#         cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

#     #out.write(result_img)
#     cv2.imshow('result', result_img)
#     if cv2.waitKey(1) == ord('q'):
#         break

# #out.release()
# cap.release()
