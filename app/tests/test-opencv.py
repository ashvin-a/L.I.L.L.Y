import cv2
import cv2.data

video_capture = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_bounding_box(video_frame):
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(image=gray_image, scaleFactor=1.1, 
                                             minNeighbors=5, minSize=(40,40))
    for (x, y, w, h) in faces:
        cv2.rectangle(img=video_frame, pt1=(x,y), pt2=(x+w, y+h), color=(0, 255, 255), thickness=3)
    return faces

while True:

    result , video_frame = video_capture.read()
    if result is False:
        break

    faces = detect_bounding_box(video_frame=video_frame)

    cv2.imshow("LILLY", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()