from xml.etree.ElementTree import tostring

import cv2
from ultralytics import YOLO
import winsound






yolo_model = YOLO('./detection/weights/best.pt')


class_names = yolo_model.model.names


print("Veri setindeki tüm objeler:")
for class_id, class_name in class_names.items():
    print(f"{class_id}: {class_name}")

def detected_object_id(object_id):
    if object_id == 0:
        return "silah"
    if object_id == 1:
        return "bicak"
    return "bilinmiyor"

print(detected_object_id(0))
print(detected_object_id(1))


def detect_objects_in_photo(image_path):
    image_orig = cv2.imread(image_path)

    yolo_model = YOLO('./detection/weights/best.pt')

    results = yolo_model(image_orig)

    alarm_triggered = False
    detected_object_name = None

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)
                if not alarm_triggered: alarm_triggered = True
                detected_object_name = detected_object_id(int(cls[pos]))

    if detected_object_name: cv2.putText(image_orig, "tespit edildi: " + detected_object_name, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2, cv2.LINE_AA)

    result_path = "./Results/test.jpeg"
    cv2.imwrite(result_path, image_orig)

    cv2.imshow('Detected Objects', image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_path

imgpath = "./testfoto/testpicture3.png"
detect_objects_in_photo(imgpath)


def detect_objects_in_video(video_path, output_folder="./Results", fps=30):
    yolo_model = YOLO('./detection/weights/best.pt')
    video_capture = cv2.VideoCapture(video_path)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_video_path = "detected_objects_video2.mp4"
    out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))


    alarm_triggered = False
    detected_object_name = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                cv2.LINE_AA)
                    if not alarm_triggered: alarm_triggered = True
                    detected_object_name = detected_object_id(int(cls[pos]))


        if detected_object_name:
            cv2.putText(frame, "tespit edildi: " + detected_object_name, (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        out.write(frame)

        cv2.imshow('Detected Objects in Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

    return result_video_path

videopth = "./testfoto/testvideo4.mp4"

output_folder = "./Results"
fps = 30
detect_objects_in_video(videopth, output_folder, fps)


def detect_objects_and_plot(path_orig):
    image_orig = cv2.imread(path_orig)

    yolo_model = YOLO('./detection/weights/best.pt')

    results = yolo_model(image_orig)

    alarm_triggered = False
    detected_object_name = None

    for result in results:
        classes = result.names
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.5:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                color = (0, int(cls[pos]), 255)
                cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                            cv2.LINE_AA)
            if not alarm_triggered:
                alarm_triggered = True
                detected_object_name = detected_object_id(int(cls[pos]))

    if detected_object_name:
        cv2.putText(image_orig, "tespit edildi: " + detected_object_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Teste", image_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imgpath2 = "./testfoto/testpicture2.jpeg"
detect_objects_and_plot(imgpath2)




def trigger_alarm():
    winsound.Beep(1000, 500)

def detect_objects_in_real_time():
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Kamera açılamadı!!!")
        return

    alarm_triggered = False
    detected_object_name = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.5:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                    color = (0, int(cls[pos]), 255)
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                                cv2.LINE_AA)
                    if not alarm_triggered:
                        alarm_triggered = True
                        detected_object_name = detected_object_id(int(cls[pos]))
                        #trigger_alarm()

        if detected_object_name:
            cv2.putText(frame, "tespit edildi: " + detected_object_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('Gerçek Zamanlı Nesne Tespiti', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


detect_objects_in_real_time()
