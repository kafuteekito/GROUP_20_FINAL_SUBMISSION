import time
import cv2
from retinaface import RetinaFace

def detect_faces_retinaface(bgr_img):
    detections = RetinaFace.detect_faces(bgr_img)
    boxes = []
    if isinstance(detections, dict):
        for info in detections.values():
            x1, y1, x2, y2 = info["facial_area"]
            score = float(info.get("score", 0.0))
            boxes.append((x1, y1, x2, y2, score))
    boxes.sort(key=lambda x: x[4], reverse=True)
    return boxes

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    fps_counter = 0
    fps_t0 = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        boxes = detect_faces_retinaface(frame)
        for (x1, y1, x2, y2, score) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        fps_counter += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_counter / dt
            fps_counter = 0
            fps_t0 = time.time()

        cv2.putText(frame, f"RetinaFace | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("RetinaFace Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
