import cv2

class SwitchCamera:
    def __init__(self):
        self.curr_camera_index = 0
        self.available_cameras = self.list_cameras(10)
        if not self.available_cameras:
            raise RuntimeError("No cameras found!")
        self.cap = cv2.VideoCapture(self.available_cameras[self.curr_camera_index], cv2.CAP_AVFOUNDATION)
    
    def list_cameras(self, max_tested=10):
        available_cameras = []
        for index in range(max_tested):
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()
        return available_cameras

    def choose_camera(self, debug=False):
        """Display camera feed and allow user to switch with A/D keys, press Enter to confirm.
        """
        print("Camera Selection: Use A/D keys to switch, ENTER to confirm, ESC to cancel")

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            camera_id = self.available_cameras[self.curr_camera_index]
            status_text = f"Camera {camera_id} ({self.curr_camera_index + 1}/{len(self.available_cameras)})"
            cv2.putText(frame, status_text, (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, "Use A/D keys to switch | ENTER to confirm | ESC to cancel", (30, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("Camera Selection", frame)
            
            key = cv2.waitKey(30) & 0xFF
            if debug:
                print(f"key={key}")

            key8 = key & 0xFF
            if debug:
                print(f"raw key={key} key8={key8}")

            # A: previous camera
            if key in (ord('a'), ord('A')):
                self.curr_camera_index = (self.curr_camera_index - 1) % len(self.available_cameras)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.available_cameras[self.curr_camera_index], cv2.CAP_AVFOUNDATION)
                print(f"Switched to camera {self.available_cameras[self.curr_camera_index]}")

            # D: next camera
            elif key in (ord('d'), ord('D')):
                self.curr_camera_index = (self.curr_camera_index + 1) % len(self.available_cameras)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.available_cameras[self.curr_camera_index], cv2.CAP_AVFOUNDATION)
                print(f"Switched to camera {self.available_cameras[self.curr_camera_index]}")

            # ENTER: confirm selection (handle 13 and 10)
            elif key in (13, 10):
                cv2.destroyWindow("Camera Selection")
                print(f"Selected Camera {self.available_cameras[self.curr_camera_index]}")
                return self.cap

            # ESC: cancel
            elif key == 27:
                self.cap.release()
                cv2.destroyWindow("Camera Selection")
                return None