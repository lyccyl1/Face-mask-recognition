from detector_counter import run
import cv2

if __name__ == "__main__":
  run()
 # # 用摄像头获取视频
 #    cap = cv2.VideoCapture(0)
 #    cap.set(3, 640)
 #    cap.set(4, 480)
 #
 #    cv2.namedWindow('tracking')
 #    # cv2.setMouseCallback('tracking', draw_boundingbox)
 #    # tracker = KCF.KCF(feature_type="gray")
 #    # boudingbox = None
 #    while(cap.isOpened()):
 #        ret, frame = cap.read()
 #
 #        cv2.imshow('tracking', frame)
 #        c = cv2.waitKey(1) & 0xFF
 #        # if c == 27 or c == ord('q'):
 #        #     break
 #
 #    cap.release()
 #    cv2.destroyAllWindows()