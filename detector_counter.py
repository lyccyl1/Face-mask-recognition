import numpy as np
import pandas as pd
import seaborn as sns
import imutils
from imutils.video import VideoStream
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

from tracker_counter import TrackerCounter
# from detect_and_classify import detect_and_classify


def run():
  classifier = tf.keras.models.load_model('mask_classifier_mobilenet.h5')

  prototxtPath = r"face_detector_ssd_caffee/deploy.prototxt" # configurations file 
  weightsPath = r"face_detector_ssd_caffee/res10_300x300_ssd_iter_140000.caffemodel" # weights file
  detector = cv2.dnn.readNet(prototxtPath, weightsPath) # load saved model

  # vs = VideoStream('./scy.mp4').start()
  vs = cv2.VideoCapture('Y:\github\mask_detector_people_counter-master\CODE\s.mp4')
  # vs = cv2.VideoCapture(0)
  # cv2.namedWindow('Frame')
  tracker = TrackerCounter()

  # Initialise total tracked faces and fps to 0 
  idd = 0
  fps_start_time = 0
  fps = 0


  # Parameters
  W = 720
  H = 1280
  left_boundary_line = int(0.3*W)

  # Store starting time for periodic data extraction
  start_time = datetime.now()
  hrs = 1 # every how many hours should the data be exported

  ret =True
  # loop over the frames from the video stream
  while ret == True:
      ret, frame = vs.read()
      # cv2.imshow("tracking",frame)
      '''VISUALS'''

      # Resize the frame
      # NB: cv2.resize lets you choose height and width, imutils.resize only lets you choose width but preserves ratio
      # frame = cv2.resize(frame, dsize=(x, y))
      # frame = cv2.resize(frame, (W, H))
      # print(frame.shape)
      # H, W = frame.shape[:2] # used in detect_and_classify also

      # Calculate ideal font scale
      scale = 0.030                      # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
      ideal_font_size = min(W, H)/(25/scale)

      # Define box locations
      yy = int(ideal_font_size*35)
      box1start = np.array((0, 0))
      box1end = np.array((int(0.40*W), yy*2))
      box2start = np.array((int(0.40*W), 0))
      box2end = np.array((W, int(yy*4.5)))

      cv2.rectangle(frame, pt1=box1start, pt2=box1end, color=(255, 255, 255), thickness=-1)
      cv2.rectangle(frame, pt1=box2start, pt2=box2end, color=(255, 255, 255), thickness=-1)

      '''DETECT & CLASSIFY'''
      # Detect and Classify for each frame
      locs, preds = detect_and_classify(frame, detector, classifier, W=W, H=H)

      '''VISUALS'''

      label_probs = [] # This is for tracker_counter to average over multiple classifications
      # Live Counter
      num_of_masked = 0
      num_of_unmasked = 0
      num_of_uncertain = 0

      # Define classification uncertainty interval
      uncertain_interval = 0.2# 0.5 means 50+% probability of a class for classification.

      # loop over face locations and mask predictions
      for box, pred in zip(locs, preds):

          # unpack the bounding box and predictions
          startX, startY, endX, endY = box
          mask, no_mask = pred

          # 1. Determine the class label 2. Add colour (BGR)
          if mask >= 0.5 + uncertain_interval:
              label = 'Mask'
              colour = (0, 255, 0)
              num_of_masked +=1
              # for tracker_counter to average over
              label_probs.append(mask)

          elif no_mask >= 0.5 + uncertain_interval:
              label = 'No Mask'
              colour = (0, 0, 255)
              num_of_unmasked +=1
              # for tracker_counter to average over
              label_probs.append(-no_mask)

          elif (mask >= 0.5) and (mask <= 0.5 + uncertain_interval):
              label = 'Uncertain'
              colour = (0, 255, 255)
              num_of_uncertain +=1
              # for tracker_counter to average over
              label_probs.append(mask)

          elif (no_mask >= 0.5) and (no_mask <= 0.5 + uncertain_interval):
              label = 'Uncertain'
              colour = (0, 255, 255)
              num_of_uncertain +=1
              # for tracker_counter to average over
              label_probs.append(-no_mask)

          # probability and text to display
          probability = max(mask, no_mask) * 100
          label_text = f'{label}: {probability:.1f}%'

          # 1. Display label
          cv2.putText(img=frame, text=label_text, org=(startX, startY - 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=colour, thickness=2)
          # 2. Display bounding box
          cv2.rectangle(img=frame, pt1=(startX, startY), pt2=(endX, endY), color=colour, thickness=2)

      '''TRACK & COUNT'''

      # objects_info = [object1_info, object2_info, ...] where object_info = [Xstart, Ystart, Xend, Yend, id_of_object]
      objects_info = tracker.update(frame, locs, label_probs, W, H, left_boundary_line, uncertain_interval=uncertain_interval, dist_same_obj=(W + H / 14))

      '''VISUALS'''

      # for all objects
      for object_info in objects_info:
          Xstart, Ystart, Xend, Yend, idd = object_info

          # ID of face
          cv2.putText(img=frame, text=f'Face {idd}', org=(Xstart, Ystart-40), fontScale=1.4, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(155, 149, 24), thickness=2)

      # Calculate fps
      fps_end_time = time.time()
      time_diff = fps_end_time - fps_start_time
      fps = int(1/time_diff)
      fps = f'FPS: {fps}'
      fps_start_time = fps_end_time

      # Calculate current time and export data
      time_difference = (datetime.now() - start_time).seconds/3600

      if  time_difference >= hrs:
        _export_data(tracker)

        # reset time
        start_time = datetime.now()

      # VISUALS: Display FPS and Current Time
      cv2.putText(img=frame, text=_get_current_time_str(), org=(box1start[0] + box1start[0]//14, yy), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 0, 0), thickness=2)
      cv2.putText(img=frame, text=fps, org=(box1start[0] + box1start[0]//14, int(yy*1.8)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 0, 0), thickness=2)

      # VISUALS: Display boundary line
      # cv2.line(img=frame, pt1=(left_boundary_line, 0), pt2=(left_boundary_line, H), color=(45, 174, 102), thickness=5)

      # VISUALS: Display live people counter
      cv2.putText(img=frame, text=f'     People Count: {tracker.people_count}', org=(box2start[0] + box2start[0]//15, yy), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 0, 0), thickness=2)
      percent_masked = 0
      percent_unmasked = 0
      percent_uncertain = 0
      if tracker.people_count != 0:
          percent_masked = np.round(tracker.mask_count / tracker.people_count * 100, 1)
          percent_unmasked = np.round(tracker.nomask_count / tracker.people_count * 100, 1)
          percent_uncertain = np.round(tracker.uncertain_count / tracker.people_count * 100, 1)
      cv2.putText(img=frame, text=f'Masked:       {tracker.mask_count} ({percent_masked}%)', org=(box2start[0] + box2start[0]//15, yy*2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 155, 0), thickness=2)
      cv2.putText(img=frame, text=f'Unmasked:    {tracker.nomask_count} ({percent_unmasked}%)', org=(box2start[0] + box2start[0]//15, int(yy*3)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 0, 155), thickness=2)
      cv2.putText(img=frame, text=f'Uncertain:     {tracker.uncertain_count} ({percent_uncertain}%)', org=(box2start[0] + box2start[0]//15, int(yy*4)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=ideal_font_size, color=(0, 155, 155), thickness=2)

      '''END STREAM'''

      # Show the output frame in real-time
      cv2.imshow("Frame", frame)

      # Terminate if `q` is pressed. waitKey(0): keeps image still until a key is pressed. waitKey(x) it will wait x miliseconds each frame
      key = cv2.waitKey(10) & 0xFF
      if key == ord("q"):
          break
  '''EXPORT RESULTS'''
  _export_data(tracker)


  # Cleanup
  vs.stop()
  cv2.destroyAllWindows()


def _export_data(tracker):
    current_time_export = _get_current_time_str().replace(':', '.')

    #Export csv file and summarise results
    detection_data_exp = pd.DataFrame(data=tracker.detection_data, columns=['Label', 'Datetime'])
    detection_data_exp.to_csv(f'{current_time_export}_detection_data.csv', index=False)

    # Unburden memory
    tracker.detection_data = []

    # Summarise and visualise results
    num_of_people = detection_data_exp.shape[0]
    if num_of_people > 0:
        mask = detection_data_exp[detection_data_exp.Label == 'mask'].shape[0]
        no_mask = detection_data_exp[detection_data_exp.Label == 'no_mask'].shape[0]
        uncertain = detection_data_exp[detection_data_exp.Label == 'uncertain'].shape[0]

        print(f'{num_of_people} people.')
        print(f'{mask} mask.')
        print(f'{no_mask} no_mask.')
        print(f'{uncertain} uncertain.')

        # PIE CHART
        dpi=110
        fig = plt.figure(figsize=(8, 6), dpi=dpi)

        if uncertain != 0:
            plt.pie(x=[mask, no_mask, uncertain], labels=['Mask', 'No Mask', 'Uncertain'], colors=['green', 'red', 'yellow'], startangle=90, autopct='%1.1f%%', textprops={'fontsize': 14})
        if uncertain == 0: 
            plt.pie(x=[mask, no_mask], labels=['Mask', 'No Mask'], colors=['green', 'red'], startangle=90, autopct='%1.1f%%', textprops={'fontsize': 14})

        plt.title(f'Total People: {num_of_people}', fontweight='bold', fontsize=15)
        plt.legend()
        plt.show()

        fig.savefig(f'{current_time_export}_face_covering_pie_chart.png', dpi = dpi)

    else:
        print('No people detected.')
        pass


def _get_current_time_str():
    return str(datetime.now())[:-7]


def detect_and_classify(frame, detector, classifier, W, H, conf=0.3):
    '''
    Input: video frame, face detection model, face mask classification model
    Output: detector bounding boxes, classifier predictions, bounding boxes for tracker

    detect faces in frame and perform mask classification
    turn frame into blob (essentially preprocessing: 1. mean subtraction, 2. scaling, 3. optionally channel swapping)
    '''

    RGB = (104.0, 177.0,
           123.0)  # Source and intuition: "Deep Learning and Mean Subtraction": https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    pixels = 224
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(pixels, pixels), mean=RGB)

    ''' DETECTION '''
    # Detect Faces
    detector.setInput(blob)
    detections = detector.forward()
    num_of_detections = detections.shape[2]

    # initialise list of faces, their locations, list of predictions for our mask classifier
    faces = []
    locs = []
    preds = []

    # loop over all face detections
    for i in range(num_of_detections):

        # Live Counter to display on video feed
        # num_of_masked = 0
        # num_of_unmasked = 0
        # num_of_uncertain = 0

        # extract the confidence (probability) in the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is  greater than the minimum confidence
        if confidence > conf:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(W - 1, endX), min(H - 1, endY))

            # Extract Face
            face = frame[startY:endY, startX:endX]  # 1. extract the face region of interest (ROI)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # 2. convert it from BGR to RGB channel ordering
            face = cv2.resize(face, (224, 224))  # 3. resize it to 224x224
            face = tf.keras.preprocessing.image.img_to_array(face)  # 4. preprocess it for the mask classifier
            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)

            # append face and bounding box to lists
            faces.append(face)
            locs.append(np.array([startX, startY, endX, endY]))

    ''' CLASSIFICATION '''
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference make batch predictions on *all* faces at the same time rather than one-by-one predictions in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = classifier.predict(faces, batch_size=32, verbose=0)

    return locs, preds

