import math
from random import randrange
import numpy as np
from datetime import datetime
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class TrackerCounter:
  people_count = 0
  mask_count = 0
  nomask_count = 0
  uncertain_count = 0

  def __init__(self):
    # Store information for each object in a dictionary
    # values: [cx (int), cy (int), has_been_before_boundary (boolean), has_been_counted_before (boolean), list_of_up_to_10_predictions (float [-1, 1])]
    self.center_points = {}

    # each time a new object id detected, the count will increase by one
    self.id_count = 0

    self.detection_data = []

  def update(self, frame, objects_rect, label_probs, W, H, left_boundary_line, uncertain_interval=0.2, dist_same_obj=100):
    '''
    object_rect: List of objects' coordinates = [(xstart1, ystart1, xend1, yend1), (xstart2, ystart2, xend2, yend2), ...]
    label_probs: List of objects' label "probabilities" = [-0.9, 0.7, ....]
    W, H: frame width and height

    dist_same_obj:
    - It is the maximum eucledian distance from the previous object in order to be considered the same object
    - dist_same_obj determines how far the detection has to be from the old one to be considered a new object.
    - If dist_same_obj is too low we might get false positives when an object is moving.
    - It also depends on the amount of pixels so needs different value when resolution changes.
    '''

    # List of info of all objects in the frame (this is what the method returns)
    objects_infos = []

    # NB: If dist_same_obj is too low then fast moving faces are seen as a new face in each frame and averaging classification is not done OR even worse they are not counted at all (because a new face appears after the boundary)
    # NB: If dist_same_obj is too high then if a face disappears at frame 15 and a new face appears at frame 16 (it is only a problem if this happens in consecutive frames since memory is not implemented i.e. we got 1 frame memory)
    # then the new face will be seen as the old face and might not be tracked
    # NB: This is very unlikely to be an issue when memory = 1 as it is now so it is a lot safer to have a high dist_same_obj than a low one.
    dist_same_obj=(W+H)/10


    counter=0 # This variable represents how many faces of previous frame have been matched to faces of the new frame (e.g. if counter = 2 it means that 2 faces from the previous frame have been matched to 2 of the new frame). This makes sure two or more faces are not labelled the same

      # Loop through all faces and their labels
    for rect, prob in zip(objects_rect, label_probs):
        xstart, ystart, xend, yend = rect
        # Get center point of new object
        cx = (xstart + xend) // 2
        cy = (ystart + yend) // 2


        '''Find out if same face was detected'''
        same_object_detected = False
        # For all objects in previous frame
        for face_id, pt in self.center_points.items():

            # calculate eucledian distance from previously detected face
            dist = math.hypot(cx - pt[0], cy - pt[1])

            # 1st condition: If it is the same object (ditsance < dist_same_obj) update previous objects location to this one's (keeps the object id).
            # 2nd condition: But create a NEW object if all faces from previous frame have been matched up already (i.e. we if we've run out of faces to match).
            # otherwise youll get the bug where when 2 people show up it is Face 1 and Face 1 as if they are the same face. So current # of objects has to be <= previous number of objects to match to previous objects which makes sense.
            if dist < dist_same_obj and counter < len(self.center_points):
                self.center_points[face_id][0] = cx      # update center x
                self.center_points[face_id][1] = cy      # update center y

                # Append classification probability to this face's classification history
                frames_to_avg_over = 10
                if len(pt[4]) < frames_to_avg_over:
                    self.center_points[face_id][4].append(prob)

                # If there are already 10 probabilities in the list then replace one at random (because the code for cycling through them one by one seems too complex in this case)
                else:
                    self.center_points[face_id][4][randrange(frames_to_avg_over)] = prob

                # Add object info to list to return
                objects_infos.append([xstart, ystart, xend, yend, face_id])
                same_object_detected = True

                counter+=1

                break

        '''New face detected'''
        # New object is detected: assign ID to that object
        if same_object_detected == False:
            self.id_count += 1
            self.center_points[self.id_count] = [cx, cy, False, False, [prob]]

            # Add new object to list to return
            objects_infos.append([xstart, ystart, xend, yend, self.id_count])

    # Clean the dictionary by center points to remove IDS not used anymore
    new_center_points = {}
    for object_info in objects_infos:
        _, _, _, _, object_id = object_info
        center = self.center_points[object_id]
        new_center_points[object_id] = center

    # Update dictionary with IDs not used removed
    self.center_points = new_center_points.copy()

    # People counter
    for face_id, pt in self.center_points.items():
        # Note if object has been before the boundary line
        if pt[0] > left_boundary_line:
            self.center_points[face_id][2] = True

        # Note if the object
        # 1. has been detected after the boundary
        # 2. has been detected before the boundary (can remove this condition, it is just to ensure that someone who appears from the bottom is not counted) and
        # 3. has never been counted before
        # Then update it to having been counted and increase people count by 1
        # NB: pt[0] = xcenter, pt[1] = ycenter
        if (pt[0] < left_boundary_line) and (pt[2] == True) and pt[3] == False:
            self.center_points[face_id][3] = True
            self.people_count += 1

            div = 3 # the larger div is the smaller the perpetrator frame will be

            # labeled count
            avg_prob = np.mean(pt[4])

            current_time = datetime.now()

            if avg_prob > 0. + uncertain_interval:
                self.mask_count += 1

                # TODO: Only for testing, remove
                # Save and display image of masked person
                print(f'MASK ({current_time})')
                self.detection_data.append(['mask', current_time])
                # Catch exception where face is moving too fast and towards bottom left and perpetrator[0] or perpetrator[1] (width or height or both) ends up being 0.
                perpetrator=frame[0:H, 0:left_boundary_line + W//5]
                perpetrator = cv2.cvtColor(perpetrator, cv2.COLOR_BGR2RGB)
                perpetrator = Image.fromarray(perpetrator, 'RGB')
                plt.imshow(perpetrator)
                # plt.show()
                str_year = current_time.strftime("%Y_")
                str_month = current_time.strftime("%m_")
                str_day = current_time.strftime("%d_")
                str_time = current_time.strftime("%H_%M_%S")
                path = './output/'+f'MASK{str_year+str_month+str_day+str_time}'+'.jpg'
                plt.savefig(path, dpi=750, bbox_inches='tight')

            elif avg_prob < -0. - uncertain_interval:
                self.nomask_count += 1

                # Save and display image of unmasked person
                print(f'NO MASK ({current_time})')
                self.detection_data.append(['no_mask', current_time])
                # Catch exception where face is moving too fast and towards bottom left and perpetrator[0] or perpetrator[1] (width or height or both) ends up being 0.
                perpetrator=frame[0:H, 0:left_boundary_line + W//5]
                perpetrator = cv2.cvtColor(perpetrator, cv2.COLOR_BGR2RGB)
                perpetrator = Image.fromarray(perpetrator, 'RGB')
                plt.imshow(perpetrator)
                # plt.show()
                str_year = current_time.strftime("%Y_")
                str_month = current_time.strftime("%m_")
                str_day = current_time.strftime("%d_")
                str_time = current_time.strftime("%H_%M_%S")
                path = './output/' + f'NO_MASK{str_year + str_month + str_day + str_time}' + '.jpg'
                plt.savefig(path, dpi=750, bbox_inches='tight')

            else:
                self.uncertain_count += 1

                # Save and display image of potentially unmasked person
                print(f'UNCERTAIN ({current_time})')
                self.detection_data.append(['uncertain', current_time])
                # Catch exception where face is moving too fast and towards bottom left and perpetrator[1] (the height of the image) ends up being 0.
                perpetrator=frame[0:H, 0:left_boundary_line + W//5]
                perpetrator = cv2.cvtColor(perpetrator, cv2.COLOR_BGR2RGB)
                perpetrator = Image.fromarray(perpetrator, 'RGB')
                plt.imshow(perpetrator)
                # plt.show()
                str_year = current_time.strftime("%Y_")
                str_month = current_time.strftime("%m_")
                str_day = current_time.strftime("%d_")
                str_time = current_time.strftime("%H_%M_%S")
                path = './output/' + f'UNCERTAIN{str_year + str_month + str_day + str_time}' + '.jpg'
                plt.savefig(path, dpi=750, bbox_inches='tight')
    return objects_infos
