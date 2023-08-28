import numpy as np
import time
import math
# Class save Ship object

class Ship():
    def __init__(self, tlwh, score, track_id):
        self.is_move = True
        self.bbox = tlwh
        self.pre_bbox = [0, 0, 0, 0]
        self.timestart = 0
        self.track_id = track_id
        self.frame_start = 0
        self.is_activate = False
        self.score = score
        self.off_frame = 0

    def change_state_move(self):
        self.is_move = not self.is_move
        self.timestart = time.time()

    def activate(self, frame_id):
        self.is_activate = True
        self.frame_start = frame_id
        self.off_frame = 0

    def update(self, bbox, score):
        self.bbox = bbox
        self.score = score

    def deactivate(self):
        self.is_activate = False
        self.off_frame += 1

    def update_bbox(self, bbox):
        self.pre_bbox = bbox

    def __repr__(self):
        return 'Ship_{}_{}'.format(self.track_id, self.is_move)

# Class manage ship object


class Ship_manager():
    def __init__(self, time_remove=30):
        self.list_ship = []
        self.time_remove = time_remove

    # Update frame to list
    def update(self, ArrayStrack, frame_id):
        old_ids = np.array([i.track_id for i in self.list_ship])
        new_ids = np.array([i.track_id for i in ArrayStrack])

        for strack in ArrayStrack:
            if strack.track_id in old_ids:
                # update
                exist = np.where(old_ids == strack.track_id)
                self.list_ship[exist[0][0]].activate(frame_id)
                self.list_ship[exist[0][0]].update(strack.tlwh, strack.score)
            else:
                # ship_new = Ship()
                ship = Ship(strack.tlwh, strack.score, strack.track_id)
                ship.activate(frame_id)
                self.list_ship.append(ship)

        for pos, ship in enumerate(self.list_ship):
            # exist=np.where(new_ids==ship.track_id)
            if ship.track_id not in new_ids:
                ship.deactivate()
                if ship.off_frame > self.time_remove:
                    self.list_ship = remove(self.list_ship, pos)
    # Check state of object, if the difference of both iou and center of bbox is too large, change state to MOVING,
    # otherwise STOP.

    def check_state(self):
        for ship in self.list_ship:
            iou = get_iou(ship.bbox, ship.pre_bbox)
            dis_center = get_discenter(ship.bbox, ship.pre_bbox)
            if iou > 0.75:
                if ship.is_move == True:
                    ship.change_state_move()
            else:
                if dis_center > 15:
                    if ship.is_move == False:
                        ship.change_state_move()

    # Update pre_bbox 
    def update_bbox(self, ArrayStrack):
        old_ids = np.array([i.track_id for i in self.list_ship])
        for strack in ArrayStrack:
            if strack.track_id in old_ids:
                exist = np.where(old_ids == strack.track_id)
                self.list_ship[exist[0][0]].update_bbox(strack.tlwh)

# Get distance between 2 box center


def get_discenter(box1, box2):
    x1 = box1[0]+box1[2]/2
    y1 = box1[1]+box1[3]/2
    x2 = box2[0]+box2[2]/2
    y2 = box2[1]+box2[3]/2
    dis = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dis

# Get IOU between 2 box

def get_iou(box1, box2, epsilon=1e-5):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height
    area_a = box1[2] * box1[3]
    area_b = box2[2] * box2[3]
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou

# Remove object at position pos

def remove(ships, pos):
    array = []
    for i in range(len(ships)):
        if i != pos:
            array.append(ships[i])
    return array
