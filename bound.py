import math
import cv2

class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.pt1 = (x, y)
        self.pt2 = (x + w, y + h)
        self.middle = (self.x + self.w // 2, self.y + self.h // 2)
        self.area = self.w * self.h


    def __repr__(self):
        return "({}, {}, {}, {})".format(self.x, self.y, self.w, self.h)


    def get_overlap_ratio(self, other):
        overlap_width = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        overlap_height = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        overlap_area = overlap_height * overlap_width

        return overlap_area / self.area


    def merge(self, other):
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y

        return BoundingBox(x, y, w, h)


    def get_distance(self, other):
        dx = self.middle[0] - other.middle[0]
        dy = self.middle[1] - other.middle[1]

        return math.sqrt(dx ** 2 + dy ** 2)


    def draw(self, img, color, thickness):
        cv2.rectangle(img, (self.x, self.y), (int(self.x + self.w), int(self.y + self.h)), color, thickness)
