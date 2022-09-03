import cv2
import numpy as np
import mss
from PIL import Image
import os


class FishingFrame:
    fishing_frame = cv2.cvtColor(np.array(Image.open("resources/fishing_still_frame_only.jpg")), cv2.COLOR_BGR2RGB)
    fish_image = np.array(Image.open("resources/fish.jpg"))
    sct = mss.mss()

    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = FishingFrame.fishing_frame.shape[1]
        self.height = FishingFrame.fishing_frame.shape[0]

        self.bar_x = 70
        self.bar_w = 40
        self.bar_y = 19
        self.bar_h = 575 - 19
        self.top_left = 0
        self.bottom_left = 0

        self.image = np.array([[]])
        self.bar = np.array([[]])

        self.fish_position = [0, 0]

        self.update_coords(self.x, self.y)

    def find(self, image=None):
        if image is not None:
            self.image = image[self.y:self.y + self.height, self.x:self.x + self.width]
            self.bar = image[self.bar_y:self.bar_y + self.bar_h, self.bar_x:self.bar_x + self.bar_w, :]
            return self.image

        ss = FishingFrame.sct.grab(self.mon)

        self.image = Image.frombytes(
            'RGB',
            (ss.width, ss.height),
            ss.rgb,
        )
        self.bar = image[self.bar_y:self.bar_y + self.bar_h, self.bar_x:self.bar_x + self.bar_w, :]

        return self.image

    def update_coords(self, x, y):
        self.x = x
        self.y = y
        self.top_left = (self.x, self.y)
        self.bottom_left = (self.x + self.width, self.y + self.height)

        self.mon = {'left': self.x, 'top': self.y, 'width': self.width, 'height': self.height}

    def find_fish(self):
        print(self.bar.shape, FishingFrame.fish_image.shape)
        res = cv2.matchTemplate(self.bar, FishingFrame.fish_image, cv2.TM_CCOEFF_NORMED)
        _, _, _, self.fish_position = cv2.minMaxLoc(res)
        return self.fish_position

    def find_fishing_bar(self):
        pass

    def _fish_bottom_right(self):
        return self.fish_position[0] + self.fish_image.shape[0], self.fish_position[1] + self.fish_image.shape[1]

    def get_image(self):
        image = self.image

        #abs_fish = self.fish_position + np.array([self.bar_x, self.bar_y])
        #abs_fish_end = self._fish_bottom_right() + np.array([self.bar_x, self.bar_y])
        #image = cv2.rectangle(image, abs_fish, abs_fish_end, 255, 2)

        return image

    @staticmethod
    def create_from_image(image, threshold=0.7):
        res = cv2.matchTemplate(image, FishingFrame.fishing_frame, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        cv2.imshow("TEST", image)
        print(min_val, max_val, min_loc, max_loc)
        if max_val < threshold:
            return None

        frame = FishingFrame()
        frame.update_coords(*max_loc)
        frame.find(image)

        return frame


