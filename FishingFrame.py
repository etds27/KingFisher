import logging

import cv2
import numpy as np
import mss
from PIL import Image
import os


logging.basicConfig(level=logging.INFO)

class FishingFrame:
    fishing_frame = cv2.cvtColor(np.array(Image.open("resources/fishing_still_frame_only.jpg")), cv2.COLOR_BGR2RGB)
    fish_image = np.array(Image.open("resources/fish.jpg"))
    sct = mss.mss()

    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = FishingFrame.fishing_frame.shape[1]
        self.height = FishingFrame.fishing_frame.shape[0]

        # Fish coordinates
        self.fish_position = [0, 0]
        self.fish_pct = 0
        self.fish_h = FishingFrame.fish_image.shape[0]
        self.fish_w = FishingFrame.fish_image.shape[1]

        # Coordinates for blue water bar within the game's frame
        self.bar_x = 70
        self.bar_w = 42
        self.bar_y = 19
        # self.bar_h = 551
        self.bar_h = 555

        # Values for fishing bar
        self.fb_pct = 0
        self.fb_upper = 0 # px
        self.fb_lower = 0 # px

        # Values for progress bar
        self.pb_x = 134
        self.pb_w = 15
        self.pb_y = 12
        self.pb_h = 570
        self.progress_px = self.pb_h
        self.progress = 0

        self.top_left = 0
        self.bottom_right = 0

        self.image = np.array([[]])
        self.bar = np.array([[]])
        self.progress_bar = np.array([[]])
        self.update_coords(self.x, self.y)

    def find(self, image=None):
        if image is not None:
            self.image = image[self.y:self.y + self.height, self.x:self.x + self.width]
            self.bar = self.image[self.bar_y:self.bar_y + self.bar_h, self.bar_x:self.bar_x + self.bar_w, :]
            self.progress_bar = self.image[self.pb_y:self.pb_y + self.pb_h,  self.pb_x: self.pb_x + self.pb_w, :]
            return self.image

        ss = FishingFrame.sct.grab(self.mon)

        self.image = np.asarray(Image.frombytes(
            'RGB',
            (ss.width, ss.height),
            ss.rgb,
        ))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.bar = self.image[self.bar_y:self.bar_y + self.bar_h, self.bar_x:self.bar_x + self.bar_w, :]
        self.bar = cv2.cvtColor(self.bar, cv2.COLOR_BGR2RGB)
        self.progress_bar = self.image[self.pb_y:self.pb_y + self.pb_h, self.pb_x: self.pb_x + self.pb_w, :]
        self.progress_bar = cv2.cvtColor(self.progress_bar, cv2.COLOR_BGR2RGB)

        return self.image

    def update_coords(self, x, y):
        self.x = x
        self.y = y
        self.top_left = (self.x, self.y)
        self.bottom_right = (self.x + self.width, self.y + self.height)

        self.mon = {'left': self.x, 'top': self.y, 'width': self.width, 'height': self.height}

    def find_fish(self):
        #print(self.bar.shape, FishingFrame.fish_image.shape)
        res = cv2.matchTemplate(self.bar, FishingFrame.fish_image, cv2.TM_CCOEFF_NORMED)
        _, _, _, self.fish_position = cv2.minMaxLoc(res)
        self.fish_pct = self.fish_position[1] / float(self.bar_h)
        logging.info("FishingFrame.find_fish: Fish found at %2f" % self.fish_pct)
        return self.fish_position

    def find_fishing_bar(self):
        """
        Find the players fishing bar by using color identification

        Start by subtracting the blue fishing background # Not needed
        Assume all colors are between the inactive transparent user bar when at the top and at the bottom
        Create an empty mask to determine bar bounds
        Search for colors between these two bounds and set the mask to true

        Sum the RGB values of mask and then sum by row. Since white is the highest absolute RGB value, we can find a
            threshold for the sum of rows that determine whether the bar covers the row
        Take the top and bottom instance of bar rows as the bounds of the fishing bar.

        Seems to work well on all test images so far

        Important color RGB
        Low Blue: [70, 114, 223] # Not needed
        High Blue: [175, 208, 243] # Not needed
        Low Inactive Green : [88, 148, 159]
        High Inactive Green: [152, 202, 165]
        Active: [130, 229, 1]

        :return:
        """
        tolerance = 25

        active_target = np.array([130, 229, 1])  # Active Color

        blue_lower = np.array([70, 114, 223]) - 10  # Small 10 value buffer
        blue_upper = np.array([175, 208, 243]) + 10  # Small 10 value buffer

        inactive_lower = np.array([88, 148, 159]) - 27  # X value buffer
        inactive_upper = np.array([160, 220, 189]) + 25

        active_lower = active_target - tolerance // 2
        active_upper = active_target + tolerance // 2

        # bar = cv2.cvtColor(self.bar, cv2.COLOR_BGR2RGB)

        mask = np.zeros(self.bar.shape)

        # bar[np.where(np.all(np.logical_and(bar > blue_lower, bar < blue_upper), axis=-1))] = [0, 0, 0]
        mask[np.where(np.all(np.logical_and(self.bar > active_lower, self.bar < active_upper), axis=-1))] = [255, 255, 255]
        mask[np.where(np.all(np.logical_and(self.bar > inactive_lower, self.bar < inactive_upper), axis=-1))] = [255, 255, 255]

        w = np.where(np.sum(mask, -1).sum(-1) > 10000)
        self.fb_upper, self.fb_lower = w[0][0], w[0][-1]
        logging.info("FishingFrame.find_fishing_bar: Bar found at %2f" % self.fb_upper)

    def _fish_bottom_right(self):
        return self.fish_position[0] + self.fish_h, self.fish_position[1] + self.fish_w

    def get_image(self):
        image = self.image.copy()

        image = cv2.rectangle(image, (self.bar_x, self.bar_y), (self.bar_x + self.bar_w, self.bar_y + self.bar_h), 255, 2)
        image = cv2.rectangle(image, (self.pb_x, self.pb_y), (self.pb_x + self.pb_w, self.pb_y + self.pb_h), 255, 2)

        abs_fish = self.fish_position + np.array([self.bar_x, self.bar_y])
        abs_fish_end = self._fish_bottom_right() + np.array([self.bar_x, self.bar_y])
        image = cv2.rectangle(image, abs_fish, abs_fish_end, (255, 0, 255), 2)

        image = cv2.line(image, (self.bar_x, self.fb_upper + self.bar_y), (self.bar_x + self.bar_w, self.fb_upper + self.bar_y), (255, 255, 0), 2)
        image = cv2.line(image, (self.bar_x, self.fb_lower + self.bar_y), (self.bar_x + self.bar_w, self.fb_lower + self.bar_y), (255, 255, 0), 2)

        image = cv2.line(image, (self.pb_x, self.pb_y + self.progress_px), (self.pb_x + self.pb_w, self.pb_y + self.progress_px), (0, 0, 255), 2)
        return image

    def find_progress(self):
        target = np.array([141, 81, 47])
        tolerance = 50

        w = np.where(np.all(np.logical_and(self.progress_bar > target - tolerance, self.progress_bar < target + tolerance), axis=-1))

        print(w)
        self.progress_px = w[0][-1]
        self.progress = 1 - (self.progress_px / self.pb_h)

    @staticmethod
    def create_from_image(image=None, threshold=0.7):

        res = cv2.matchTemplate(image, FishingFrame.fishing_frame, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # cv2.imshow("TEST", image)
        #print(max_val)
        if max_val < threshold:
            return None

        #print("Found frame!")
        frame = FishingFrame()
        frame.update_coords(*max_loc)
        frame.find(image)

        return frame


