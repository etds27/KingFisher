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
    anchor = np.array(cv2.imread("resources/fishing_anchor.jpg"))
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
        self.fb_upper = 0  # px
        self.fb_lower = 0  # px
        self.fb_upper_pct = 0
        self.fb_lower_pct = 0

        # Values for progress bar
        self.pb_x = 134
        self.pb_w = 15
        self.pb_y = 12
        self.pb_h = 570
        self.progress_px = self.pb_h
        self.progress = 0

        self.chest_pct = False
        self.chest_visible = False

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
            self.bar = cv2.cvtColor(self.bar, cv2.COLOR_BGR2RGB)
            self.progress_bar = self.image[self.pb_y:self.pb_y + self.pb_h, self.pb_x: self.pb_x + self.pb_w, :]
            self.progress_bar = cv2.cvtColor(self.progress_bar, cv2.COLOR_BGR2RGB)
            return self.image

        ss = FishingFrame.sct.grab(self.mon)

        self.image = np.asarray(Image.frombytes(
            'RGB',
            (ss.width, ss.height),
            ss.rgb,
        ))
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        if not self.is_frame_still_visible():
            return False

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

        inactive_lower = np.array([88, 148, 1]) - 27  # X value buffer
        # inactive_upper = np.array([160, 220, 189]) + 30
        inactive_upper = np.array([220, 230, 210])

        active_lower = active_target - tolerance // 2
        active_upper = active_target + tolerance // 2

        # bar = cv2.cvtColor(self.bar, cv2.COLOR_BGR2RGB)

        mask = np.zeros(self.bar.shape)

        # bar[np.where(np.all(np.logical_and(bar > blue_lower, bar < blue_upper), axis=-1))] = [0, 0, 0]
        mask[np.where(np.all(np.logical_and(self.bar > active_lower, self.bar < active_upper), axis=-1))] = [255, 255,
                                                                                                             255]
        mask[np.where(np.all(np.logical_and(self.bar > inactive_lower, self.bar < inactive_upper), axis=-1))] = [255,
                                                                                                                 255,
                                                                                                                 255]
        cv2.imshow("MASK", mask)

        w = np.where(np.sum(mask, -1).sum(-1) > 17500)
        if not len(w[0]):
            return
        self.fb_upper, self.fb_lower = w[0][0], w[0][-1]

        self.fb_upper_pct, self.fb_lower_pct = self.fb_upper / self.bar_h, self.fb_lower / self.bar_h
        logging.info("FishingFrame.find_fishing_bar: Bar found at %2f" % self.fb_upper)

    def find_progress(self):
        """
        Using a brown found in the progress bar, find the lowest occurance of that brown. That will be the top of the current progress
        :return:
        """
        target = np.array([103, 51, 30])
        tolerance = 10
        w = np.where(
            np.all(np.logical_and(self.progress_bar > target - tolerance, self.progress_bar < target + tolerance),
                   axis=-1))

        mask = np.zeros(self.progress_bar.shape)
        mask[np.where(
            np.all(np.logical_and(self.progress_bar > target - tolerance, self.progress_bar < target + tolerance),
                   axis=-1))] = [255, 255, 255]

        self.progress_px = w[0][-1]
        self.progress = 1 - (self.progress_px / self.pb_h)

    def find_chest(self):
        pass

    def find_all(self):
        self.find_fish()
        self.find_progress()
        self.find_fishing_bar()
        self.find_chest()

    def _fish_bottom_right(self):
        return self.fish_position[0] + self.fish_h, self.fish_position[1] + self.fish_w

    def is_frame_still_visible(self):
        """
        Determine if the fishing frame is still displayed on the screen by looking for the fishing pole anchor image
        :return:
        """

        res = cv2.matchTemplate(self.image, FishingFrame.anchor, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > 0.75:
            logging.debug("FishingFrame.is_frame_still_visible: Frame is visible")
            return True
        logging.debug("FishingFrame.is_frame_still_visible: Frame is not visible")
        return False

    def get_params(self):
        return {
            "fish_location": self.fish_pct,
            "bar_upper_pct": self.fb_upper_pct,
            "bar_lower_pct": self.fb_lower_pct,
            "progress": self.progress
        }

    def get_image(self):
        image = self.image.copy()

        image = cv2.rectangle(image, (self.bar_x, self.bar_y), (self.bar_x + self.bar_w, self.bar_y + self.bar_h), 255,
                              2)
        image = cv2.rectangle(image, (self.pb_x, self.pb_y), (self.pb_x + self.pb_w, self.pb_y + self.pb_h), 255, 2)

        abs_fish = self.fish_position + np.array([self.bar_x, self.bar_y])
        abs_fish_end = self._fish_bottom_right() + np.array([self.bar_x, self.bar_y])
        image = cv2.rectangle(image, abs_fish, abs_fish_end, (255, 0, 255), 2)

        image = cv2.line(image, (self.bar_x, self.fb_upper + self.bar_y),
                         (self.bar_x + self.bar_w, self.fb_upper + self.bar_y), (255, 255, 0), 2)
        image = cv2.line(image, (self.bar_x, self.fb_lower + self.bar_y),
                         (self.bar_x + self.bar_w, self.fb_lower + self.bar_y), (255, 255, 0), 2)

        image = cv2.line(image, (self.pb_x, self.pb_y + self.progress_px),
                         (self.pb_x + self.pb_w, self.pb_y + self.progress_px), (0, 0, 255), 2)

        image = cv2.putText(image, "%i %%" % (self.progress * 100), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                            2, cv2.LINE_AA)
        return image

    def __str__(self):
        s = ""
        s += "%-20s: %f\n" % ("Fish Location", self.fish_pct)
        s += "%-20s: %f\n" % ("Bar Upper", self.fb_upper)
        s += "%-20s: %f\n" % ("Bar Upper", self.fb_lower)
        s += "%-20s: %f\n" % ("Percent", self.progress)
        s += "%-20s: %f\n" % ("Chest", self.chest_pct if self.chest_visible else "None")
        return s

    @staticmethod
    def create_from_image(image=None, threshold=0.7):

        res = cv2.matchTemplate(image, FishingFrame.fishing_frame, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # cv2.imshow("TEST", image)
        print(max_val)
        if max_val < threshold:
            return None

        # print("Found frame!")
        frame = FishingFrame()
        frame.update_coords(*max_loc)
        frame.find(image)

        return frame
