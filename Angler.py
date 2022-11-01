import logging
import time

import numpy as np
import pyautogui
import mss
from PIL import Image
import cv2

from enum import Enum

import AnglerAI
import pyautogui

import Inventory
import TackleBox


class GameState(Enum):
    IDLE = 0,
    FISHING = 1,
    INVENTORY = 2,
    CASTING = 3,
    HIT = 4,
    EXHAUSTED = 5


class Angler:
    CAST_TIME = 0.925
    CAST_WAIT_TIME = 30

    def __init__(self):
        self.game_state = GameState.IDLE
        self.angler_ai = AnglerAI.AnglerAI()

        self.exclamation_anchor = cv2.imread("resources/exclamation_solid.jpg")
        self.exclamation_anchor = cv2.cvtColor(self.exclamation_anchor, cv2.COLOR_RGB2GRAY)

        self.exhaustion_anchor = cv2.imread("resources/exhausted_anchor.jpg")
        self.exhaustion_anchor = cv2.cvtColor(self.exhaustion_anchor, cv2.COLOR_RGB2GRAY)

        self.exhausted = False

        self.ff = None
        self.inventory = Inventory.Inventory()

    def run(self):
        while True:
            if self.game_state == GameState.IDLE:
                logging.info("Angler.run: game_state = %s | Going to cast" % "IDLE")
                self.cast()

            if self.game_state == GameState.EXHAUSTED:
                logging.info("Angler.run: game_state = %s | TOO TIRED. Breaking" % "EXHAUSTED")
                break

            if self.game_state == GameState.CASTING:
                logging.info("Angler.run: game_state = %s | Waiting for bite" % "CASTING")
                self.waitForBite()

            if self.game_state == GameState.HIT:
                logging.info("Angler.run: game_state = %s | Determining fish or trash" % "HIT")
                self.determineHit()

            if self.game_state == GameState.FISHING:
                logging.info("Angler.run: game_state = %s | AI catching fish" % "FISHING")

                self.angler_ai.catch_fish()
                time.sleep(15)
                pyautogui.mouseDown()
                pyautogui.mouseUp()
                time.sleep(3)

                logging.info("Angler.run: game_state = %s | Fishing over. Checking inventory" % "FISHING")
                self.determineInventory()

            if self.game_state == GameState.INVENTORY:
                logging.info("Angler.run: game_state = %s | Closing inventory" % "INVENTORY")
                self.manageInventory()

            if self.exhausted:
                break


    def cast(self):
        """
        Responsible for casting a perfect cast with the fishing rod
        :return:
        """
        self.game_state = GameState.CASTING

        # Move mouse to 1, 1 so that it doesn't cover the OK button
        pyautogui.moveTo(1, 1)

        logging.info("Angler.cast: game_state = %s | Casting line for %f sec" % (self.game_state, Angler.CAST_TIME))

        pyautogui.mouseDown()
        start_time = time.time()
        while time.time() - start_time < Angler.CAST_TIME:
            pass
        # pyautogui.sleep(Angler.CAST_TIME)
        pyautogui.mouseUp()

        self.checkForExhaustion()

    def checkForExhaustion(self):
        mon = {'left': 15, 'top': 850, 'width': 400, 'height': 150}
        image = TackleBox.getScreenshot(mon, gray=True)

        res = cv2.matchTemplate(image, self.exhaustion_anchor, cv2.TM_CCOEFF_NORMED)
        _, max_pct, _, max_loc = cv2.minMaxLoc(res)

        if max_pct > .90:
            self.game_state = GameState.EXHAUSTED
            logging.info("Angler.checkForExhaustion: game_state = %s | Player is exhausted" % self.game_state)



    def waitForBite(self):
        """
        Responsible for observing game until the "!" marker appears above head
        :return:
        """
        start_time = time.time()

        time_out = True

        while time.time() - start_time < Angler.CAST_WAIT_TIME:
            image = TackleBox.getScreenshot(gray=True)
            res = cv2.matchTemplate(image, self.exclamation_anchor, cv2.TM_CCOEFF_NORMED)
            _, max_pct, _, _ = cv2.minMaxLoc(res)

            if max_pct > 0.80:
                time_out = False
                break

        pyautogui.mouseDown()
        pyautogui.mouseUp()

        # If the "!" was never seen, go back to the idle state
        self.game_state = GameState.IDLE if time_out else GameState.HIT

    def determineHit(self):
        """
        Determines if the "!" that occured when fishing resulted in a fish or just garbage
        Just check to see if the fishing frame appears
        :return:
        """

        self.ff = TackleBox.waitForFF(timeout=3)
        if self.ff:
            logging.info("Angler.determineHit: game_state = %s | FISH FOUND" % self.game_state)
            self.game_state = GameState.FISHING
        else:
            logging.info("Angler.determineHit: game_state = %s | Hooked some garbage" % self.game_state)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            self.determineInventory()

    def determineInventory(self):
        """
        Checks to see if the inventory page is displayed
        :return:
        """
        self.game_state = GameState.INVENTORY if self.inventory.isInventoryDisplayed() > 0.95 else GameState.IDLE
        logging.info("Angler.determineHit: game_state = %s | Updated game_state" % self.game_state)

    def manageInventory(self):
        """
        Responsible for closing out the inventory menu
        It must do this by pressing the "OK" button

        TODO:
        Maybe allow for actual inventory management
        :return:
        """
        logging.info("Angler.manageInventory: game_state = %s | replacing garbage is applicable" % self.game_state)

        self.inventory.swapItemsWithGarbage()
        self.inventory.close()
        self.game_state = GameState.IDLE
