import logging
import os
import time

import cv2
import pyautogui

import TackleBox


class Inventory:
    """
    Class responsible for managing the players inventory
    Need to have UI Scale set to 100%


    TODO:
    Possible edge cases?
        Empty inventory slot and a chest that has multiple items. May just throw away everything
        Untested with larger backpacks

    """

    INVENTORY_TOP_MON = {'left': 550, 'top': 8, 'width': 800, 'height': 95}
    INVENTORY_BOT_MON = {'left': 550, 'top': 980, 'width': 800, 'height': 95}
    INVENTORY_PLAYER_MID_MON = {'left': 556, 'top': 504, 'width': 775, 'height': 205}
    INVENTORY_FISHING_MID_MON = {'left': 552, 'top': 193, 'width': 775, 'height': 205}

    GARBAGE_ITEM_DIR = "resources/garbage_items"

    COLS = 12
    ROWS = 3

    ITEM_THRESHOLD = 0.85

    def __init__(self):
        self.bad_items = []

        for f in os.listdir(Inventory.GARBAGE_ITEM_DIR):
            image = cv2.imread(os.path.join(Inventory.GARBAGE_ITEM_DIR, f))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            self.bad_items.append(image)

        self.blank_slot = cv2.imread("resources/blank_slot.jpg")
        self.blank_slot = cv2.cvtColor(self.blank_slot, cv2.COLOR_RGB2GRAY)

        self.trash_can = cv2.imread("resources/trash_can.jpg")
        self.trash_can = cv2.cvtColor(self.trash_can, cv2.COLOR_RGB2GRAY)

        self.inventory_anchor = cv2.imread("resources/inventory_anchor.jpg")
        self.inventory_anchor = cv2.cvtColor(self.inventory_anchor, cv2.COLOR_RGB2GRAY)

    def isInventoryDisplayed(self):
        image = TackleBox.getScreenshot(gray=True)

        res = cv2.matchTemplate(image, self.inventory_anchor, cv2.TM_CCOEFF_NORMED)
        _, max_pct, _, max_loc = cv2.minMaxLoc(res)

        logging.debug("Inventory.isInventoryDisplayed: max_pct = %f%% | max_loc = %s" % (max_pct, max_loc))
        return True if max_pct > 0.95 else False

    def getSlotMon(self, row, col, inventory="PLAYER"):
        mon = Inventory.INVENTORY_PLAYER_MID_MON if inventory == "PLAYER" else Inventory.INVENTORY_FISHING_MID_MON
        delta_x = mon["width"] // Inventory.COLS
        delta_y = mon["height"] // Inventory.ROWS

        slot_mon = {
            "left": mon["left"] + (delta_x * col),
            "top": mon["top"] + (delta_y * row),
            "width": delta_x,
            "height": delta_y
        }
        return slot_mon

    def getMiddleOfSlot(self, row, col, inventory="PLAYER"):
        logging.debug("Inventory.getInventoryImage: finding middle of %i, %i" % (row, col))
        slot_mon = self.getSlotMon(row, col, inventory)
        return TackleBox.middleOfMon(slot_mon)

    def getInventoryImage(self, row, col, inventory="PLAYER", gray=True):
        logging.debug("Inventory.getInventoryImage: grabbing image at %i, %i" % (row, col))
        slot_mon = self.getSlotMon(row, col, inventory)

        slot_image = TackleBox.getScreenshot(slot_mon, gray=gray)
        return slot_image

    def swapAndDispose(self, loc_1, loc_2):
        pyautogui.moveTo(*loc_1)
        pyautogui.mouseDown()
        pyautogui.mouseUp()
        pyautogui.sleep(0.1)
        pyautogui.moveTo(*loc_2)
        pyautogui.sleep(0.1)
        pyautogui.mouseDown()
        pyautogui.mouseUp()
        pyautogui.sleep(0.1)

        TackleBox.findAndClickImage(self.trash_can)

    def swapItemsWithGarbage(self):
        fishing_slots = list(self.findFilledSlotsInFishingInventory())
        garbage_slots = list(self.findGarbageInInventory())

        for fish_slot, garbage_slot in zip(fishing_slots, garbage_slots):
            fish_center = self.getMiddleOfSlot(*fish_slot, "FISHING")
            garbage_center = self.getMiddleOfSlot(*garbage_slot, "PLAYER")
            logging.info(
                "Inventory.swapItemsWithGarbage: swapping %s with %s" % (str(fish_center), str(garbage_center)))

            self.swapAndDispose(fish_center, garbage_center)

    def findFilledSlotsInFishingInventory(self):
        i = 0
        while True:
            row = i // Inventory.COLS
            col = i % Inventory.COLS

            inv_image = self.getInventoryImage(row, col, "FISHING")

            # Search for the first blank slot
            res = cv2.matchTemplate(inv_image, self.blank_slot, cv2.TM_CCOEFF_NORMED)
            _, max_pct, _, max_loc = cv2.minMaxLoc(res)
            if max_pct > Inventory.ITEM_THRESHOLD:
                break

            yield row, col
            i += 1

    def findGarbageInInventory(self):
        """
        Search through all the inventory to find the garbage items
        :return:
        """
        for row in range(Inventory.ROWS):
            for col in range(Inventory.COLS):
                for garbage_image in self.bad_items:
                    inv_image = self.getInventoryImage(row, col, "PLAYER")
                    res = cv2.matchTemplate(inv_image, garbage_image, cv2.TM_CCOEFF_NORMED)
                    _, max_pct, _, max_loc = cv2.minMaxLoc(res)
                    if max_pct > Inventory.ITEM_THRESHOLD:
                        yield row, col

    def findFirstGarbageInInventory(self):
        location = None
        for location in self.findGarbageInInventory():
            break
        return location

    def displayInventories(self):
        time.sleep(2)

        fishing_inventory = TackleBox.getScreenshot(Inventory.INVENTORY_FISHING_MID_MON, gray=True)
        player_inventory = TackleBox.getScreenshot(Inventory.INVENTORY_PLAYER_MID_MON, gray=True)

        delta = Inventory.INVENTORY_FISHING_MID_MON["width"] // 12
        for i in range(1, 13):
            start_point, end_point = (i * delta, 0), (i * delta, Inventory.INVENTORY_FISHING_MID_MON["height"])
            print(start_point, end_point)
            cv2.line(fishing_inventory, start_point, end_point, (0, 255, 0), 2)
            cv2.line(player_inventory, start_point, end_point, (0, 255, 0), 2)

        delta = Inventory.INVENTORY_FISHING_MID_MON["height"] // 3
        for i in range(1, 4):
            start_point, end_point = (0, i * delta), (Inventory.INVENTORY_FISHING_MID_MON["width"], i * delta)
            print(start_point, end_point)
            cv2.line(fishing_inventory, start_point, end_point, (0, 255, 0), 2)
            cv2.line(player_inventory, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow("Fish", fishing_inventory)
        cv2.imshow("Player", player_inventory)

        cv2.waitKey(100000)

    def close(self):
        TackleBox.findAndClickImage(self.inventory_anchor)
