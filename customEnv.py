from mss import mss
import pyautogui
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete

class Game(Env):
    def __init__(self):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        
        # setup game capture frame locations
        self.cap = mss()
        self.game_location = {'top': 670, 'left': 450, 'width': 1000, 'height': 350}
        self.done_location = {'top': 600, 'left': 860, 'width': 400, 'height': 100}

    def step(self, action):
        action_map = {
            0:'space',
            1:'down',
            2:'no_op'
        }
        if action < 2:
            # press the button for the action given
            pyautogui.press(action_map[action])

        # check if game over
        done, done_cap = self.get_done()
        # get the next step
        new_state = self.get_observation()
        # set reward
        reward = 1
        info = {}
        return new_state, reward, done, info
    
    # restart the game
    def reset(self):
        time.sleep(1)
        pyautogui.click(x=250, y=250)
        pyautogui.press('space')
        return self.get_observation()

    def get_observation(self):
        # capture the screen in the defined area
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        # convert image to grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resize
        resized = cv2.resize(gray, (100,83))
        # put channels at the first (for stable baselines) and (width, height)
        channel = np.reshape(resized, (1,83,100))
        return channel
    
    def get_done(self):
        # capture the screen in the defined area for the 'Game Over' text
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        done_strings = ['GAME', 'GAHE']
        done=False

        # read the text from the image (apply OCR) and extract the first 4 characters
        res = pytesseract.image_to_string(done_cap)[:4]
        # if the first 4 characters are either "GAME" or "GAHE", end the episode
        if res in done_strings:
            done = True
        return done, done_cap