# -*- coding: utf-8 -*-
"""
Created on Tue May  7 22:33:08 2019

@author: JCMat
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:57:27 2019

@author: JCMat
"""

# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from my_ai_brain import DeepQNetwork

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# represents our Q-function
brain = DeepQNetwork(5,3,0.9)
#index=0:go straight index=1:turn left index=2:turn right
action2rotation = [0,20,-20]
last_reward = 0
scores = []

# if there was no proir run initialize all the below variables
initial_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global initial_update
    sand = np.zeros((screen_width,screen_height))
    goal_x = 20
    goal_y = screen_height - 20
    initial_update = False

# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        #taking the sum of the pixel values and then dividing by 400 number of pixxels to obtain the density of sand
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>screen_width-10 or self.sensor1_x<10 or self.sensor1_y>screen_height-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>screen_width-10 or self.sensor2_x<10 or self.sensor2_y>screen_height-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>screen_width-10 or self.sensor3_x<10 or self.sensor3_y>screen_height-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    #evn = None

#    def __init__(self, *args, **kwargs):
#        Widget.__init__(self, *args, **kwargs)
#        self.ellipse_pos_x = 50
#        self.ellipse_pos_y = 50
#        self.ellipse_pos = (self.ellipse_pos_x, self.ellipse_pos_y)
#        self.ellipse_width = 50
#        self.ellipse_height = 50
#        self.ellipse_size = (self.ellipse_width, self.ellipse_height)
#        self.move = 5
#        self.evn = Clock.schedule_interval(self.update, 0.01)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
#updating the simulation
    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global screen_width
        global screen_height
        #obtaining width of the window-longuer
        screen_width = self.width
        #obtaining height of the screen-screen_height
        screen_height = self.height
        if initial_update:
            init()
        #to find orientation of the car 
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
#        print("current scores")
#        print(scores)
        rotation = action2rotation[action]
        self.car.move(rotation)
        #calculating the distance of the car from its goal points
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        
        
        #drawing the source and destination
        self.ellipse_pos_x = self.width - goal_x
        self.ellipse_pos_y = self.height - goal_y
        self.ellipse_pos = (self.ellipse_pos_x, self.ellipse_pos_y)
        self.ellipse_width = 50
        self.ellipse_height = 50
        self.ellipse_size = (self.ellipse_width, self.ellipse_height)
        
        with self.canvas:
            Color(rgb=(0, 255, 0))
            Ellipse(pos=self.ellipse_pos, size=self.ellipse_size)
#            if self.ellipse_pos_x + self.ellipse_width >= self.width:
#                self.move = -5
#            elif self.ellipse_pos_x <= 0:
#                self.move = +5
#            self.ellipse_pos_x += self.move
#            self.ellipse_pos = (self.ellipse_pos_x, self.ellipse_pos_y)
        
        
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.6,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            print(touch)

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        #clearbtn = Button(text = 'clear', pos = (4 * parent.width, 0))
        #savebtn = Button(text = 'save', pos = (parent.width, 0))
        #loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        loadbtn = Button(text = 'load', pos = (parent.width, 0))
        grabbtn = Button(text = 'grab')
        #clearbtn.bind(on_release = self.clear_canvas)
        #savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        grabbtn.bind(on_release = self.grab)
        parent.add_widget(self.painter)
        #parent.add_widget(clearbtn)
        parent.add_widget(grabbtn)
        #parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent
        
    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((screen_width,screen_height))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()
    
    def grab(self, obj):
        print("grabbing screenshot and results")
        print("current scores")
        print(scores[-1])
        import win32gui
        import win32ui
        import win32con
        import win32api
        # grab a handle to the main desktop window
        hdesktop = win32gui.GetDesktopWindow() 
        # determine the size of all monitors in pixels
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN) 
        # create a device context
        desktop_dc = win32gui.GetWindowDC(hdesktop)
        img_dc = win32ui.CreateDCFromHandle(desktop_dc)
        # create a memory based device context
        mem_dc = img_dc.CreateCompatibleDC() 
        # create a bitmap object
        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(img_dc, width, height)
        mem_dc.SelectObject(screenshot) 
        # copy the screen into our memory device context
        mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top),win32con.SRCCOPY) 
        # save the bitmap to a file
        screenshot.SaveBitmapFile(mem_dc, 'C:\\Users\\JCMat\\Desktop\\project\\working_code\\Screenshots.bmp')
        # free our objects
        mem_dc.DeleteDC()
        win32gui.DeleteObject(screenshot.GetHandle())
        plt.plot(scores)
        plt.show()


# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
