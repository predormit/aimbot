import cv2
import torch
import time
import win32api
import win32con
import numpy as np
from ctypes import windll
import pyautogui
from mss import mss
from ultralytics import YOLO
from mouse_driver.MouseMove import mouse_move
from pynput.mouse import Button, Listener
import pydirectinput

pyautogui.FAILSAFE = False
class csgo:
        def __init__(self):
                self.model = YOLO("E:/ultralytics-main/runs/detect/train23/weights/best.pt")
                self.initialize_camera()
                self.auto_lock = True
                self.mouse_button_1 = 'right'
                self.mouse_button_2 = 'x2'
                self.auto_lock_button = 'middle'
                self.fire = False
                self.detect_length = 480
                self.locking = False
                self.pos_factor = 0.2
                self.visualization=True
                self.detect_center_x, self.detect_center_y = self.detect_length // 2, self.detect_length // 2
                listener = Listener(on_click=self.on_click)
                listener.start()
        def initialize_camera(self):
                self.sct = mss()
                detect_length = 480
                self.screen_width, self.screen_height = pyautogui.size()
                self.top, self.left=self.screen_height//2-detect_length//2,self.screen_width//2-detect_length//2

                self.monitor = {"top": self.top, "left": self.left, "width": detect_length, "height": detect_length}

        def get_screen(self):
                sct_img = self.sct.grab(self.monitor)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换为BGR格式
                return img

                #return cv2.cvtColor(np.asarray(self.sct.grab(self.monitor)), cv2.COLOR_BGR2RGB)

        def draw_boxes(self,img, results):
            for result in results:

                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = f"{self.model.names[cls]} {conf:.2f}"
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return img

        def on_click(self, x, y, button, pressed):
                # Turn on and off auto_lock
                if button == getattr(Button, self.auto_lock_button) and pressed:
                    if self.auto_lock:
                        self.auto_lock = False
                        print('---------------------Control is off.---------------------')
                    else:
                        self.auto_lock = True
                        print('---------------------Control is on.---------------------')

                # Press the left button to turn on auto aim
                if button in [getattr(Button, self.mouse_button_1)] and self.auto_lock:
                    if pressed:
                        self.locking = True
                        print('On...')
                    else:
                        self.locking = False
                        print('OFF')


                if button == getattr(Button, self.mouse_button_2) and pressed:
                    if pressed:
                        self.locking = True
                        print('On...')
                    else:
                        self.locking = False
                        print('OFF')

        def custom_sort(self,item):
            label_priority = {'1': 1, '3': 2}  #只锁头
            return (label_priority.get(item['label'], float('inf')), item['move_dis'])
        def sort_target(self, results):
            target_sort_list = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    #print(self.model.names[cls])
                    target_x, target_y = (x1 + x2) / 2, (y1 + y2) / 2# - self.pos_factor * (y2 - y1)
                    move_dis = ((target_x - self.detect_center_x) ** 2 + (target_y - self.detect_center_y) ** 2) ** (1 / 2)
                    #if label in self.args.enemy_list and conf >= self.args.conf and move_dis < self.args.max_lock_dis:
                    target_info = {'target_x': target_x, 'target_y': target_y, 'move_dis': move_dis, 'label': self.model.names[cls],
                                       'conf': conf}
                    if conf > 0.5 and (target_info['label'] == '1' or target_info['label'] == '3'):
                        target_sort_list.append(target_info)
            # Sort the list by label and then by distance
            return sorted(target_sort_list, key=self.custom_sort)


        def get_move_dis(self, target_sort_list):
            # Get the target with the lowest distance
            target_info = min(target_sort_list, key=lambda x: ( x['move_dis']))
            target_x, target_y, move_dis = target_info['target_x'], target_info['target_y'], target_info['move_dis']

            move_rel_x = (target_x - self.detect_center_x)
            move_rel_y = (target_y - self.detect_center_y)
            return move_rel_x, move_rel_y, move_dis

        def lock(self, target_sort_list):
                if len(target_sort_list) > 0 and self.locking:
                    target_info = target_sort_list[0]
                    print(target_info['target_x'], target_info['target_y'])
                    move_rel_x, move_rel_y, move_dis = self.get_move_dis(target_sort_list)

                    mouse_move(move_rel_x, move_rel_y)
                    #pydirectinput.moveRel(xOffset=int(move_rel_x),
                      #       yOffset=int(move_rel_y), relative=True)
                    if abs(move_rel_x + move_rel_y) < 8:
                        pydirectinput.click(button='left')
                 
        def forward(self):
            screen_img = self.get_screen() # 转换为BGR格式

            results = self.model(screen_img)

            #print(self.model.names)
            target_sort_list = self.sort_target(results)

            #print(target_sort_list)
            self.lock(target_sort_list)
            if self.visualization:
                screen_img = self.draw_boxes(screen_img, results)

                #显示图像
                cv2.imshow('CSGO2 Detection', screen_img)



if __name__ == '__main__':
    cs = csgo()
    while True:
        cs.forward()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()