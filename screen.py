import cv2
import torch
import numpy as np
from mss import mss
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO("E:/ultralytics-main/runs/detect/train23/weights/best.pt")
# 设置截屏参数
sct = mss()
monitor = {"top": 160, "left": 0, "width": 800, "height": 640}  # 设置你需要截屏的区域


def grab_screen(monitor):
    sct_img = sct.grab(monitor)
    img = np.array(sct_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换为BGR格式
    return img


def draw_boxes(img, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


while True:
    # 截屏
    screen_img = grab_screen(monitor)

    # YOLOv8模型推理
    results = model(screen_img)


    # 绘制检测结果

    screen_img = draw_boxes(screen_img, results)
    target_sort_list=[]
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            print(cls)
            print(model.names[cls])
            target_x, target_y = (x1 + x2) / 2, (y1 + y2) / 2
            target_info = {'target_x': target_x, 'target_y': target_y,
                           'label': model.names[cls],
                           'conf': conf}
            target_sort_list.append(target_info)
    # 显示图像
    cv2.imshow('CSGO2 Detection', screen_img)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()