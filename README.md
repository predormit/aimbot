## 数据集：CS2.v1i.yolov8 包含了我使用的数据集的一部分
### train23 ： 剪枝前的模型
### step_0_fintune2 : 剪枝后的模型
### test.py:检测视频
### screen.py：对屏幕实时检测
### aimbot.py:自瞄
#### 直接python ./aimbot.py
### 使用`prune_v8.py`前要往`ultralytics-main\ultralytics\utils\loss.py`的第192行插入下述代码（用来对模型剪枝）
  `mydevice=torch.device('cuda:0')`
  `self.proj=self.proj.to(mydevice)`
