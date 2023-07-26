'''
步骤:
  1. 使用Mx_yolov3训练好模型
  2. 将 .tflite 文件转成 K210模型文件 .kmodel 文件
  3. 将 .kmodel 文件保存至SD卡
  4. 修改改代码的 .kmodel 文件路径
  5. 运行改代码即可
'''


import sensor, image, lcd, time
import KPU as kpu
from fpioa_manager import fm
from machine import UART

fm.register(16, fm.fpioa.UART1_RX, force=True)
fm.register(18, fm.fpioa.UART1_TX, force=True)
uart = UART(UART.UART1, 115200, read_buf_len = 4096) # 初始化串口


# 初始化LCD freq：LCD(实际上指 SPI 的通讯速率)的频率, 最大值: 15000000
lcd.init(freq = 15000000)
lcd.rotation(2)
sensor.reset() # 初始化单目摄像头
sensor.set_pixformat(sensor.RGB565) # 设置摄像头输出格式 要与训练模型一致
sensor.set_framesize(sensor.QVGA)
sensor.set_vflip(1)    # 摄像头后置模式
sensor.run(1)

# .kmodel 文件路径 格式: /sd/文件名.kmodel
task = kpu.load('/sd/number.kmodel')
anchor = (1.2354, 0.8828, 1.4911, 1.2493, 1.7308, 1.6297, 1.9877, 2.0355, 2.2831, 2.7267)
sensor.set_windowing((224, 224))

kpu_value = kpu.init_yolo2(task, 0.6, 0.3, 5, anchor)

classset = ['3', '4', '8', '7', '5', '6', '2', '1']

while(1):
  # 拍摄这一张照片并且返回图像对象
  img = sensor.snapshot()

  code = kpu.run_yolo2(task, img)

  if code:
    print("识别到code\n")
    for i in code:
      print("i = ", i)
      print("i.rect:", i.rect())
      print("i.x:", i.x())
      print("i.y:", i.y())
      print("i.x1:", i.x())
      print("i.y:", i.y()+12)
      print("i.value:", i.value())

      a = img.draw_rectangle(i.rect(), color = [208, 20, 26], thickness = 5)
      a = lcd.display(img)

      list1 = list(i.rect())
      print(list1)

      # 中心位置坐标
      b = (list1[0] + list1[2]) / 2
      c = (list1[1] + list1[3]) / 2

      #x1 = list1[0]
      #x2 = list1[2]
      #y1 = list1[1]
      #y2 = list1[3]

      print("识别到的物体:", classset[i.classid()])
      print(type(classset[i.classid()]))

      uart.write('(' + classset[i.classid()] + ')') #数据回传

      print("概率为:", 100.00 * i.value())
      print("坐标为:", b, c)

      for i in code:
        # 显示识别标签
        lcd.draw_string(i.x(), i.y(), classset[i.classid()], lcd.GREEN, lcd.WHITE)
        # 显示置信度
        lcd.draw_string(i.x(), i.y()+12, '%f'%i.value(), lcd.YELLOW, lcd.WHITE)

  else:
    print("没有识别到code\n")
    a = lcd.display(img)
  time.sleep_ms(50)
a = kpu.deinit(task)
