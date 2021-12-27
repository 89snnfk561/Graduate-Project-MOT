
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QFileDialog, QPushButton, QVBoxLayout, QMainWindow, QTableWidget, QDialog
from PyQt5.QtCore import QRect, Qt, QLine, QTimer, QThread, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.uic import loadUi

import copy
import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from deep_sort_pytorch.deep_sort.counting import Counting, Line

import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

drawing_frame = []
lines = []
pic_coef = 1
Videofilename = ""

class Main(QMainWindow):

    def __init__(self):

        super().__init__()
        self.setWindowTitle("主窗口")
        button = QPushButton("選擇影片", self)
        button.clicked.connect(self.show_VP)

        button_2 = QPushButton("畫線", self)
        button_2.clicked.connect(self.show_draw)
        button_2.setGeometry(0, 30, 100, 30)

        button_3 = QPushButton("執行", self)
        button_3.clicked.connect(self.show_Run)
        button_3.setGeometry(0, 60, 100, 30)

        self.VP_window = SelectPreviewVideo()
        self.draw_window = DrawLineGUI()
        self.Run_window = RunForm()

    def show_VP(self):
        self.VP_window.show()
    def show_draw(self):
        self.draw_window.show()
    def show_Run(self):
        self.Run_window.show()


class MouseControlDrawLine(QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False
    n = 0
    line = QLine

    #鼠标点击事件
    def mousePressEvent(self,event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()

    #鼠标释放事件
    def mouseReleaseEvent(self,event):
        self.n = self.n+1

        print(self.line)

        self.flag = False

    #鼠标移动事件
    def mouseMoveEvent(self,event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    #绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QRect(self.x0, self.y0, (self.x1-self.x0), (self.y1-self.y0))
        self.line = QLine(self.x0, self.y0, self.x1, self.y1)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red,2,Qt.SolidLine))
        # painter.drawRect(rect)
        painter.drawLine(self.line)


class DrawLineGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.path = ""

        self.button = QPushButton('Show picture')
        self.button.clicked.connect(self.openVideoPicture)
        self.button_saveLine = QPushButton('Save line')
        self.button_saveLine.clicked.connect(self.save_line)
        self.button_undo = QPushButton('Undo')
        self.button_undo.clicked.connect(self.Undo)

        self.button.setShortcut((str("Space")))
        self.button_saveLine.setShortcut((str("Ctrl+S")))
        self.button_undo.setShortcut((str("Ctrl+Z")))

        self.resize(200, 200)
        self.setWindowTitle('在label中畫線')
        self.lb = MouseControlDrawLine(self)  #重定义的label
        self.table = QTableWidget()
        self.Textlabel = QLabel("", self)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.lb)
        # self.layout.addWidget(self.Textlabel)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.button_saveLine)
        self.layout.addWidget(self.button_undo)
        self.setLayout(self.layout)


    def Update(self):
        self.Textlabel.setText(self.lb.textString)
        self.update()

    def openVideoPicture(self):
        self.path = "temp.jpg"
        self.img = cv2.imread(self.path)
        cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
        self.original_img = self.img.copy()
        self.UpdateWidgets()

    def UpdateWidgets(self):
        height, width, bytesPerComponent = self.img.shape
        self.resize(60 + width, 50 + height)
        self.lb.setGeometry(QRect(30, 30, width, height))

        bytesPerLine = 3 * width

        self.QImg = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(self.QImg)

        self.lb.setPixmap(pixmap)
        self.lb.setCursor(Qt.CrossCursor)
        # self.Textlabel.setGeometry(QRect(30, 30 + height, 30 + width, 30))

    def qline_to_point(self):
        pt1 = (self.lb.line.x1(), self.lb.line.y1())
        pt2 = (self.lb.line.x2(), self.lb.line.y2())
        return [pt1, pt2]

    def point_to_qline(self, line):
        qline = QLine(line[0][0], line[0][1], line[1][0], line[1][1])
        return qline

    def save_line(self):
        lines.append(self.qline_to_point())
        # QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        print(lines)
        self.UpdateLine()

    def Undo(self):
        if len(lines) > 0:
            lines.pop()
        print(lines)
        self.UpdateLine()

    def UpdateLine(self):
        self.img = self.original_img.copy()
        for line in lines:
            print(line)
            self.img = cv2.line(self.img, line[0], line[1], (0, 0, 255), 3)
        self.UpdateWidgets()


class SelectPreviewVideo(QDialog):

    def __init__(self, parent=None):
        super(SelectPreviewVideo, self).__init__(parent)
        loadUi('./PythonGUI/VP.ui', self)
        self.frame_count = 0
        self.timer_camera = QTimer()  # 定义定时器
        video = 'C:/Users/Kenny/Videos/隧道.mp4'  # 加载视频文件
        self.cap = cv2.VideoCapture(video)
        self.pushButton_start.clicked.connect(self.slotStart)  # 按钮关联槽函数
        self.pushButton_stop.clicked.connect(self.slotStop)
        self.pushButton_select_file.clicked.connect(self.openFileNameDialog)
        # self.slider_video.connect(self.frame_count)

    def slotStart(self):
        """ Slot function to start the progamme
        """

        self.timer_camera.start(100)
        self.timer_camera.timeout.connect(self.openFrame)

    def slotStop(self):
        """ Slot function to stop the programme
        """
        print(self.frame_count)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)
        ret, drawing_frame = self.cap.read()
        if ret:
            cv2.imwrite("temp.jpg", drawing_frame)
            # print(drawing_frame)
        # self.cap.release()
        self.timer_camera.stop()  # 停止计时器


    def openFrame(self):
        """ Slot function to capture frame and process it
        """
        if (self.cap.isOpened()):
            ret, frame = self.cap.read()

            print(self.frame_count)

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame.data, width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.label_frame.width(), self.label_frame.height())
                self.label_frame.setPixmap(QPixmap.fromImage(q_image))


            else:
                self.frame_count = 0
                self.cap.release()
                self.timer_camera.stop()  # 停止计时器
            self.frame_count = self.cap.get(1)

    def openFileNameDialog(self):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                                                  "Video files(*.mp4 *.avi)", options=options)
        if fileName:
            self.cap.release()
            self.timer_camera.stop()
            print(fileName)
            self.path = fileName
            global Videofilename
            Videofilename = copy.copy(fileName)
            print(Videofilename)
            self.cap = cv2.VideoCapture(self.path)
            self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1


class RunForm(QMainWindow):
    def __init__(self, parent=None):
        super(RunForm, self).__init__(parent)
        loadUi('./PythonGUI/RunWindow.ui', self)
        # 實例化線程
        self.work = WorkThread()
        self.runButton.clicked.connect(self.start_thread)
        self.stopButton.clicked.connect(self.stop_thread)

    def start_thread(self):
        # 启动线程
        self.work.start()
        # 線程自訂議信號連接的槽函數
        self.work.trigger.connect(self.display)
        self.work.trigger2.connect(self.line_display)

    def stop_thread(self):
        self.work.stop()

    def display(self, str):
        # 由于自定义信号时自动传递一个字符串参数，所以在这个槽函数中要接受一个参数
        self.listWidget.addItem(str)
        self.listWidget.scrollToBottom()

    def line_display(self, str):
        # 由于自定义信号时自动传递一个字符串参数，所以在这个槽函数中要接受一个参数
        self.listWidget.addItem(str)
        self.listWidget.scrollToBottom()


class WorkThread(QThread):

    trigger = pyqtSignal(str)
    trigger2 = pyqtSignal(str)


    def __init__(self):
        super(WorkThread, self).__init__()
        self.text = ""
        self._mutex = QMutex()

    def run(self):
        self.detect()

    def stop(self):
        self.setTerminationEnabled(True)

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        self._mutex.lock()

        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        label %= 2 ** 5
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        self._mutex.unlock()
        return tuple(color)



    def detect(self):
        # out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        #     opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        #     opt.save_txt, opt.img_size, opt.evaluate

        with torch.no_grad():
            out = 'inference/output'
            source = 'inference/input/MOT16-02.mp4'
            yolo_weights = 'yolov5/yolov5s.pt'
            deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
            show_vid = True
            save_vid = True
            save_txt = False
            imgsz = 640
            evaluate = False
            config_deepsort = "deep_sort_pytorch/configs/deep_sort.yaml"
            augment = False
            device = ''

            iou_thres = 0.5
            conf_thres = 0.4
            classes = [0, 2, 3, 5, 7]
            agnostic_nms = False

            if Videofilename is not "":
                source = Videofilename

            webcam = source == '0' or source.startswith(
                'rtsp') or source.startswith('http') or source.endswith('.txt')

            # initialize deepsort
            cfg = get_config()
            cfg.merge_from_file(config_deepsort)
            attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
            deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                use_cuda=True)

            # Initialize
            device = select_device(device)

            # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
            # its own .txt file. Hence, in that case, the output folder is not restored
            if not evaluate:
                if os.path.exists(out):
                    pass
                    shutil.rmtree(out)  # delete output folder
                os.makedirs(out)  # make new output folder

            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size

            if half:
                model.half()  # to FP16

            # Set Dataloader
            vid_path, vid_writer = None, None
            # Check if environment supports image displays
            if show_vid:
                show_vid = check_imshow()

            if webcam:
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            else:
                dataset = LoadImages(source, img_size=imgsz)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            t0 = time.time()

            save_path = str(Path(out))
            # extract what is in between the last '/' and last '.'
            txt_file_name = source.split('\\')[-1].split('.')[0]
            txt_path = str(Path(out)) + '\\' + txt_file_name + '.txt'

            count = Counting(cls_names=names, classes=classes, lines=lines)

            for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_sync()
                pred = model(img, augment=augment)[0]

                # Apply NMS
                pred = non_max_suppression(
                    pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
                t2 = time_sync()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                    else:
                        p, s, im0 = path, '', im0s

                    s += '%gx%g ' % img.shape[2:]  # print string
                    save_path = str(Path(out) / Path(p).name)

                    annotator = Annotator(im0, line_width=2, pil=not ascii)

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        xywhs = xyxy2xywh(det[:, 0:4])
                        confs = det[:, 4]
                        clss = det[:, 5]

                        # pass detections to deepsort
                        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)
                        count.ClearTrail(deepsort.tracker._next_id)

                        # draw boxes for visualization
                        if len(outputs) > 0:

                            for j, (output, conf) in enumerate(zip(outputs, confs)):

                                bboxes = output[0:4]
                                id = output[4]
                                cls = output[5]

                                c = int(cls)  # integer class
                                label = f'{id} {names[c]} {conf:.2f}'

                                color = self.compute_color_for_labels(id)
                                annotator.box_label(bboxes, label, color=color)

                                count.TrackTail(bbox=bboxes, track_id=id)
                                count.DrawTrail(img=im0, track_id=id, color=color)
                                count.updateCounting(cls=cls, track_id=id)

                                if save_txt:
                                    # to MOT format
                                    bbox_top = output[0]
                                    bbox_left = output[1]
                                    bbox_w = output[2] - output[0]
                                    bbox_h = output[3] - output[1]
                                    # Write MOT compliant results to file
                                    with open(txt_path, 'a') as f:
                                        f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_top,
                                                                       bbox_left, bbox_w, bbox_h, -1, -1, -1,
                                                                       -1))  # label format

                            count.TrackZero(track_ids=outputs[:, 4])
                    else:
                        deepsort.increment_ages()



                    # Print time (inference + NMS)
                    print('%sDone. (%.3fs)' % (s, t2 - t1))
                    self.trigger.emit(str(s))

                    # Stream results
                    count.DrawAllLine(canvas=im0, color=(0, 0, 255), color2=(255, 0, 0))
                    count.printCounting(canvas=im0, color=(0, 0, 0))
                    im0 = annotator.result()


                    if show_vid:

                        cv2.imshow(p, im0)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration

                    # Save results (image with detections)
                    if save_vid:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'

                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)



            if save_txt or save_vid:
                print('Results saved to %s' % os.getcwd() + os.sep + out)
                if platform == 'darwin':  # MacOS
                    os.system('open ' + save_path)

            print('Done. (%.3fs)' % (time.time() - t0))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    x = Main()
    x.show()
    sys.exit(app.exec_())
