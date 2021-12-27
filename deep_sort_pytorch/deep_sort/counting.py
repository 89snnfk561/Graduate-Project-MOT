import numpy as np
import math
import cv2
from _collections import deque
from PyQt5.QtCore import QMutex


class Line:

    def __init__(self, x1, y1, x2, y2, classes, totalclass):
        self.totalclass = totalclass
        self.classes = classes
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.A = 0
        self.B = 0
        self.C = 0
        self.getA_B_C()
        self.go_right = [0 for _ in range(self.totalclass)]
        self.go_left = [0 for _ in range(self.totalclass)]
        self.confirm_track_id = deque(maxlen=30)

    def getA_B_C(self):
        # Ax + By + C
        if self.x1 == self.x2:
            self.A = 0
            self.B = 1
            self.C = -self.x1
        else:
            self.A = int((self.y1 - self.y2) / (self.x1 - self.x2))
            self.B = -1
            self.C = int(-self.A*self.x1 + self.B*self.y1)

    def center(self):
        return int((self.x1+self.x2)/2), int((self.y1+self.y2)/2)

    def drawLine(self, canvas, color):
        cv2.line(canvas, pt1=(self.x1, self.y1), pt2=(self.x2, self.y2), color=color, thickness=5)

    def intersection(self, start, end):
        x3, y3 = start
        x4, y4 = end
        a = self.x2 - self.x1
        b = x3 - x4
        c = self.y2 - self.y1
        d = y3 - y4
        g = x3 - self.x1
        h = y3 - self.y1
        f = a * d - b * c   #行列式
        if math.fabs(f) < 1.0e-06:
            return False
        t = float(d*g-b*h)/f
        s = float(-c*g+a*h)/f
        if 0>t or t>1:
            return False
        if 0>s or s>1:
            return False

        # dbx, dby 兩線段相交的點
        dbX = self.x1+t*(self.x2-self.x1)
        dbY = self.y1+t*(self.y2-self.y1)

        return True

    def isLeft(self, point):
        x, y = point
        return ((self.x2 - self.x1) * (y - self.y1) - (self.y2 - self.y1) * (x - self.x1)) > 0

    def updateLine(self, start, end, cls, track_id):

        if track_id in self.confirm_track_id:
            return

        if self.intersection(start=start, end=end):
            if self.isLeft(point=end):
                self.go_right[cls] = self.go_right[cls]+1
                print("GoRight ", cls, "class + 1")
            else:
                self.go_left[cls] = self.go_left[cls]+1
                print("GoRight ", cls, "class + 1")
            self.confirm_track_id.append(track_id)

    def printLine(self):
        print("Point 1 :  x=", self.x1, "y=", self.y1)
        print("Point 2 :  x=", self.x2, "y=", self.y2)
        print("Ax+By+C=0: ", self.A, "x+", self.B, "y+", self.C, "=0")
        print("Center  :  x, y = ", self.center())

    def Counttext(self):
        temptext = []
        for cls in self.classes:
            temptext.append([cls, self.go_right[cls], self.go_left[cls]])
        return temptext

    def Counttext_position(self):
        if self.x1 < self.x2:
            x = self.x1
        else:
            x = self.x2
        if self.y1 < self.y2:
            y = self.y1
        else:
            y = self.y2

        return (x, y)


class Group:
    def __init__(self):
        self.Lines = []

    pass


class Counting:

    def __init__(self, cls_names, classes, lines):
        self.Lines = []
        self.Groups = []
        self.n = 1000
        self.pts = [deque(maxlen=30) for _ in range(self.n)]
        self.linesIntersection = []
        self.cls_names = cls_names
        self.classes = classes
        self.totalclass = len(cls_names)
        self._mutex = QMutex()

        for line in lines:
            self.AddLine(line[0], line[1])

    def printAllLine(self):
        for line in self.Lines:
            print(line)

    def AddLine(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        line = Line(x1=x1, y1=y1, x2=x2, y2=y2, classes=self.classes, totalclass=self.totalclass)
        self.Lines.append(line)

    def DelLine(self, number_of_line):
        self.Lines.pop(number_of_line)

    def TrackTail(self, bbox, track_id):
        track_id %= self.n
        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        self.pts[track_id].append(center)

    def TrackZero(self, track_ids):
        for track_id in range(self.n):
            if track_id not in track_ids:
                self.pts[track_id].append((0, 0))

    def DrawTrail(self, img, track_id, color):
        track_id %= self.n
        for j in range(1, len(self.pts[track_id])):
            if self.pts[track_id][j - 1] is None or self.pts[track_id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 2)

            if self.pts[track_id][j] != (0, 0):
                for i in range(1, j+1):
                    if self.pts[track_id][j-i] != (0, 0):
                        cv2.line(img, (self.pts[track_id][j-i]), (self.pts[track_id][j]), color, thickness)
                        break


    def ClearTrail(self, _next_id):
        self._mutex.lock()
        _next_id += 10
        for j in range(10):
            self.pts[(_next_id+j) % self.n].clear()
        self._mutex.unlock()

    def DrawAllLine(self, canvas, color, color2):
        self._mutex.lock()
        for line in self.Lines:
            if line in self.linesIntersection:
                line.drawLine(canvas, color2)
            else:
                line.drawLine(canvas, color)
        self._mutex.unlock()
        self.linesIntersection.clear()


    def updateCounting(self, cls, track_id):

        track_id %= self.n
        for line in self.Lines:
            length = len(self.pts[track_id])
            if length > 2 and self.pts[track_id][length-1] != (0, 0):
                end = self.pts[track_id][length-1]
                for i in range(1, length-1):
                    if self.pts[track_id][length-1-i] != (0, 0):
                        start_n = length-1-i
                        start = self.pts[track_id][start_n]
                        line.updateLine(start=start, end=end, cls=cls, track_id=track_id)
                        if line.intersection(start=start,end=end):
                            self.linesIntersection.append(line)
                        break


    def printCounting(self, canvas, color):
        for line in self.Lines:
            x, y = line.Counttext_position()
            for text in line.Counttext():

                s = "{:^9}: {:3}/{:3}".format(self.cls_names[text[0]], text[1], text[2])
                cv2.putText(canvas, s, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
                y += 18

    def stringCounting(self):
        s = ""
        for i, line in enumerate(self.Lines):
            s += "line " + str(i) + "\n"
            for text in line.Counttext():
                s += "{:^9}: {:3}/{:3}\n".format(self.cls_names[text[0]], text[1], text[2])
        return s




def InitCanvas(width, height, color=(255, 255, 255)):
    canvas = np.ones((height, width, 3), dtype="uint8")
    canvas[:] = color
    return canvas


# if __name__ == "__main__":
#     start = (150, 00)
#     end = (150, 300)
#     canvas = InitCanvas(300, 300)
#     L = Line(300, 150, 0, 150)
#     L.printLine()
#     L.drawLine(canvas=canvas, color=(0, 0, 255))
#     L.updateLine(start=start, end=end, cls=1, track_id=1)
#     cv2.line(canvas, pt1=start, pt2=end, color=(0,0,0))
#     print(L.intersection(start, end))
#     print(L.isLeft(end))
#     print(L.go_left[1], L.go_right[1])
#
#     cv2.imshow("Canvas", canvas)
#     cv2.waitKey(0)
