
from tkinter import *

import operator
from skimage.transform import resize
import numpy as np
import pyscreenshot as ImageGrab

from easyai.support.examples import MNIST, plt


class Draw(object):

    def __init__(self, master, model):
        self.model = model.k_model
        self.master = master

        self.res = ""
        self.pre = [None, None]
        self.bs = 8.5

        self.c = Canvas(self.master, bd=3, relief="ridge", width=300, height=282, bg="white")
        self.c.pack(side=LEFT)

        f1 = Frame(self.master, padx=5, pady=5)

        self.pr = Label(f1, text="Prediction: None", fg="blue", font=("", 20, "bold"))
        self.pr.pack(pady=20)

        self.preds = [Label(f1, text="{}: None".format(i), fg="black", font=("", 15)) for i in range(10)]
        for p in self.preds:
            p.pack(pady=10)

        Button(f1, font=("", 15), fg="white", bg="red", text="Clear Canvas", command=self.clear).pack(side=BOTTOM)

        f1.pack(side=RIGHT, fill=Y)
        self.c.bind("<Button-1>", self.putPoint)
        self.c.bind("<ButtonRelease-1>", self.getResult)
        self.c.bind("<B1-Motion>", self.paint)

    def getResult(self, e):
        x = self.master.winfo_rootx() + self.c.winfo_x()
        y = self.master.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        pred = self.predict(ImageGrab.grab().crop((x, y, x1, y1))).flatten() * 100

        self.res = str(np.argmax(pred))
        self.pr["text"] = "Prediction: " + self.res

        print(pred)
        for num in range(len(self.preds)):
            self.preds[num]["text"] = str(num)  + ": " + str(round(pred[num], 3)) + "%"

    def clear(self):
        self.c.delete("all")

    def putPoint(self, e):
        self.c.create_oval(e.x - self.bs, e.y - self.bs, e.x + self.bs, e.y + self.bs, outline="black", fill="black")
        self.pre = [e.x, e.y]

    def paint(self, e):
        self.c.create_line(self.pre[0], self.pre[1], e.x, e.y, width=self.bs * 2, fill="black", capstyle=ROUND,
                           smooth=TRUE)
        self.pre = [e.x, e.y]

    def predict(self, img):
        def crop(img, bounding):
            start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
            end = tuple(map(operator.add, start, bounding))
            slices = tuple(map(slice, start, end))
            return img[slices]

        image = crop(resize(np.invert(np.array(img)), (30, 30)), (28, 28))[:, :, 0]

        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.show()
        image = image.reshape(1, np.prod(image.shape))
        return self.model.predict(image)


    @staticmethod
    def run_root(root, title):
        root.title(title)
        root.resizable(0, 0)
        root.mainloop()


if __name__ == "__main__":
    model = MNIST.mlp("digits")

    root = Tk()
    Draw(root, model)
    Draw.run_root(root, title="Digit Classifier")