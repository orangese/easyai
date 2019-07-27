from tkinter import *
from PIL import ImageTk, Image, ImageDraw
import PIL

def draw(fileName):
    width = 200
    height = 200
    white = (255, 255, 255)
    def save():
        image1.save(fileName)
    def drawIm(event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        cv.create_oval(x1, y1, x2, y2, width=5, fill="black")
        draw.line(((x2,y2),(x1,y1)), fill="black", width=10)

    root = Tk()
    cv = Canvas(root, width=width, height=height, bg="white")
    cv.pack()
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)
    cv.bind("<B1-Motion>", drawIm)
    button=Button(text="save", command=save)
    button.pack()
    root.mainloop()
    
def getActivation(fileName):
    img = Image.open(fileName)
    img = img.resize((28, 28))
    img = np.take(np.asarray(img), [0], axis = 2).reshape(28, 28)
    return np.abs(img-255)
