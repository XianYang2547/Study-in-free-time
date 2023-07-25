from PIL import Image,ImageDraw
img=Image.open('images/01.jpg')
draw=ImageDraw.Draw(img)
# draw.rectangle((45,93,282,344),width=3)
# draw.rectangle((205,71,375,339),width=3)
x1=163-int(237/2)
y1=218-int(251/2)
x2=163+int(237/2)
y2=218+int(251/2)
draw.rectangle((x1,y1,x2,y2),width=3)
img.show()
#images/01.jpg 2 163 218 237 251 	 1 290 205 170 268