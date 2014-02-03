#!/usr/bin/python                                                                                 

import os
import ImageFont, ImageDraw, Image

dir1 =  "/data/lisa/data/ift6266h10/allfonts/"
#dir1 = "/Tmp/allfonts/"

img = Image.new("L", (132,132))
draw = ImageDraw.Draw(img)
L = [chr(ord('0')+x) for x in range(10)] + [chr(ord('A')+x) for x in range(26)] + [chr(ord('a')+x) for x in range(26)]

for f in os.listdir(dir1):
    try:
        font = ImageFont.truetype(dir1+f, 25)
        for l in L:
            draw.text((60,60), l, font=font, fill="white")
    except:
        print dir1+f
