
import Numeric, Image
    #""" Transforme une image PIL en objet numpy.array et vice versa"""


def image2array(im):
    #""" image vers array numpy"""
    if im.mode not in ("L", "F"):
        raise ValueError, "can only convert single-layer images"
    if im.mode == "L":
        a = Numeric.fromstring(im.tostring(), Numeric.UnsignedInt8)
    else:
        a = Numeric.fromstring(im.tostring(), Numeric.Float32)
    a.shape = im.size[1], im.size[0]
    return a

def array2image(a):
    #""" array numpy vers image"""
    if a.typecode() == Numeric.UnsignedInt8:
        mode = "L"
    elif a.typecode() == Numeric.Float32:
        mode = "F"
    else:
        raise ValueError, "unsupported image mode"
    return Image.fromstring(mode, (a.shape[1], a.shape[0]), a.tostring())
