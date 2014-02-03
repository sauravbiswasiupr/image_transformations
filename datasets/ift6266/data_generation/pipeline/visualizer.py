#!/usr/bin/python

import numpy
import Image
from image_tiling import tile_raster_images
import pylab
import time

class Visualizer():
    def __init__(self, num_columns=10, image_size=(32,32), to_dir=None, on_screen=False):
        self.list = []
        self.image_size = image_size
        self.num_columns = num_columns

        self.on_screen = on_screen
        self.to_dir = to_dir

        self.cur_grid_image = None

        self.cur_index = 0

    def visualize_stop_and_flush(self):
        self.make_grid_image()

        if self.on_screen:
            self.visualize()
        if self.to_dir:
            self.dump_to_disk()

        self.stop_and_wait()
        self.flush()

        self.cur_index += 1

    def make_grid_image(self):
        num_rows = len(self.list) / self.num_columns
        if len(self.list) % self.num_columns != 0:
            num_rows += 1
        grid_shape = (num_rows, self.num_columns)
        self.cur_grid_image = tile_raster_images(numpy.array(self.list), self.image_size, grid_shape, tile_spacing=(5,5), output_pixel_vals=False)

    def visualize(self):
        pylab.imshow(self.cur_grid_image)
        pylab.draw()

    def dump_to_disk(self):
        gi = Image.fromarray((self.cur_grid_image * 255).astype('uint8'), "L")
        gi.save(self.to_dir + "/grid_" + str(self.cur_index) + ".png")
        
    def stop_and_wait(self):
        # can't raw_input under gimp, so sleep)
        print "New image generated, sleeping 5 secs"
        time.sleep(5)

    def flush(self):
        self.list = []
    
    def get_parameters_names(self):
        return []

    def regenerate_parameters(self):
        return []

    def after_transform_callback(self, image):
        self.transform_image(image)

    def end_transform_callback(self, final_image):
        self.visualize_stop_and_flush()

    def transform_image(self, image):
        sz = self.image_size
        self.list.append(image.copy().reshape((sz[0] * sz[1])))

