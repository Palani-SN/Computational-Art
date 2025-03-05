import json
from pathlib import Path
import numpy as np
import cv2
import os
import yaml
from types import SimpleNamespace

yml_dict = yaml.safe_load(Path("World-Wonders/art-of-wonders.yml").read_text())

class Canvas:

    def __init__(self, ccs:int, css:int):
        
        self.ccs = ccs
        self.css = css
        self.height = 2*(ccs*css) + 2*(css)
        self.width = 2*(ccs*css) + 2*(css)
        self.center = (self.css) + (self.ccs*self.css)
        self.front = np.zeros((self.height,self.width,3), np.uint8)
        self.back = np.zeros((self.height,self.width,3), np.uint8)

    def write(self):

        cv2.imwrite('front', self.front)
        cv2.imwrite('back', self.back)

    def make_square(self, image, color=(0, 0, 0)):
        """ Converts a rectangular image to a square by adding borders. """
        h, w = image.shape[:2]
        size = max(h, w)  # Determine the square size

        # Calculate padding (equally on both sides)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left

        # Add padding with given color
        squared_image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=color)
        return squared_image

    def img(self, obj):

        path = Path(obj.img)
        inp = cv2.imread(path.absolute(), cv2.IMREAD_COLOR_BGR)
        height, width, channels = inp.shape
        if height >= width:
            mult = (height / self.height) +0.1
        else:
            mult = (width / self.width) +0.1
            
        new_width = int(width/mult)
        new_height = int(height/mult)
        return cv2.resize(inp, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def auto_canny(self, obj, img):
        v = np.median(img)
        lower = int(max(0, (1.0 - obj.canny.sigma) * v))
        upper = int(min(255, (1.0 + obj.canny.sigma) * v))
        edged = cv2.Canny(img, lower, upper)
        return edged

    def canny(self, obj, img):

        return cv2.Canny(img, obj.canny.low, obj.canny.high)
    
    def dilate(self, obj, img):

        path = Path(obj.dilate.mask)
        inp = cv2.imread(path.absolute(), cv2.COLOR_GRAY2BGR)
        height, width = inp.shape
        if height >= width:
            mult = (height / self.height) +0.1
        else:
            mult = (width / self.width) +0.1
            
        new_width = int(width/mult)
        new_height = int(height/mult)
        mask = cv2.resize(inp, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        dilated = cv2.dilate(img, tuple(obj.dilate.kernel), iterations=obj.dilate.iter)
        masked = np.where(mask == 255, dilated, img)
        return masked
    
    def erode(self, obj, img):

        path = Path(obj.erode.mask)
        inp = cv2.imread(path.absolute(), cv2.COLOR_GRAY2BGR)
        height, width = inp.shape
        if height >= width:
            mult = (height / self.height) +0.1
        else:
            mult = (width / self.width) +0.1
            
        new_width = int(width/mult)
        new_height = int(height/mult)
        mask = cv2.resize(inp, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        eroded = cv2.erode(img, tuple(obj.erode.kernel), iterations=obj.erode.iter)
        masked = np.where(mask == 255, eroded, img)
        return masked

    def diffuse(self, conf:dict):

        os.makedirs('Artifacts', exist_ok=True)

        for k, v in conf.items():

            # print(k)
            # print(json.dumps(v, indent=2))

            obj = json.loads(json.dumps(v, indent=2), object_hook=lambda d: SimpleNamespace(**d))
            path = Path(obj.img)
            img = self.img(obj)
            edge = self.auto_canny(obj, img)
            
            dilated = self.dilate(obj, edge)
            dilated_file = f"Artifacts/dilated-{path.parent.name}.jpg"
            # print(dilated_file)
            # cv2.imwrite(dilated_file, dilated)

            eroded = self.erode(obj, edge)
            eroded_file = f"Artifacts/eroded-{path.parent.name}.jpg"
            # print(eroded_file)
            # cv2.imwrite(eroded_file, eroded)

            merged = cv2.bitwise_or(dilated, eroded)
            merged_file = f"Artifacts/result-{path.parent.name}.jpg"
            # print(merged_file)
            # cv2.imwrite(merged_file, merged)

            colored = np.zeros_like(img)
            colored[merged != 0] = obj.clr
            result_file = f"Artifacts/colored-{path.parent.name}.jpg"
            squared = self.make_square(colored)
            cv2.imwrite(result_file, squared)

            # path = Path(v['img'])

            # inp = cv2.imread(path.absolute(), cv2.IMREAD_COLOR_BGR)
            # height, width, channels = inp.shape
            
            # if height >= width:
            #     mult = (height / self.height) +0.1
            # else:
            #     mult = (width / self.width) +0.1
                
            # new_width = int(width/mult)
            # new_height = int(height/mult)
            # overlay = cv2.resize(inp, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # y = int(self.center - (new_height/2))
            # x = int(self.center - (new_width/2))

            # edged = cv2.Canny(overlay, 250, 255)
            # dilated = cv2.dilate(edged, (10,10), iterations=5)
            # np.where(mask == 255, dilated, image)
            # bgr_ed = cv2.cvtColor( dilated, cv2.COLOR_GRAY2BGR)

            # cv2.imwrite(f"out/{path.name}", bgr_ed)
            # self.front[y:y+new_height, x:x+new_width] = bgr_ed

    def mask(self):

        black = (0, 0, 0)
        white = (255, 255, 255)
        center = (self.css) + (self.ccs*self.css)
        for cs in range(self.css):
            if cs % 2 == 0:
                color = black
            else:
                color = white
            radius = int((cs + 1) * self.css) 
            cv2.circle(self.front, (center, center), radius, color, self.css) 
            cv2.circle(self.back, (center, center), radius, color, self.css) 

if __name__ == '__main__':

    canvas = Canvas(ccs=15, css=250)
    images = yml_dict['Images']

    for img in images:
        canvas.diffuse(img)