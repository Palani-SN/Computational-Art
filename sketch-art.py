import os
import json
import pandas as pd
import numpy as np
import cv2


class Computation_Art:

    def __init__(self, imgs: list, ccs: int=15, css: int=250):
        
        # Width, Height & Center of the resultant image
        self.height = 2*(ccs*css) + 2*(css)
        self.width = 2*(ccs*css) + 2*(css)
        self.center = (css) + (ccs*css)

        self.front = np.zeros((self.height,self.width,3), np.uint8)
        self.back = np.zeros((self.height,self.width,3), np.uint8)

        sorted = self.parse_and_transform(imgs, ccs, css)
        # sorted.to_excel('out.xlsx')

        os.makedirs('Artifacts', exist_ok=True)
        # spirals = {}
        for idx, row in sorted.iterrows():
            # Calculate archimedian spiral
            spiral = self.synth_seq(row['radius'], row['angle'], ccs, css)
            # process the image as per need
            nm, img = self.refine_img(row)
            cv2.imwrite(f"Artifacts/colored-{nm}.jpg", img)
            for cs in range(ccs):
                # Template initialized
                temp = np.zeros((self.height,self.width,3), np.uint8)
                # radius of the circluar section
                radius = int((cs + 1) * css) 
                # Circluar mask generation
                if cs == 0:
                    circle_mask = cv2.circle(temp, (self.center, self.center), radius, (255, 255, 255), css)
                else:
                    circle_mask = cv2.circle(temp, (self.center, self.center), radius, (255, 255, 255), css)
                # Fetching image data for circular mask
                circle_result = np.where(
                    circle_mask == 255, 
                    img, 
                    temp
                )
                # Rotating the circular result with image data
                rotation_matrix = cv2.getRotationMatrix2D((self.center, self.center), spiral[cs], 1.0)
                rotated = cv2.warpAffine(circle_result, rotation_matrix, (self.width, self.height))
                self.front = cv2.bitwise_or(self.front, rotated)
                # cv2.imwrite(f"Artifacts/{nm}-{cs}.jpg", rotated)

        cv2.imwrite(f"Artifacts/front.jpg", self.front)
            # cv2.imwrite(f"Artifacts/colored-{nm}.jpg", img)
            # spirals[row['name']] = spiral

        # print(spirals)

    def refine_img(self, row):

        path = Path(row['img'])

        # read an image
        inp = cv2.imread(path.absolute(), cv2.IMREAD_COLOR_BGR)
        
        # Size normalization calculation
        height, width, channels = inp.shape
        if height >= width:
            mult = (height / self.height) +0.1
        else:
            mult = (width / self.width) +0.1    
        out_width = int(width/mult)
        out_height = int(height/mult)

        # getting resized input image
        input = cv2.resize(inp, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

        # Applying auto-canny edge filter on the image
        v = np.median(input)
        lower = int(max(0, (1.0 - row['canny']['sigma']) * v))
        upper = int(min(255, (1.0 + row['canny']['sigma']) * v))
        edged = cv2.Canny(input, lower, upper)

        # Reading dilate mask file & resizing to the output size
        mask = cv2.imread(Path(row['dilate']['mask']).absolute(), cv2.COLOR_GRAY2BGR)
        dilate_mask = cv2.resize(mask, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

        # Dilation filter applied
        dilated = np.where(
            dilate_mask == 255, 
            cv2.dilate(edged, tuple(row['dilate']['kernel']), iterations=row['dilate']['iter']), 
            edged)

        # Reading erode mask file & resizing to the output size
        inp = cv2.imread(Path(row['erode']['mask']).absolute(), cv2.COLOR_GRAY2BGR)
        erode_mask = cv2.resize(inp, (out_width, out_height), interpolation=cv2.INTER_LINEAR)

        # Erosion filter applied
        eroded = np.where(
            erode_mask == 255, 
            cv2.erode(edged, tuple(row['dilate']['kernel']), iterations=row['dilate']['iter']), 
            edged
        )
        
        # Merged the dilated & eroded images
        merged = cv2.bitwise_or(dilated, eroded)

        # Colored the images
        colored = np.zeros_like(input)
        colored[merged != 0] = row['clr']
        squared = self.make_square(colored)
        return (path.parent.name, cv2.resize(squared, (self.width, self.height), interpolation=cv2.INTER_LINEAR))

    def make_square(self, image, color=(0, 0, 0)):
        """ Converts a rectangular image to a square by adding borders. """
        h, w = image.shape[:2]
        size = max(h, w) 

        # Calculate padding (equally on both sides)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left

        # Add padding with given color
        squared_image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=color)
        return squared_image

    def synth_seq(self, radius, angle, ccs, css):

        spiral = []
        for c_rad in range(ccs):
            spiral.append((angle + ((c_rad*css) - radius) / 10) % 360)

        return spiral

    def parse_and_transform(self, imgs, ccs, css):

        df = pd.DataFrame(imgs)
        df[['radius', 'angle']] = df.apply(self.cartesian_to_polar, axis=1)
        return df.sort_values(by='radius')

    def cartesian_to_polar(self, row):
        x, y = tuple(row['loc'])
        radius = np.sqrt(x**2 + y**2)
        angle = np.degrees(np.arctan2(y, x))  # Convert radians to degrees
        return pd.Series([radius, angle], index=['radius', 'angle'])

if __name__ == '__main__':

    import yaml
    from pathlib import Path

    yml_dict = yaml.safe_load(Path("World-Wonders/art-of-wonders.yml").read_text())

    Computation_Art(
        imgs=yml_dict['Images']
    )