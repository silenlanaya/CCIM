import numpy as np
import cv2
import argparse
import os

def hsvcolor(hMin, hMax, sMin, sMax, vMin, vMax):
    """Returns the lower and upper HSV values for a given color."""
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    return lower, upper

yellowrgb = (0,255,255)
purplergb = (255,0,0)
redrgb = (0,0,255)
redrgb_rev = (255,255,0)
bluergb = (255,0,0)
greenrgb = (0,255, 0)
whitergb = (255,255, 255)

yel_lower, yel_upper = hsvcolor(26, 34, 43, 255, 50, 255)
ora_lower, ora_upper = hsvcolor(11, 25, 43, 255, 70, 255)
green_lower, green_upper = hsvcolor(35, 77, 43, 255, 70, 255)
qing_lower, qing_upper = hsvcolor(78, 99, 43, 255, 70, 255)
blue_lower, blue_upper = hsvcolor(100, 124, 43, 255, 100, 255)
white_lower, white_upper = hsvcolor(0, 180, 0, 30, 150, 255)

red_lower, red_upper = hsvcolor(0, 10, 70, 255, 100, 255)
red_lower2, red_upper2 = hsvcolor(171, 180, 70, 255, 100, 255)


if __name__ == "__main__":
    # Read images
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgdir", default="./label", help="directory of input images")
    parser.add_argument("--outdir", default="./results", help="directory of output images")

    args = parser.parse_args()

    files_names = os.listdir(args.imgdir)
    print(f"num of imgs {len(files_names)}")

    for file_name in files_names:
        img_path = os.path.join(args.imgdir, file_name)
        img = cv2.imread(img_path)

        # convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # get green mask
        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        # mask_green = cv2.erode(mask_green, None, iterations=2)
        # mask_green = cv2.dilate(mask_green, None, iterations=2)

        # get green images
        img_green = cv2.bitwise_and(img, img, mask=mask_green)
        # background use white
        img_green_nobg = img_green.copy()
        img_green_nobg[np.where((img_green_nobg==[0,0,0]).all(axis=2))] = whitergb

        # get red mask
        mask_red = cv2.inRange(hsv, red_lower, red_upper)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red, mask_red2)
        # mask_red = cv2.erode(mask_red, None, iterations=2)
        # mask_red = cv2.dilate(mask_red, None, iterations=2)

        # get red images
        img_red = cv2.bitwise_and(img, img, mask=mask_red)
        # background use white
        img_red_nobg = img_red.copy()
        img_red_nobg[np.where((img_red_nobg==[0,0,0]).all(axis=2))] = whitergb


        # save green and red images
        cv2.imwrite(os.path.join(args.outdir, "green", file_name), img_green_nobg)
        cv2.imwrite(os.path.join(args.outdir, "red", file_name), img_red_nobg)

        # merge red & green 
        img_merge = cv2.bitwise_or(img_green, img_red)
        # background use white
        img_merge[np.where((img_merge==[0,0,0]).all(axis=2))] = whitergb
        
        cv2.imwrite(os.path.join(args.outdir, "merge", file_name), img_merge)
