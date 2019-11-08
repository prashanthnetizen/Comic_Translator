from Functions import *

IMAGE_PATH = ''

if __name__ == '__main__':
    img = cv2.imread(IMAGE_PATH)
    copy_img = mark_text_bubbles(img)
    show_image("contour", copy_img)
