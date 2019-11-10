from config import *

# Calling the OCR Engine
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import pandas as pd
import re
from PIL import Image, ImageDraw, ImageFont
import textwrap
from googletrans import Translator


class ComicTranslator:
    def __init__(self, path):
        self.image = cv2.imread(path)
        self.contours = []
        self.image_data = np.zeros((1, 1))
        self.sortis = []
        self.actual_images = []
        # If you don't have tesseract executable in your PATH, include the following:
        pytesseract.pytesseract.tesseract_cmd = r'D:/Tesseract/tesseract.exe'

    def mark_text_bubbles(self):
        # Converting the image from BGR format to HSV
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # For white color
        max_white = np.array([0, 0, 255])
        min_white = np.array([0, 0, 230])
        final_mask = cv2.inRange(hsv_image, min_white, max_white)
        # Fetching the matching components in the original image
        result = cv2.bitwise_and(self.image, self.image, mask=final_mask)
        # Deriving Contours for the Image
        self.contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Copying the image and checking whether contours are matching.
        copy_img = self.image.copy()
        cv2.drawContours(copy_img, self.contours, -1, (0, 255, 0), 1)
        show_image("Marked", copy_img)

    def extract_text_bubbles(self):
        # Calculating Area for each Contour
        area = []
        sorted_contours = sorted(self.contours, key=cv2.contourArea, reverse=True)
        for c in sorted_contours:
            area.append(cv2.contourArea(c))

        # Getting the top Indices of contoured Images that matter
        top_indices = [i for i in range(len(area)) if area[i] > np.mean(area)]

        # Cropping out Images that matter
        self.image_data = np.zeros((len(top_indices), 4), dtype=int)
        self.actual_images = []
        for i in top_indices:
            x, y, w, h = cv2.boundingRect(sorted_contours[i])
            ROI = self.image[y:y + h, x:x + w]
            self.image_data[i] = np.array([x, y, w, h])
            self.actual_images.append(ROI)

        (x_total, y_total, _) = self.image.shape
        index = self.image_data[:, 0] / x_total + self.image_data[:, 1] / y_total
        self.image_data = np.column_stack((index, self.image_data))
        final_data = pd.DataFrame(self.image_data, columns=['i', 'x', 'y', 'w', 'h'])
        self.sortis = final_data.sort_values(by=['i']).index

        for i in self.sortis:
            cv2.imwrite('manga_test_images/%d.png' % i, self.actual_images[i])
            print(pytesseract.image_to_string(self.actual_images[i], timeout=4, lang='fra'))

        return self.actual_images, self.image_data, self.sortis

    def output_translated_image(self):
        sample = self.image.copy()
        self.image_data = self.image_data.astype(int)

        translator = Translator()

        for i in self.sortis:
            # Extracting Text from the Image
            some_text = pytesseract.image_to_string(self.actual_images[i], timeout=4, lang='fra')
            some_text = re.sub(r'\n\s*\n', ' ', some_text)
            [_, x1, y1, w1, h1] = self.image_data[i]
            some_text = textwrap.wrap(some_text, width=20)
            whole_text = ""
            for one in some_text:
                whole_text += one + "\n"
            balloon_box = Image.new('RGB', (w1, h1), color=(255, 255, 255))
            if len(whole_text) == 0:
                continue
            # Translating Text before writing
            whole_text = translator.translate(whole_text, src='fr', dest='en')

            # Writing Text onto the newly created Image
            d = ImageDraw.Draw(balloon_box)
            font = ImageFont.truetype("fonts/SF_Arch_Rival.ttf", size=20)
            d.text((10, 10), whole_text.text, fill=(0, 0, 0), align="left", font=font)

            # Converting Image to BGR format suited for Opencv and constructing Borders
            opencvImage = cv2.cvtColor(np.array(balloon_box), cv2.COLOR_RGB2BGR)
            opencvImage = cv2.copyMakeBorder(src=opencvImage, top=5, bottom=5, left=5, right=5,
                                             borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # Superimposing images on the Comic Image
            sample[y1:opencvImage.shape[0] + y1, x1:opencvImage.shape[1] + x1, :] = opencvImage

        show_image("final", sample)
        cv2.imwrite("output.jpg", sample)
