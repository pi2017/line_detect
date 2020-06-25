import cv2 as cv
import numpy as np


def region_of_interest(image, vertices):
    # Tip: The Bitwise operations should be applied on input image of same dimensions
    mask = np.zeros_like(image)
    # calculates channels 1 for gray & 3 for BGR
    # channel_count = image.shape[2]
    # creates (255,255,255) for BGR & 255 for grayscale
    match_mask_color = 255  # (255,) * channel_count
    # the polygon is filled with white color
    cv.fillPoly(mask, vertices, match_mask_color)
    # using bitwise_and
    # 0(black) and anyvalue = (0)
    # 255(white) and anyvalue = anyvalue
    masked_image = cv.bitwise_and(image, mask)  # Hence the image is replaced in polygon
    cv.imshow('show', masked_image)
    return masked_image


def detect_lines(cropped_image, img):
    lines = cv.HoughLinesP(cropped_image, 6, np.pi / 180, 160, lines=np.array([]), minLineLength=5, maxLineGap=10)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    print(blank_image.shape)
    print(img.shape)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 7)
    output = cv.addWeighted(img, 0.5, blank_image, 1, 0)
    return output


def process(img2):
    w, h = img2.shape[1], img2.shape[0]
    region_of_interest_vertices = [(0, h), (w / 2, h / 1.35), (w / 1.3, h)]
    gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    canny = cv.Canny(gray, 255, 255)
    cropped_image = region_of_interest(canny, np.array([region_of_interest_vertices], np.int32))
    output = detect_lines(cropped_image, img2)
    cv.imshow('show', output)
    cv.waitKey(1)


# img = cv.imread('road.jpg')
# img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cap = cv.VideoCapture("video/testvideo2.mp4")
while (cap.isOpened()):
    ret, frame = cap.read()
    process(frame)
    if (cv.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
#plt.imshow(output)
# plt.show()
