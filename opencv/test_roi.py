import cv2
from roi import create_roi


# Load the enhanced image
image = cv2.imread("enhanced.jpg")

if image is None:
    raise ValueError("Could not load enhanced.jpg")

# Resize to our standard size
image = cv2.resize(image, (512, 512))

display = image.copy()

points = []


def mouse_callback(event, x, y, flags, param):

    global display

    if event == cv2.EVENT_LBUTTONDOWN:

        # First click = A
        if len(points) == 0:
            points.append((x, y))

            cv2.circle(
                display,
                (x, y),
                6,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                display,
                "A",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            print("Point A:", (x, y))

        # Second click = B
        elif len(points) == 1:
            points.append((x, y))

            cv2.circle(
                display,
                (x, y),
                6,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                display,
                "B",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            print("Point B:", (x, y))

            # Create ROI
            roi, mask = create_roi(
                image,
                points[0],
                points[1],
                width=40
            )

            # Draw corridor on original image
            cv2.line(
                display,
                points[0],
                points[1],
                (0, 255, 255),
                40
            )

            # Redraw A and B on top
            cv2.circle(
                display,
                points[0],
                6,
                (0, 255, 0),
                -1
            )

            cv2.circle(
                display,
                points[1],
                6,
                (255, 0, 0),
                -1
            )

            # Save results
            cv2.imwrite("roi.jpg", roi)
            cv2.imwrite("roi_selection.jpg", display)

            print("ROI saved as roi.jpg")
            print("Visualization saved as roi_selection.jpg")


cv2.namedWindow("Select A and B")
cv2.setMouseCallback("Select A and B", mouse_callback)

print("Click once for A")
print("Click again for B")
print("Press ESC to exit")

while True:

    cv2.imshow("Select A and B", display)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

cv2.destroyAllWindows()