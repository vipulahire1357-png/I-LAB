import cv2
from pathlib import Path


VIDEO_PATH = Path("data/input/original - Trim.mp4")

WINDOW_NAME = "Fan Zone Editor"

points = []


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        print(f"Point {len(points)}: ({x}, {y})")


def draw_points(frame):
    image = frame.copy()

    # Draw points
    for i, (x, y) in enumerate(points):

        cv2.circle(
            image,
            (x, y),
            6,
            (255, 0, 255),
            -1
        )

        cv2.putText(
            image,
            f"P{i + 1} ({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

    # Draw polygon
    if len(points) >= 2:

        for i in range(len(points) - 1):

            cv2.line(
                image,
                points[i],
                points[i + 1],
                (255, 0, 255),
                2
            )

    if len(points) == 4:

        cv2.line(
            image,
            points[3],
            points[0],
            (255, 0, 255),
            2
        )

        overlay = image.copy()

        polygon = __import__("numpy").array(
            points,
            dtype="int32"
        )

        cv2.fillPoly(
            overlay,
            [polygon],
            (255, 0, 255)
        )

        image = cv2.addWeighted(
            overlay,
            0.15,
            image,
            0.85,
            0
        )

    # Instructions
    cv2.putText(
        image,
        "Click 4 points | R = reset | ENTER = finish | ESC = exit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return image


def main():

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Could not open video.")
        return

    # Read first frame
    ret, frame = cap.read()

    cap.release()

    if not ret:
        print("Could not read frame.")
        return

    print()
    print("====================================")
    print(" FAN ZONE EDITOR")
    print("====================================")
    print("Click exactly 4 points.")
    print("Point order:")
    print("P1 -> P2 -> P3 -> P4")
    print()
    print("R     = Reset")
    print("ENTER = Finish")
    print("ESC   = Exit")
    print("====================================")
    print()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        1440,
        900
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    while True:

        display = draw_points(frame)

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        # ENTER
        if key in (13, 10):

            if len(points) == 4:

                print()
                print("========== COORDINATES ==========")

                print("[")

                for point in points:
                    print(f"    {point},")

                print("]")

                print("=================================")
                print()

                break

            else:

                print(
                    f"Need 4 points. Currently selected: {len(points)}"
                )

        # R = reset
        elif key in (ord("r"), ord("R")):

            points.clear()

            print("Points reset.")

        # ESC
        elif key == 27:

            print("Cancelled.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()