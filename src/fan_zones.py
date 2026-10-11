import cv2
import numpy as np


class FanZoneManager:

    # Camera resolution:
    # Width  = 1440
    # Height = 1080
    #
    # FAN1 = Left Front
    # FAN2 = Left Back
    # FAN3 = Right Front
    # FAN4 = Right Back

    ZONES = {

        # Left Front - 1st row
        "FAN1":[
            (915, 312),
            (589, 489),
            (209, 122),
            (484, 5),
        ],

        # Left Back - 2 rows
        "FAN2":[
            (937, 342),
            (1299, 205),
            (1064, 12),
            (573, 14),
        ],

        # Right Front - 1st row
        "FAN3": [
    
   
            (6, 1080),
            (1648, 1080),
            (1758, 738),
            (1102, 358),
        ],

        # Right Back - 2 rows
        "FAN4":[
            (1099, 312),
            (1448, 142),
            (1877, 344),
            (1773, 666),
        ]
    }

    def get_zone(self, foot_x, foot_y, frame_width, frame_height):

        point = (float(foot_x), float(foot_y))

        for zone_name, points in self.ZONES.items():

            polygon = np.array(
                points,
                dtype=np.int32
            )

            inside = cv2.pointPolygonTest(
                polygon,
                point,
                False
            )

            if inside >= 0:
                return zone_name

        return None
    def get_zone_from_bottom(
            self,
            bbox,
            frame_width,
            frame_height,
            threshold=0.50
        ):
            x1, y1, x2, y2 = map(int, bbox)

            # Bounding box ki height
            box_height = y2 - y1

            if box_height <= 0:
                return None

            # Bottom 30% start point
            bottom_start_y = int(
                y2 - (box_height * 0.30)
            )

            # Bottom 30% ka rectangle
            bottom_points = np.array([
                [x1, bottom_start_y],
                [x2, bottom_start_y],
                [x2, y2],
                [x1, y2]
            ], dtype=np.int32)

            # Bottom portion ka mask
            bottom_mask = np.zeros(
                (frame_height, frame_width),
                dtype=np.uint8
            )

            cv2.fillPoly(
                bottom_mask,
                [bottom_points],
                255
            )

            bottom_area = cv2.countNonZero(bottom_mask)

            if bottom_area == 0:
                return None

            # Har fan zone check karo
            for zone_name, points in self.ZONES.items():

                polygon = np.array(
                    points,
                    dtype=np.int32
                )

                zone_mask = np.zeros(
                    (frame_height, frame_width),
                    dtype=np.uint8
                )

                cv2.fillPoly(
                    zone_mask,
                    [polygon],
                    255
                )

                # Bottom 30% AND zone
                overlap = cv2.bitwise_and(
                    bottom_mask,
                    zone_mask
                )

                overlap_area = cv2.countNonZero(overlap)

                overlap_ratio = (
                    overlap_area / bottom_area
                )

                if overlap_ratio >= threshold:
                    return zone_name

            return None

    def draw_zones(self, frame):

        for zone_name, points in self.ZONES.items():

            polygon = np.array(
                points,
                dtype=np.int32
            )

            # Draw boundary
            cv2.polylines(
                frame,
                [polygon],
                True,
                (255, 0, 255),
                3
            )

            # Calculate label position
            center_x = int(
                sum(x for x, y in points) / len(points)
            )

            center_y = int(
                sum(y for x, y in points) / len(points)
            )

            cv2.putText(
                frame,
                zone_name,
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 255),
                3
            )

        return frame