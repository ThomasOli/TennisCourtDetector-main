import cv2
import numpy as np


class PickleballCourtReference:
    """
    Pickleball court reference model (scaled to 20 pixels per foot)
    """
    def __init__(self):
        # Defining court dimensions in pixels (scaled)
        self.court_length = 880  # 44 ft scaled to 880 pixels
        self.court_width = 400   # 20 ft scaled to 400 pixels
        self.non_volley_zone = 140  # 7 ft scaled to 140 pixels
        self.service_area_length = 300  # 15 ft scaled to 300 pixels
        self.service_area_width = 200  # 10 ft scaled to 200 pixels (each service area)

        self.line_width = 1  # Line thickness

        self.top_bottom_border = 300  # Top and bottom border width
        self.right_left_border = 180  # Right and left border width

        # Adjusted court total dimensions
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_length + self.top_bottom_border * 2

        # Top baseline coordinates (shifted for border)
        self.baseline_top = ((self.right_left_border, self.top_bottom_border),
                             (self.right_left_border + self.court_width, self.top_bottom_border))

        # Bottom baseline coordinates (shifted for border)
        self.baseline_bottom = ((self.right_left_border, self.top_bottom_border + self.court_length),
                                (self.right_left_border + self.court_width, self.top_bottom_border + self.court_length))

        # Net (horizontal line at the middle of the court, shifted for border)
        self.net = ((self.right_left_border, self.top_bottom_border + self.court_length // 2),
                    (self.right_left_border + self.court_width, self.top_bottom_border + self.court_length // 2))

        # Sidelines (vertical boundaries, shifted for border)
        self.left_sideline = ((self.right_left_border, self.top_bottom_border),
                              (self.right_left_border, self.top_bottom_border + self.court_length))

        self.right_sideline = ((self.right_left_border + self.court_width, self.top_bottom_border),
                               (self.right_left_border + self.court_width, self.top_bottom_border + self.court_length))

        # Non-Volley Line (Kitchen Line) on both sides of the net (shifted for border)
        self.non_volley_line_top = ((self.right_left_border, self.top_bottom_border + self.court_length // 2 - self.non_volley_zone),
                                    (self.right_left_border + self.court_width, self.top_bottom_border + self.court_length // 2 - self.non_volley_zone))

        self.non_volley_line_bottom = ((self.right_left_border, self.top_bottom_border + self.court_length // 2 + self.non_volley_zone),
                                       (self.right_left_border + self.court_width, self.top_bottom_border + self.court_length // 2 + self.non_volley_zone))

        self.top_middle_line = ((self.right_left_border + self.court_width // 2, self.top_bottom_border),
                           (self.right_left_border + self.court_width // 2, self.top_bottom_border + self.service_area_length))
        
        self.bottom_middle_line = ((self.right_left_border + self.court_width // 2, self.top_bottom_border + self.court_length // 2 + self.non_volley_zone), 
                                   (self.right_left_border + self.court_width // 2, self.top_bottom_border + self.court_length))

    def build_court_reference(self):
        """
        Create pickleball court reference image using the line positions
        """
        # Create an empty court image (with borders)
        court = np.zeros((self.court_total_height, self.court_total_width), dtype=np.uint8)

        # Draw the baselines, net, sidelines, non-volley lines, and centerline
        cv2.line(court, *self.baseline_top, 1, self.line_width)
        cv2.line(court, *self.baseline_bottom, 1, self.line_width)
        cv2.line(court, *self.net, 1, self.line_width)
        cv2.line(court, *self.left_sideline, 1, self.line_width)
        cv2.line(court, *self.right_sideline, 1, self.line_width)
        cv2.line(court, *self.non_volley_line_top, 1, self.line_width)
        cv2.line(court, *self.non_volley_line_bottom, 1, self.line_width)
        cv2.line(court, *self.top_middle_line, 1, self.line_width)
        cv2.line(court, *self.bottom_middle_line, 1, self.line_width)

        # Dilate the lines to make them thicker
        court = cv2.dilate(court, np.ones((3, 3), dtype=np.uint8))

        return court

    def get_important_lines(self):
        """
        Returns all lines of the pickleball court
        """
        lines = [*self.baseline_top, *self.baseline_bottom, *self.net,
                 *self.left_sideline, *self.right_sideline,
                 *self.non_volley_line_top, *self.non_volley_line_bottom,
                 *self.top_middle_line, *self.bottom_middle_line]
        return lines


if __name__ == '__main__':
    pickleball_court = PickleballCourtReference()
    court_image = pickleball_court.build_court_reference()

    # Display the court
    cv2.imshow('Pickleball Court', court_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
