###############################################################################
### Useful function for visualization of 
### face, hand, body, holistic and object pose estimation
###############################################################################

import cv2
import numpy as np

# Define default camera intrinsic
img_width  = 640
img_height = 480
intrin_default = {
    'fx': img_width*0.9, # Approx 0.7w < f < w https://www.learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/
    'fy': img_width*0.9,
    'cx': img_width*0.5, # Approx center of image
    'cy': img_height*0.5,
    'width': img_width,
    'height': img_height,
}

class DisplayHand:
    def __init__(self, draw3d=False, draw_camera=False, intrin=None, max_num_hands=1, vis=None):
        self.max_num_hands = max_num_hands
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        # Define kinematic tree linking keypoint together to form skeleton
        self.ktree = [0,          # Wrist
                      0,1,2,3,    # Thumb
                      0,5,6,7,    # Index
                      0,9,10,11,  # Middle
                      0,13,14,15, # Ring
                      0,17,18,19] # Little

        # Define color for 21 keypoint
        self.color = [[0,0,0], # Wrist black
                      [255,0,0],[255,60,0],[255,120,0],[255,180,0], # Thumb
                      [0,255,0],[60,255,0],[120,255,0],[180,255,0], # Index
                      [0,255,0],[0,255,60],[0,255,120],[0,255,180], # Middle
                      [0,0,255],[0,60,255],[0,120,255],[0,180,255], # Ring
                      [0,0,255],[60,0,255],[120,0,255],[180,0,255]] # Little
        self.color = np.asarray(self.color)
        self.color_ = self.color / 255 # For Open3D RGB
        self.color[:,[0,2]] = self.color[:,[2,0]] # For OpenCV BGR
        self.color = self.color.tolist()


    def draw2d(self, img, param):
        img_height, img_width, _ = img.shape

        # Loop through different hands
        for p in param:
            if p['class'] is not None:
                # Label left or right hand
                x = int(p['keypt'][0,0]) - 30
                y = int(p['keypt'][0,1]) + 40
                # cv2.putText(img, '%s %.3f' % (p['class'], p['score']), (x, y),
                cv2.putText(img, '%s' % (p['class']), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Red

                # Loop through keypoint for each hand
                for i in range(21):
                    x = int(p['keypt'][i,0])
                    y = int(p['keypt'][i,1])
                    if x>0 and y>0 and x<img_width and y<img_height:
                        # Draw skeleton
                        start = p['keypt'][self.ktree[i],:]
                        x_ = int(start[0])
                        y_ = int(start[1])
                        if x_>0 and y_>0 and x_<img_width and y_<img_height:
                            cv2.line(img, (x_, y_), (x, y), self.color[i], 2)

                        # Draw keypoint
                        cv2.circle(img, (x, y), 5, self.color[i], -1)
                        # cv2.circle(img, (x, y), 3, self.color[i], -1)

                        # # Number keypoint
                        # cv2.putText(img, '%d' % (i), (x, y),
                        #     cv2.FONT_HERSHEY_SIMPLEX, 1, self.color[i])

                        # # Label visibility and presence
                        # cv2.putText(img, '%.1f, %.1f' % (p['visible'][i], p['presence'][i]),
                        #     (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color[i])

		        # Label gesture
                if p['gesture'] is not None:
                    size = cv2.getTextSize(p['gesture'].upper(),
                        # cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    x = int((img_width-size[0]) / 2)
                    cv2.putText(img, p['gesture'].upper(),
                        # (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                        (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                    # Label joint angle
                    self.draw_joint_angle(img, p)

            # Label fps
            if p['fps']>0:
                cv2.putText(img, 'FPS: %.1f' % (p['fps']),
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return img


    def draw2d_(self, img, param):
        # Different from draw2d
        # draw2d_ draw 2.5D with relative depth info
        # The closer the landmark is towards the camera
        # The lighter and larger the circle

        img_height, img_width, _ = img.shape

        # Loop through different hands
        for p in param:
            if p['class'] is not None:
                # Extract wrist pixel
                x = int(p['keypt'][0,0]) - 30
                y = int(p['keypt'][0,1]) + 40
                # Label left or right hand
                cv2.putText(img, '%s %.3f' % (p['class'], p['score']), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Red

                min_depth = min(p['joint'][:,2])
                max_depth = max(p['joint'][:,2])

                # Loop through keypt and joint for each hand
                for i in range(21):
                    x = int(p['keypt'][i,0])
                    y = int(p['keypt'][i,1])
                    if x>0 and y>0 and x<img_width and y<img_height:
                        # Convert depth to color nearer white, further black
                        depth = (max_depth-p['joint'][i,2]) / (max_depth-min_depth)
                        color = [int(255*depth), int(255*depth), int(255*depth)]
                        size = int(10*depth)+2
                        # size = 2

                        # Draw skeleton
                        start = p['keypt'][self.ktree[i],:]
                        x_ = int(start[0])
                        y_ = int(start[1])
                        if x_>0 and y_>0 and x_<img_width and y_<img_height:
                            cv2.line(img, (x_, y_), (x, y), color, 2)

                        # Draw keypoint
                        cv2.circle(img, (x, y), size, color, size)

            # Label fps
            if p['fps']>0:
                cv2.putText(img, 'FPS: %.1f' % (p['fps']),
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return img


    def draw_joint_angle(self, img, p):
        # Create text
        text = None
        if p['gesture']=='Finger MCP Flexion':
            text = 'Index : %.1f \
                  \nMiddle: %.1f \
                  \nRing  : %.1f \
                  \nLittle : %.1f' % \
                (p['angle'][3], p['angle'][6], p['angle'][9], p['angle'][12])

        elif p['gesture']=='Finger PIP DIP Flexion':
            text = 'PIP: \
                  \nIndex : %.1f \
                  \nMiddle: %.1f \
                  \nRing  : %.1f \
                  \nLittle : %.1f \
                  \nDIP: \
                  \nIndex : %.1f \
                  \nMiddle: %.1f \
                  \nRing  : %.1f \
                  \nLittle : %.1f' % \
                (p['angle'][4], p['angle'][7], p['angle'][10], p['angle'][13],
                 p['angle'][5], p['angle'][8], p['angle'][11], p['angle'][14])

        elif p['gesture']=='Thumb MCP Flexion':
            text = 'Angle: %.1f' % p['angle'][1]

        elif p['gesture']=='Thumb IP Flexion':
            text = 'Angle: %.1f' % p['angle'][2]

        elif p['gesture']=='Thumb Radial Abduction':
            text = 'Angle: %.1f' % p['angle'][0]

        elif p['gesture']=='Thumb Palmar Abduction':
            text = 'Angle: %.1f' % p['angle'][0]

        elif p['gesture']=='Thumb Opposition':
            # Dist btw thumb and little fingertip
            dist = np.linalg.norm(p['joint'][4] - p['joint'][20])
            text = 'Dist: %.3f' % dist

        elif p['gesture']=='Forearm Neutral' or \
             p['gesture']=='Forearm Pronation' or \
             p['gesture']=='Forearm Supination' or \
             p['gesture']=='Wrist Flex/Extension' or \
             p['gesture']=='Wrist Radial/Ulnar Dev':
            text = 'Angle: %.1f' % p['angle'][0]

        if text is not None:
            x0 = 10 # Starting x coor for placing text
            y0 = 60 # Starting y coor for placing text
            dy = 25 # Change in text vertical spacing
            vert = len(text.split('\n'))
            # Draw black background
            cv2.rectangle(img, (x0, y0), (140, y0+vert*dy+10), (0,0,0), -1)
            # Draw text
            for i, line in enumerate(text.split('\n')):
                y = y0 + (i+1)*dy
                cv2.putText(img, line,
                    (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


    def draw_game_rps(self, img, param):
        img_height, img_width, _ = img.shape

        # Init result of 2 hands to none
        res = [None, None]

        # Loop through different hands
        for j, p in enumerate(param):
            # Only allow maximum of two hands
            if j>1:
                break

            if p['class'] is not None:
                # Loop through keypoint for each hand
                for i in range(21):
                    x = int(p['keypt'][i,0])
                    y = int(p['keypt'][i,1])
                    if x>0 and y>0 and x<img_width and y<img_height:
                        # Draw skeleton
                        start = p['keypt'][self.ktree[i],:]
                        x_ = int(start[0])
                        y_ = int(start[1])
                        if x_>0 and y_>0 and x_<img_width and y_<img_height:
                            cv2.line(img, (x_, y_), (x, y), self.color[i], 2)

                        # Draw keypoint
                        cv2.circle(img, (x, y), 5, self.color[i], -1)

                # Label gesture
                text = None
                if p['gesture']=='fist':
                    text = 'rock'
                elif p['gesture']=='five':
                    text = 'paper'
                elif (p['gesture']=='three') or (p['gesture']=='yeah'):
                    text = 'scissor'
                res[j] = text

                # Label result
                if text is not None:
                    x = int(p['keypt'][0,0]) - 30
                    y = int(p['keypt'][0,1]) + 40
                    cv2.putText(img, '%s' % (text.upper()), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) # Red

        # Determine winner
        text = None
        winner = None
        if res[0]=='rock':
            if res[1]=='rock'     : text = 'Tie'
            elif res[1]=='paper'  : text = 'Paper wins'  ; winner = 1
            elif res[1]=='scissor': text = 'Rock wins'   ; winner = 0
        elif res[0]=='paper':
            if res[1]=='rock'     : text = 'Paper wins'  ; winner = 0
            elif res[1]=='paper'  : text = 'Tie'
            elif res[1]=='scissor': text = 'Scissor wins'; winner = 1
        elif res[0]=='scissor':
            if res[1]=='rock'     : text = 'Rock wins'   ; winner = 1
            elif res[1]=='paper'  : text = 'Scissor wins'; winner = 0
            elif res[1]=='scissor': text = 'Tie'

        # Label gesture
        if text is not None:
            size = cv2.getTextSize(text.upper(),
                # cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x = int((img_width-size[0]) / 2)
            cv2.putText(img, text.upper(),
                # (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Draw winner text
        if winner is not None:
            x = int(param[winner]['keypt'][0,0]) - 30
            y = int(param[winner]['keypt'][0,1]) + 80
            cv2.putText(img, 'WINNER', (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2) # Yellow

        return img


FACE_CONNECTIONS = frozenset([
    # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    # Left eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    # Right eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    # Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)
])  

# Adapted from https://github.com/google/mediapipe/blob/350fbb2100ad531bc110b93aaea23d96af5a5064/mediapipe/python/solutions/objectron.py
BOX_CONNECTIONS = frozenset([
    (1, 2),
    (1, 3),
    (1, 5),
    (2, 4),
    (2, 6),
    (3, 4),
    (3, 7),
    (4, 8),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 8),
])
