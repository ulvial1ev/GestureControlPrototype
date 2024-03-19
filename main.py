import cv2
import mediapipe as mp
import time
import math

def display_time_with_border(img, time_text):
    shadow_offset = 2
    shadow_position = (int(time_pos[0]) + shadow_offset, int(time_pos[1]) + shadow_offset)
    cv2.putText(img, time_text, shadow_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

    # Add text
    cv2.putText(img, time_text, time_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2, cv2.LINE_AA)

# Function to draw fancy item
def draw_item(img, position):
    cv2.putText(img, "ITEM", (position[0] - 20, position[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
grab_mode = False  
grabbed_item = False 
prev_thumb_tip = (0, 0)  
screen_width, screen_height = cap.get(3), cap.get(4) 
font = cv2.FONT_HERSHEY_DUPLEX
welcometext_pos=(int(screen_width-200),80)
fps_pos = (10, 30)
time_pos = (int(screen_width - 200), 30)
grab_pos = (10, 80)
annot_pos = (10, 140)
item_position = (int(screen_width // 2), int(screen_height // 2))  # Initial position of ITEM

#distance thresholds for QUIT functionality and GRAB mode.
distance_thres_grab = 40
distance_thres_quit = 25


# Timeout duration between grab mode toggling
timeout_duration = 1.0  # in seconds
last_toggle_time = time.time()

while cap.isOpened():
    success, img = cap.read()

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get landmark coordinates
            landmarks = []
            for lm in handLms.landmark:
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

            # Check if the thumb and pointer finger tips are close
            thumb_tip = landmarks[4]
            pointer_tip = landmarks[8]
            pinky = landmarks[20]
            distance = math.sqrt((pointer_tip[0] - thumb_tip[0]) ** 2 + (pointer_tip[1] - thumb_tip[1]) ** 2)
            distance_quit = math.sqrt((pinky[0] - thumb_tip[0]) ** 2 + (pinky[1] - thumb_tip[1]) ** 2)

            # If the distance is less than a threshold, toggle grab mode
            current_time = time.time()
            if current_time - last_toggle_time > timeout_duration:
                if distance < distance_thres_grab: 
                    grab_mode = not grab_mode
                    last_toggle_time = current_time

            if current_time - last_toggle_time > timeout_duration:
                if distance_quit < distance_thres_quit: 
                    cap.release()
                    cv2.destroyAllWindows()

            # When GRAB mode is on, and ITEM is not grabbed, check if the hand is near.
            if grab_mode and not grabbed_item:
                center_position = ((pointer_tip[0] + thumb_tip[0]) // 2, (pointer_tip[1] + thumb_tip[1]) // 2)
                item_distance = math.sqrt((center_position[0] - screen_width // 2) ** 2 +
                                            (center_position[1] - screen_height // 2) ** 2)
                
                if item_distance < distance_thres_grab:
                    grabbed_item = True
                    prev_thumb_tip = thumb_tip

            # When grabbed, change position
            if grabbed_item:
                dx = thumb_tip[0] - prev_thumb_tip[0]
                dy = thumb_tip[1] - prev_thumb_tip[1]
                item_position = (item_position[0] + dx, item_position[1] + dy)
                prev_thumb_tip = thumb_tip

    # Reset
    if not grab_mode:
        grabbed_item = False




    # Display FPS
    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime
    cv2.putText(img, f"FPS: {fps}", fps_pos, font, 0.5, (128, 0, 128), 1)

    # Display current time
    current_time = time.strftime("%H:%M:%S")
    display_time_with_border(img, current_time)

    # Display GRAB mode status
    grab_mode_text = "GRAB MODE : ON" if grab_mode else "GRAB MODE : OFF"
    color = (0, 255, 0) if grab_mode else (0, 0, 255)
    cv2.putText(img, grab_mode_text, grab_pos, font, 0.7, color, 1)
    cv2.putText(img, "Touch pinky and thumb to quit..", annot_pos, font, 1, color, 1)

    draw_item(img, item_position)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
