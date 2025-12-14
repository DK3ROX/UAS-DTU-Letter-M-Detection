import cv2 as cv
import numpy as np
import math

#Function to learn what the Letter looks like by learning its contour
def template(filepath):

    img = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

    display_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    #blurring
    blurred = cv.GaussianBlur(img, (5,5), 0)
    _ , thresh = cv.threshold(blurred, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    contours, _  = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    
    #to find the largest contour
    if contours:
        contours = sorted(contours, key = cv.contourArea, reverse= True)
        largest_contour =  contours[0]

        cv.drawContours(display_img, [largest_contour], -1, (0, 255, 0), 2)
        window_title = f"Template Contour: {filepath.split('/')[-1]}"
        cv.imshow(window_title, display_img)
        return largest_contour

template_files = {
    'M' : ".venv/Dept. Task/M-Spartan bold.png"
}    

template_contours = {}

for label, filepath in template_files.items():
    template_contour = template(filepath)
    if template_contour is not None:
        template_contours[label] = template_contour

def detect():
    cap = cv.VideoCapture(0)

    Tolerance = 0.3
    print(f"Templates: {list(template_contours.keys())}")
    print(f"Tolerance: {Tolerance}")

    while True:
        ret, frame = cap.read() 
        frame = cv.flip(frame, 1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5,5), 0)
        _ , threshold = cv.threshold(blurred, 127, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        contours , _ = cv.findContours(threshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv.contourArea(cnt)
            
            if not (1200 < area < 20000):
                continue
            
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = float(w) / h

            epsilon = 0.02 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)
            corners = len(approx)

            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = (cx, cy)

            #Heading
            dist = []
            for p in cnt:
                X, Y = p[0]
                d = np.hypot(X - cx, Y - cy)
                dist.append((d, (X, Y)))
                dist.sort(key=lambda X: X[0], reverse=True)


            farthest_point1 = dist[0][1]
            for d, p in dist[1:]:
                if np.hypot(p[0] - farthest_point1[0], p[1] - farthest_point1[1]) > 30:
                    farthest_point2 = p
                    break
            else:
                farthest_point2 = dist[1][1]

            middle_point = (int((farthest_point1[0] + farthest_point2[0]) / 2), int((farthest_point1[1] + farthest_point2[1]) / 2))
            dx = middle_point[0] - cx  
            dy = cy - middle_point[1]
            angle = math.degrees(math.atan2(-dx, -dy))
            if angle < 0:
                angle += 360

            corners_match = (corners == 12) and (aspect_ratio > 0.85)

            best_score = float('inf')
            
            if not template_contours:
                continue

            for label, template_cnt in template_contours.items():
                score = cv.matchShapes(cnt, template_cnt, cv.CONTOURS_MATCH_I3, 0.0)

                if score < best_score:
                    best_score = score
            
            shape_match = (best_score < 0.4)
            
            if corners_match and shape_match:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv.circle(frame, center, 5, (0, 0, 255), -1) 
                cv.circle(frame, farthest_point1, 5, (0, 0, 255), -1) 
                cv.circle(frame, farthest_point2, 5, (0, 0, 255), -1)
                cv.circle(frame, middle_point, 5, (255, 0, 0), -1) 

                text = f"M(Angle : {angle:.1f} degrees)" 
                cv.putText(frame, text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv.imshow("Thresholded View", threshold)
        cv.imshow("Live Shape", frame)

        if cv.waitKey(1) == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()

detect()
