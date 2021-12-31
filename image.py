import cv2

img = cv2.imread(r"D:\AI car and pedestrian detection\Mini Project Lab\Mini Project Lab\car_img.jpg")

#Our pre-trained car and pedestrian classifier
car_tracker_file = 'car_detection.xml'
#pedestrian_tracker_file = 'haar_cascade.xml'

#create car and pedestrian classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
#pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

#conversion of black and white
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
#car_tracker = cv2.CascadeClassifier(classifier_file)

 #detection of cars
cars = car_tracker.detectMultiScale(black_n_white)
print(cars)

# #Draw Rectangles around the cars
car2 = cars[1]
(x, y, w, h) = car2
for(x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)



car2=cars[1]
# # car3=cars[2]
# # car4=cars[3]
(a,b,c,d) = car2
 ##(p,q,r,s)=car3
 ##(e,f,g,m)=car4


cv2.rectangle(img, (a, b), (a+c, b+d), (0, 0, 255), 1)
# # cv2.rectangle(img, (p, q), (p+r, q+s), (0, 0, 255), 1)
# # cv2.rectangle(img, (e, f), (e+g, f+m), (0, 0, 255), 1)

print(car2)



# #display thhe image with the faces spotted
cv2.imshow('Car Detector Image', img)

# #dont autoclose image
cv2.waitKey()
