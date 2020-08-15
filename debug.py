import csv

with open('image_feature.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    features = [x for x in reader]


