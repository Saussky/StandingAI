Very basic AI image recognition to track how many minutes per day I spend standing at my desk, in an effort to increase time standing.

This uses the MobileNetV2 as a base model with only one layer on top of that which is trained from the folders under images/sitting and images/standing.

When running it captures an image every minute from a webcam. It counts how many minutes were spent standing and puts them into a csv file.

Processing and displaying the data via this file is yet to come.
