Very basic AI image recognition to track how many minutes per day spent standing at my desk, in an effort to increase time standing.

This uses the MobileNetV2 as a base model with only one layer on top of that which is trained from the files in the images/sitting and images/standing directories.
To make your own, simply replace these images with yourself sitting or standing. Be sure there's the same amount of images in each folder.

Images where the AI was less sure of its decision are put in the to_sort folder. Manually classify and save these to the appropriate image directories, improving the AI's accuracy.

While running, it captures an image every minute from a webcam. It counts how many minutes were spent standing and puts them into a csv file.

Processing and displaying the data via this file is yet to come.
