**ADVERTISEMENT INSERTION**

_Mechanism for ads insertion based on OpenCV package._

To insert ad you need to have the following:
- The logo to insert. Preferable formats are .png, .jpg;
- The video file for ad insertion.

The mechanism's main properties as follows:
- Detect stable contours in video using image threshold;
- Insert ad into detected contours.

To run the mechanism do the following:
- Download the repository with all consisting files;
- Install or upgrade necessary packages from requirements.txt;
- Add logo and video file to the root folder;
- Open ad_insertion_executor.py, set film and logo name (line 74, 75);
- Set preferable minimum time period for appearing unique logo in video. By default each logo will appear not less than 1.5 seconds. If you want to change time period find configurations.yml file and set parameter contour_threshold to the desired value;
- Set preferable minimum contour area for logo insertion. By default minimum contour area is 3000 pixels. If you want to change contour area find configurations.yml file and set parameter min_area_threshold to the desired value;
- After all the preparations run ad_insertion_executor.py.
 