/*
 * main.cpp
 *
 *  Created on: 11 Jan, 2018
 *      Author: sriram
 */

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main2(int, char**) {
	VideoCapture cap(1); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	cv::Mat cam(3, 3, cv::DataType<float>::type);
	cam.at<float>(0, 0) = 398.01382890106737f;
	cam.at<float>(0, 1) = 0.0f;
	cam.at<float>(0, 2) = 298.92568865695881f;

	cam.at<float>(1, 0) = 0.0f;
	cam.at<float>(1, 1) = 396.84964168912825f;
	cam.at<float>(1, 2) = 208.29412915843463f;

	cam.at<float>(2, 0) = 0.0f;
	cam.at<float>(2, 1) = 0.0f;
	cam.at<float>(2, 2) = 1.0f;

	cv::Mat dist(4, 1, cv::DataType<float>::type);
	dist.at<float>(0, 0) = -0.079659463111512702f;
	dist.at<float>(1, 0) = -0.061415813834196108f;
	dist.at<float>(2, 0) = -0.00073342334235905048f;
	dist.at<float>(3, 0) = 0.0f;
	Mat frame;
	Mat newCamMat;
	cv::Mat map1, map2;
	fisheye::estimateNewCameraMatrixForUndistortRectify(cam,
			dist, cv::Size(640,480), Matx33d::eye(), newCamMat, 1);
	fisheye::initUndistortRectifyMap(cam, dist, Matx33d::eye(),
			newCamMat, cv::Size(640,480),
			CV_16SC2, map1, map2);


	namedWindow("frame", 1);
	int count = 0;
	for (;;) {
		cap >> frame; // get a new frame from camera
		flip(frame, frame, 1);

		Mat m_undistImg;

		cv::remap(frame, m_undistImg, map1, map2, cv::INTER_LINEAR);
		imshow("frame", frame);
		imshow("undist image", m_undistImg);
		if (waitKey(30) >= 0) {
        	stringstream s;
        	s<<"left/"<<count<<".jpg";
        	count++;
        	imwrite(s.str(),frame);
//			break;
		}
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
