/*
 * perspective_correction.cpp
 *
 *  Created on: 24 Jan, 2018
 *      Author: sriram
 */

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
#include <opencv2/ml.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::ml;
int top, bottom, left1, right1;
int ax, ay, bx, by, cx, cy, dx, dy;
int posx = 26, posy = 26;
int pixelZoomVal = 50;
bool firstRect = true;
int closestEdge = -1;

Mat returnLBPHistogram(Mat image) {
//	LBP implementation
	image.convertTo(image, CV_16S);
	Mat T1 = (Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, 0);
	Mat T2 = (Mat_<double>(3, 3) << 0, 1, 0, 0, -1, 0, 0, 0, 0);
	Mat T3 = (Mat_<double>(3, 3) << 0, 0, 1, 0, -1, 0, 0, 0, 0);
	Mat T4 = (Mat_<double>(3, 3) << 0, 0, 0, 0, -1, 1, 0, 0, 0);
	Mat T5 = (Mat_<double>(3, 3) << 0, 0, 0, 0, -1, 0, 0, 0, 1);
	Mat T6 = (Mat_<double>(3, 3) << 0, 0, 0, 0, -1, 0, 0, 1, 0);
	Mat T7 = (Mat_<double>(3, 3) << 0, 0, 0, 0, -1, 0, 1, 0, 0);
	Mat T8 = (Mat_<double>(3, 3) << 0, 0, 0, 1, -1, 0, 0, 0, 0);
	Mat imT1, imT2, imT3, imT4, imT5, imT6, imT7, imT8;
	filter2D(image, imT1, -1, T1, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(image, imT2, -1, T2, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(image, imT3, -1, T3, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(image, imT4, -1, T4, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(image, imT5, -1, T5, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(image, imT6, -1, T6, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(image, imT7, -1, T7, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(image, imT8, -1, T8, Point(-1, -1), 0, BORDER_DEFAULT);

	threshold(imT1, imT1, -1, 1, CV_THRESH_BINARY);
	threshold(imT2, imT2, -1, 1, CV_THRESH_BINARY);
	threshold(imT3, imT3, -1, 1, CV_THRESH_BINARY);
	threshold(imT4, imT4, -1, 1, CV_THRESH_BINARY);
	threshold(imT5, imT5, -1, 1, CV_THRESH_BINARY);
	threshold(imT6, imT6, -1, 1, CV_THRESH_BINARY);
	threshold(imT7, imT7, -1, 1, CV_THRESH_BINARY);
	threshold(imT8, imT8, -1, 1, CV_THRESH_BINARY);
//	LBP image
	Mat imLBP = imT1 + (2 * imT2) + (4 * imT3) + (8 * imT4) + (16 * imT5)
			+ (32 * imT6) + (64 * imT7) + (128 * imT8);

	imLBP.convertTo(imLBP, CV_8U);
	Mat imLBPHistogram;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	int histSize = 256;
//	Compute histogram for the LBP image
	calcHist(&imLBP, 1, 0, Mat(), imLBPHistogram, 1, &histSize, &histRange,
			true, false);
	normalize(imLBPHistogram, imLBPHistogram, 1, 0, NORM_MINMAX);
	imLBPHistogram.convertTo(imLBPHistogram, CV_32FC1);
	return imLBPHistogram;
}

float findDistance3(int x, int y, int a, int b) {
	return sqrt(pow(x - a, 2) + pow(y - b, 2));
}

int findClosestEdge3(int x, int y) {
	float threshold = 10.0;
	float distance = findDistance3(x, y, ax, ay);
	if (distance < threshold)
		return 1;
	distance = findDistance3(x, y, bx, by);
	if (distance < threshold)
		return 2;
	distance = findDistance3(x, y, cx, cy);
	if (distance < threshold)
		return 3;
	distance = findDistance3(x, y, dx, dy);
	if (distance < threshold)
		return 4;
	return -1;
}

void CallBackFunc3(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN) {
		cout << "Left button of the mouse is clicked - position (" << x << ", "
				<< y << ")" << endl;
		if (firstRect) {
			ax = x;
			ay = y;
			dx = x;
			by = y;
		} else {
			//find the closest edge
			closestEdge = findClosestEdge3(x, y);

		}
	} else if (event == EVENT_LBUTTONUP) {
		cout << "Right button of the mouse is clicked - position (" << x << ", "
				<< y << ")" << endl;
		if (firstRect) {
			cx = x;
			cy = y;
			bx = x;
			dy = y;
			firstRect = false;
		} else {
			if (closestEdge != -1) {

				//update coordinates
				if (closestEdge == 1) {
					ax = x;
					ay = y;

				} else if (closestEdge == 2) {
					bx = x;
					by = y;

				} else if (closestEdge == 3) {
					cx = x;
					cy = y;

				} else if (closestEdge == 4) {
					dx = x;
					dy = y;
				}

			}
		}

	} else if (event == EVENT_MOUSEMOVE) {
		posx = x;
		posy = y;
	} else if (event == EVENT_MOUSEWHEEL) {
		if (getMouseWheelDelta(flags) > 0)
			pixelZoomVal += 1;
		else
			pixelZoomVal -= 1;
	}
}
int main(int, char**) {

	Mat images;
	vector<int> labels;
	Mat img;
	string fileLine, classLine, classResult = "";

	ifstream fileName("dataSet.csv");
	ifstream fileClass("class.csv");
	if (fileName.is_open() && fileClass.is_open()) {
		while (getline(fileName, fileLine) && getline(fileClass, classLine)) {
			img = imread(fileLine, IMREAD_GRAYSCALE);
			if (images.empty())
				images = returnLBPHistogram(img).t();
			else
				vconcat(images, returnLBPHistogram(img).t(), images);

			stringstream cname(classLine);
			int cn = 0;
			cname >> cn;

			labels.push_back(cn);
			imshow ("s",img);
			waitKey(100);

		}
	}
	cout << "DIMS " << images.rows << " " << images.cols << endl << flush;

	Mat Labels_Mat(labels, true);
	//Mat Labels_Mat(labels.size(), 1, CV_32SC1, labels);
	//cout<<"dimensions "<<Labels_Mat.rows<<Labels_Mat.cols<<flush<<end;

	fileName.close();
	fileClass.close();

	//Convert the mat to array

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setC(12.5);
	svm->setGamma(0.50625);
	Ptr<TrainData> td = TrainData::create(images, ROW_SAMPLE, Labels_Mat);
	svm->train(td);
	svm->save("inventry_model.yml");

	vector<string> classesName;

	classesName.push_back("olive");
	classesName.push_back("brut");
	classesName.push_back("snapple");
	classesName.push_back("deo");
	classesName.push_back("glue");
	classesName.push_back("vaseline");
	classesName.push_back("cookie jar");
	classesName.push_back("tulsi");
	classesName.push_back("pasta");
	classesName.push_back("rubik");

	vector<Point> pixelPoints;
	int imgIndex = 0;

	VideoCapture cap(1); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	cv::Mat cam(3, 3, cv::DataType<float>::type);
	cv::Mat dist(4, 1, cv::DataType<float>::type);
	cam.at<float>(0, 0) = 3.7306267303732216e+02;
	cam.at<float>(0, 1) = 0.0f;
	cam.at<float>(0, 2) = 2.8650833543615556e+02;

	cam.at<float>(1, 0) = 0.0f;
	cam.at<float>(1, 1) = 3.7520893368200296e+02;
	cam.at<float>(1, 2) = 2.6534218289422870e+02;

	cam.at<float>(2, 0) = 0.0f;
	cam.at<float>(2, 1) = 0.0f;
	cam.at<float>(2, 2) = 1.0f;

	dist.at<float>(0, 0) = -2.6477519176603855e-02;
	dist.at<float>(1, 0) = -9.7344321800763578e-02;
	dist.at<float>(2, 0) = 1.9663779112528337e-02;
	dist.at<float>(3, 0) = 0.0f;

	Mat frame;
	Mat newCamMat;
	cv::Mat map1, map2;
	fisheye::estimateNewCameraMatrixForUndistortRectify(cam, dist,
			cv::Size(640, 480), Matx33d::eye(), newCamMat, 1);
	fisheye::initUndistortRectifyMap(cam, dist, Matx33d::eye(), newCamMat,
			cv::Size(640, 480),
			CV_16SC2, map1, map2);

	namedWindow("frame", 1);
	namedWindow("zoomed", 2);
	for (;;) {
		cap >> frame; // get a new frame from camera
		flip(frame, frame, 1);

		Mat m_undistImg;
		Mat zoomed;
//		cv::remap(frame, m_undistImg, map1, map2, cv::INTER_LINEAR);
//		imshow("frame", frame);
		frame.copyTo(m_undistImg);
		resize(m_undistImg, m_undistImg,
				Size(m_undistImg.cols * 2, m_undistImg.rows * 2));
		int k = waitKey(1);
		if (k >= 176 && k <= 186) {

			int top = (ay < by ? ay : by);
			int left = (ax < dx ? ax : dx);
			int right = (bx > cx ? bx : cx);
			int bottom = (cy > dy ? cy : dy);
			Rect region_of_interest = Rect(left, top, right - left,
					bottom - top);
			Mat cropWindow = m_undistImg(region_of_interest);

			cvtColor(cropWindow,cropWindow,CV_BGR2GRAY);

			Mat testMat = returnLBPHistogram(cropWindow).t();
			Mat testResponse1;
			svm->predict(testMat, testResponse1);
			std::vector<float> testResponse(testResponse1.begin<float>(),
					testResponse1.end<float>());
			classResult = classesName[testResponse[0]];

			cout << "The class is " << testResponse[0] << endl << flush;

			imshow("cropped", cropWindow);

		}
		if (!firstRect) {
			//draw rect
			Point points[1][4];
			points[0][0] = Point(ax, ay);
			points[0][1] = Point(bx, by);
			points[0][2] = Point(cx, cy);
			points[0][3] = Point(dx, dy);

			int lineType = 1;
			const Point* ppt[1] = { points[0] };
			int npt[] = { 4 };
			polylines(m_undistImg, ppt, npt, 1, true, Scalar(255, 0, 255), 1,
					lineType);
			circle(m_undistImg, Point(ax, ay), 2, Scalar(0, 255, 0), 3, LINE_8);
			circle(m_undistImg, Point(bx, by), 2, Scalar(0, 255, 0), 3, LINE_8);
			circle(m_undistImg, Point(cx, cy), 2, Scalar(0, 255, 0), 3, LINE_8);
			circle(m_undistImg, Point(dx, dy), 2, Scalar(0, 255, 0), 3, LINE_8);
			pixelPoints.clear();
			pixelPoints.push_back(Point(ax, ay));
			pixelPoints.push_back(Point(bx, by));
			pixelPoints.push_back(Point(cx, cy));
			pixelPoints.push_back(Point(dx, dy));

			cout << pixelPoints << endl << flush;

		}

		//draw zoom window
		if (posx > pixelZoomVal / 2 && posy > pixelZoomVal / 2
				&& posx < 2 * 640 - pixelZoomVal / 2
				&& posy < 2 * 480 - pixelZoomVal / 2) {
			Rect region_of_interest = Rect(posx - pixelZoomVal / 2,
					posy - pixelZoomVal / 2, pixelZoomVal, pixelZoomVal);
			Mat cropWindow = m_undistImg(region_of_interest);
			resize(cropWindow, cropWindow, Size(50, 50), 0, 0, INTER_CUBIC);
			line(cropWindow, Point(15, 25), Point(35, 25), Scalar(255, 0, 0), 1,
					LINE_8);
			line(cropWindow, Point(25, 15), Point(25, 35), Scalar(255, 0, 0), 1,
					LINE_8);
			imshow("zoomed", cropWindow);
		}
		putText(m_undistImg, classResult, Point2f(10, 40),
							FONT_HERSHEY_PLAIN, 3, Scalar(255, 255, 0),3);
		imshow("undist image", m_undistImg);

		setMouseCallback("undist image", CallBackFunc3, NULL);
		if (waitKey(1) >= 0) {
			//break;
		}
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
