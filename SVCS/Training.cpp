/*
 * Training.cpp
 *
 *  Created on: 1 Feb, 2018
 *      Author: sriram
 */

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::ml;

Mat returnLBPHistogram3(Mat image) {
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

int maine(void) {

	Mat images;
	vector<int> labels;
	Mat img;
	string fileLine, classLine;
	ifstream fileName("dataSet.csv");
	ifstream fileClass("class.csv");
	if (fileName.is_open() && fileClass.is_open()) {
		while (getline(fileName, fileLine) && getline(fileClass, classLine)) {
			img = imread(fileLine, IMREAD_GRAYSCALE);
			if (images.empty())
				images = returnLBPHistogram3(img).t();
			else
				vconcat(images, returnLBPHistogram3(img).t(), images);

			stringstream cname(classLine);
			int cn = 0;
			cname >> cn;

			labels.push_back(cn);
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

	//vector<int> testResponse;
	Mat testResponse1;
	//cv::Mat testResponse(testResponse1.size(), 1, CV_32FC1);

	svm->predict(images, testResponse1);
	std::vector<float> testResponse(testResponse1.begin<float>(),
			testResponse1.end<float>());

	int accuracy = 0;
	cout << "mat " << testResponse1 << flush << endl;
	for (int i = 0; i < labels.size(); i++) {
		if (labels[i] == testResponse[i])
			accuracy += 1;
	}
	cout << "accuracy " << accuracy << flush << endl;

	/*
	 0 = olive
	 1 = brut
	 2 = snapple
	 3 = deo
	 4 = glue
	 5 = vaseline
	 6 = cookie jar
	 7 = tulsi
	 8 = pasta
	 9 = rubik
	 */

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
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened()) {  // check if we succeeded
		cout << "sdd" << endl << flush;
		return -1;
	}
	cout << "testing code " << flush << endl;

	//apture background
	Mat background;
	Mat MeanMat;
	Mat SDMat;
	vector<Mat> BkVector;
	for (int i = 0; i < 10; i++) {
		cap >> background;
		cvtColor(background, background, COLOR_BGR2GRAY);
		background.convertTo(background, CV_32FC1);
		background /= 255;
		BkVector.push_back(background);
		if (!MeanMat.empty())
			add(MeanMat,background , MeanMat);
		else
			background.copyTo(MeanMat);
	}
	MeanMat /= 10;

	for (int i = 0; i < 10; i++) {
		Mat diff;
		subtract(BkVector[i],MeanMat,diff);
		multiply(diff,diff,diff);

		if (!SDMat.empty())
			add(diff, SDMat, SDMat);
		else
			diff.copyTo(SDMat);
	}
	SDMat /= 10;
	sqrt(SDMat,SDMat);
	imshow("Mean", MeanMat);
	waitKey(0);
//	cvtColor(background, background, CV_BGR2HSV);
//	vector<Mat> channelsback;
//	split(background, channelsback);

	int dilation_type;
	int dilation_elem = 0;
	int dilation_size = 10;
	if (dilation_elem == 0) {
		dilation_type = MORPH_RECT;
	} else if (dilation_elem == 1) {
		dilation_type = MORPH_CROSS;
	} else if (dilation_elem == 2) {
		dilation_type = MORPH_ELLIPSE;
	}

	Mat element = getStructuringElement(dilation_type,
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			Point(dilation_size, dilation_size));

	namedWindow("edges", 1);
	Mat finalImg;
	for (;;) {
		Mat frame, gimg;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, gimg, COLOR_BGR2GRAY);
		frame.copyTo(finalImg);
		cvtColor(frame, frame, CV_BGR2HSV);
		vector<Mat> channels;
		split(frame, channels);
		gimg.convertTo(gimg, CV_32FC1);
		//find if there is a change in the background
		Mat absdiff2;
		Mat absdiff1;
		absdiff(gimg/255,MeanMat , absdiff2);
		subtract(absdiff2,3*SDMat, absdiff2);

		threshold(absdiff2,absdiff1, 0, 255, THRESH_BINARY_INV);
		imshow("absdiff1", absdiff1);

		absdiff1.convertTo(absdiff1, CV_8U);
		/// Apply the dilation operation
		dilate(absdiff1, absdiff1, element);

		vector<vector<Point> > contours;
		vector<Point> contour;
		int selectedIndex = 0;
		int maxArea = 0;
		vector<Vec4i> hierarchy;
		findContours(absdiff1, contours, hierarchy, CV_RETR_EXTERNAL,
				CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		cout <<"sdsd"<<endl<<flush;

		if (contours.size() > 0) {
			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			for (int i = 0; i < contours.size(); i++) {
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				boundRect[i] = boundingRect(Mat(contours_poly[i]));
				int area = contourArea(contours[i], true);
				if (area > maxArea) {
					maxArea = area;
					contour = contours[i];
					selectedIndex = i;
				}
			}

			Mat drawing = Mat::zeros(absdiff1.size(), CV_8UC3);
			//for( int i = 0; i< contours.size(); i++ )
			//{
			Scalar color = Scalar(255, 0, 0);
			drawContours(drawing, contours_poly, selectedIndex, color, 1, 8,
					vector<Vec4i>(), 0, Point());
			rectangle(drawing, boundRect[selectedIndex].tl(),
					boundRect[selectedIndex].br(), color, 2, 8, 0);

			Mat selectedArea = gimg(boundRect[selectedIndex]);

			// }
			imshow("selectedArea", selectedArea);

			waitKey(1);

			Mat testMat = returnLBPHistogram3(selectedArea).t();
			//vconcat(testMat, returnLBPHistogram2(frame).t(), testMat);

			cout << "DIMS " << testMat.rows << " " << testMat.cols << endl
					<< flush;

			Mat testResponse1;
			svm->predict(testMat, testResponse1);
			std::vector<float> testResponse(testResponse1.begin<float>(),
					testResponse1.end<float>());
			imshow("box", drawing);
			cout << "The class is " << testResponse[0] << endl << flush;
			putText(finalImg, classesName[testResponse[0]], Point2f(10, 40),
					FONT_HERSHEY_PLAIN, 3, Scalar(0, 255, 0),2);
			imshow("Final Result", finalImg);

		}

		if (waitKey(30) >= 0)
			break;
	}
	return 0;
}
