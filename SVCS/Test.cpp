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

Mat returnLBPHistogram2(Mat image) {
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
	normalize(imLBPHistogram, imLBPHistogram, 1, 0,NORM_MINMAX);
	imLBPHistogram.convertTo(imLBPHistogram, CV_32FC1);
	return imLBPHistogram;
}

int mainTe(int, char**)
{

	Ptr<SVM> svm = SVM::create();
	svm->load("inventry_model.yml");
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("edges",1);
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        imshow("img", frame);
        waitKey(0);
        Mat testMat = returnLBPHistogram2(frame).t();
        //vconcat(testMat, returnLBPHistogram2(frame).t(), testMat);

        cout<<"DIMS "<<testMat.rows<<" "<<testMat.cols<<endl<<flush;

        Mat testResponse1;
        svm->predict(testMat, testResponse1);
        std::vector<float>testResponse(testResponse1.begin<float>(), testResponse1.end<float>());

        cout << "The class is " << testResponse[0] << endl << flush;
        imshow("img", frame);
        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
