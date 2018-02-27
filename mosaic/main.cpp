/*
 * main.cpp
 *
 *  Created on: 24 Feb, 2018
 *      Author: sriram
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"

#include <iostream>
using namespace std;
using namespace cv;



int main(int argc, char* argv[]) {
	Mat pano;
	Stitcher::Mode mode = Stitcher::SCANS;
	VideoCapture video(1);
	Ptr<Stitcher> stitcher = Stitcher::create(mode, false);
	if(!video.isOpened()){
		cout << "Video port not opened !!"<<endl<<flush;
		return -1;
	}
	vector<Mat> imgs;
	while(true){
		Mat image;
		video >> image;
		imgs.push_back(image);

		if (imgs.size() > 2){
			resize(imgs[0],image,Size(imgs[0].cols/2,imgs[0].rows/2));
			imshow("1",image);
			resize(imgs[1],image,Size(imgs[0].cols/2,imgs[0].rows/2));
			imshow("2",image);
			resize(imgs[2],image,Size(imgs[0].cols/2,imgs[0].rows/2));
			imshow("3",image);
			waitKey(30);
			Stitcher::Status status = stitcher->stitch(imgs, pano);
			if (status == Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL) {
				imgs.erase(imgs.begin()+imgs.size());

				cout << imgs.size()<<endl<<flush;
				continue;
			}
			else if (status != Stitcher::OK) {
				cout << "Can't stitch images, error code = " << int(status) << endl;
				imgs.erase(imgs.begin()+imgs.size());
				continue ;
				//return -1;
			}

			imshow("pano", pano);
			waitKey(10);
			//if(waitKey(30) >= 0) break;
			imgs.clear();
			imgs.push_back(pano);
		}
	}
	return 0;
}
