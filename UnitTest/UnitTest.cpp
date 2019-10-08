// UnitTest.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "cv.h"
#include "highgui.h"
#include "skin_analysis.h"
using namespace cv;
int _tmain(int argc, _TCHAR* argv[])
{
	Mat pic;
	cvNamedWindow("原图",0);
//	pic = imread("RNew.jpg");
	pic = imread("FNew.jpg");

	if (!pic.empty())
	{
		imshow("原图", pic);
	}







	//斑点检测
	SkinAnalysis spot;
	Mat hsvImg;
	vector<Mat> mChannels;
	vector<Mat> mHSVChannels;
	split(pic, mChannels);
	Mat mat_blue;
	Mat mat_green;
	Mat mat_red;
	mat_green = mChannels[1];
	mat_red = mChannels[2];
	mat_blue = mChannels[0];
	//spot.acneAndSpotDetect(mat_blue);
	Mat greenPoint =Mat::zeros( mat_blue.rows, mat_blue.cols, CV_8UC1);
	Mat yellowPoint = Mat::zeros( mat_blue.rows, mat_blue.cols, CV_8UC1);
	Mat redPoint = Mat::zeros(mat_blue.rows, mat_blue.cols, CV_8UC1);
	for (int i=0; i<mat_blue.rows; i++)
	{
		for(int j=0 ; j<mat_blue.cols; j++)
		{
			if (mat_green.at<uchar>(i ,j)>130 && mat_red.at<uchar>(i, j)<50&&mat_blue.at<uchar>(i, j)<50)
			{
				greenPoint.at<uchar>(i, j) = 255;
			}
			if (mat_red.at<uchar>(i, j)>200 && mat_green.at<uchar>(i,j)>200&&mat_blue.at<uchar>(i,j)<50)
			{
				yellowPoint.at<uchar>(i, j) = 255;
			}
			if (mat_red.at<uchar>(i,j)>150 && mat_green.at<uchar>(i,j)<80 && mat_blue.at<uchar>(i,j)<80 && j<mat_blue.cols/3)
			{
				redPoint.at<uchar>(i,j) = 255;
			}
		}
	}
	cvNamedWindow("绿点", 0);
	imshow("绿点", greenPoint);
	imwrite("黄点检测.jpg", yellowPoint);
	imwrite("绿点检测.jpg", greenPoint);
	Mat output = yellowPoint+greenPoint+redPoint;
	imwrite("out.jpg", output);
	vector<Vec3f> cir;
	vector<int> dist;
	HoughCircles(output, cir, CV_HOUGH_GRADIENT, 1, 50, 100, 15, 20, 100 );
	std::cout<<"circle size"<<cir.size()<<std::endl;
	for (int i=0; i<cir.size(); i++)
	{
		circle(pic, Point(cir[i][0], cir[i][1]), cir[i][2], Scalar(255,0,0), 1, 8);
		 
	}

	if (cir.size()==3)
	{
	}
	imwrite("hough.jpg", pic);
	cv::Mat logh_binary;
	cv::threshold(mat_red, logh_binary, 120,255, CV_8UC1);
	cvtColor(pic, hsvImg, CV_BGR2HSV);
	split(hsvImg, mHSVChannels);
	cvNamedWindow("red", 0);
	cvNamedWindow("green", 0);
	cvNamedWindow("blue", 0);
	cvNamedWindow("H", 0);
	cvNamedWindow("S", 0);
	cvNamedWindow("V", 0);
	imshow("red", mat_red);
	imshow("green", mat_green);
	imshow("blue", mat_blue);
	imshow("H", mHSVChannels[0]);
	imshow("S", mHSVChannels[1]);
	imshow("V", mHSVChannels[2]);

	//cv::threshold(mHSVChannels[2], logh_binary, 100, 255, CV_8UC1);
//	imshow("V", logh_binary);

	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 200;

	params.filterByArea = true;
	params.minArea = 10;
	params.maxArea = 6000;

	params.filterByCircularity = true;
	params.minCircularity = 0.5;

	params.filterByConvexity = false;
	params.minConvexity = 0.9;

	params.filterByInertia = false;
	params.maxInertiaRatio = 0.5;


	SimpleBlobDetector detertor(params);
	vector<KeyPoint> keypoints;

	detertor.detect( output, keypoints);

	Mat img_keypoints;
	drawKeypoints( output, keypoints, img_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


	for (int i=0; i<keypoints.size(); i++)
	{
		std::cout<<i<<"_"<<keypoints.at(i).pt.x<<"_"<<keypoints.at(i).pt.y<<std::endl;
	}
	imshow("V", img_keypoints);
	imwrite("0311-1A_Mergeblob.jpg", img_keypoints);



	waitKey(0);


	return 0;
}

