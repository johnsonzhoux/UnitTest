/* skin analysis of polarized light.
 * include brown spots and red region.
 */
#ifndef SKIN_POLARITY_H
#define SKIN_POLARITY_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


class SkinPolarity {
public:
	SkinPolarity();
	SkinPolarity(cv::Mat &src, std::vector<cv::Point> pts);
	~SkinPolarity();
	//brown
	std::vector<double> createSigma(double start, double step, double end);
	cv::Mat createLOGKernel(int size, double sigma);
	cv::Mat createScaleSpace(cv::Mat &im_in, std::vector<double> &sigma, std::vector<cv::Mat> &scale_space);
	void createLogMergeImg(cv::Mat &src, cv::Mat &dst, std::vector<double> sigmas,  int padding = 10);
	void createLOGImg(cv::Mat &src, double sigma, cv::Mat &dst, int padding = 10);
	void transToBrownImage(cv::Mat &src, cv::Mat &dst);
	//read
	void enhanceBlackRegion(cv::Mat &src, cv::Mat &dst, cv::Size sz);
	void transToRedImage(cv::Mat &src, cv::Mat &dst);

	
	void createSpotMask();
	double meanMat(cv::Mat &img, int thresh);

	void preprocess();
	void brownAnalysis(double thresh, double area_l, double area_h);
	void redAnalysis(double log_sigma, double binary_thresh, double mean_thresh, double area_l, double area_h);

private:
	//input
	cv::Mat m_src;
	std::vector<cv::Point> m_pts;//face feature points

	//tmp
	cv::Mat m_mask_spot;
	cv::Mat m_brown_gray;
	
public:
	cv::Mat m_brown_img;
	cv::Mat m_brown_dst;
	int m_brown_spot_num;

	cv::Mat m_red_img;
	cv::Mat m_red_dst;
	int m_red_spot_num;
	
};



#endif