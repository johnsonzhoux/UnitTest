#include "StdAfx.h"
#include "skin_polarity.h"
#include "image_process.h"
#include "skin_analysis.h"

SkinPolarity::SkinPolarity()
{

}

SkinPolarity::SkinPolarity(cv::Mat &src, std::vector<cv::Point> pts)
{
	m_src = src;
	m_pts.assign(pts.begin(), pts.end());
}

SkinPolarity::~SkinPolarity()
{
	
}


std::vector<double> SkinPolarity::createSigma(double start, double step, double end)
{
	std::vector<double> sigma;
	while (start <= end + 1e-8) {
		double s = exp(start);
		sigma.push_back(s);
		start += step;
	}
	return sigma;
}

cv::Mat SkinPolarity::createLOGKernel(int size, double sigma)
{
	cv::Mat H(size, size, CV_64F);
	int cx = (size - 1) / 2;
	int cy = (size - 1) / 2;
	double sum = 0;
	for (int i = 0; i < H.rows; i++) {
		for (int j = 0; j < H.cols; j++) {
			int nx = i - cx;
			int ny = j - cy;
			double s = -1 / (3.1415926 * sigma*sigma*sigma*sigma);
			s = s* (1 - (nx*nx + ny*ny) / (2 * sigma*sigma));
			s = s*exp(-(nx*nx + ny*ny) / (2 * sigma*sigma));
			sum += s;
			H.at<double>(i, j) = s;
		}
	}

	double mean = sum / (size*size);
	for (int i = 0; i < H.rows; i++) {
		for (int j = 0; j < H.cols; j++) {
			H.at<double>(i, j) -= mean;
		}
	}
	return H;
}

cv::Mat SkinPolarity::createScaleSpace(cv::Mat &im_in, std::vector<double> &sigma, std::vector<cv::Mat> &scale_space)
{
	cv::Mat merge = cv::Mat::zeros(im_in.size(), im_in.type());
	for (int i = 0; i < sigma.size(); i++) {
		int n = ceil(sigma[i] * 3) * 2 + 1;
		cv::Mat LOG = createLOGKernel(n, sigma[i]);
		cv::Mat filter;
		filter2D(im_in.mul(sigma[i] * sigma[i]), filter, -1, LOG, cv::Point(-1, -1));
		scale_space.push_back(filter);
		merge += filter;
	}
	return merge;
}

void SkinPolarity::createLogMergeImg(cv::Mat &src, cv::Mat &dst, std::vector<double> sigmas,  int padding)
{
	cv::Mat addborder, norm;
	cv::copyMakeBorder(src, addborder, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(255));
	normalize(addborder, norm, 1, 0, cv::NORM_MINMAX, CV_64F);
	std::vector<cv::Mat> all_ims;
	cv::Mat merge = createScaleSpace(norm, sigmas, all_ims);
	cv::Mat merge2;
	merge.convertTo(merge2, CV_8UC1, 255.0);
	merge2(cv::Range(10, merge2.rows - 10), cv::Range(10, merge2.cols - 10)).copyTo(dst);
}

void SkinPolarity::createLOGImg(cv::Mat &src, double sigma, cv::Mat &dst, int padding)
{
	//normalization
	cv::Mat addborder, norm;
	cv::copyMakeBorder(src, addborder, padding, padding, padding, padding, cv::BORDER_CONSTANT, cv::Scalar(255));
	normalize(addborder, norm, 1, 0, cv::NORM_MINMAX, CV_64F);

	int n = ceil(sigma * 3) * 2 + 1;
	cv::Mat kernel = createLOGKernel(n, sigma);
	cv::Mat LOG;
	cv::filter2D(norm.mul(sigma * sigma), LOG, -1, kernel, cv::Point(-1, -1));

	cv::Mat tmp;
	LOG.convertTo(tmp, CV_8UC1, 255);
	tmp(cv::Range(padding, tmp.rows - padding), cv::Range(padding, tmp.cols - padding)).copyTo(dst);
}

void SkinPolarity::transToBrownImage(cv::Mat &src, cv::Mat &dst)
{
	//get blue image
	std::vector<cv::Mat> bgr_channels;
	cv::split(src, bgr_channels);
	cv::Mat blue = bgr_channels[0];
	cv::Mat green = bgr_channels[1];
	cv::Mat red = bgr_channels[2]*0.8;

	std::vector<double> sigmas = createSigma(1.0, 0.3, 4.0);
	cv::Mat spot;
	createLogMergeImg(blue, spot, sigmas);
	spot.copyTo(m_brown_gray);

	cv::Mat illu;
	illu = red + spot;
	illu = cv::Scalar(255) - illu;

	std::vector<cv::Mat> la_channels(3);
	la_channels[0] = illu;
	la_channels[1] = cv::Mat(src.size(), CV_8UC1, cv::Scalar(168/*143*/));
	la_channels[2] = cv::Mat(src.size(), CV_8UC1, cv::Scalar(180/*154*/));
	cv::Mat la_merge;
	cv::merge(la_channels, la_merge);
	cv::cvtColor(la_merge, dst, CV_Lab2BGR);
}

void SkinPolarity::enhanceBlackRegion(cv::Mat &src, cv::Mat &dst, cv::Size sz)
{
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, sz);
	cv::Mat morp;
	cv::morphologyEx(src, morp, cv::MORPH_BLACKHAT, element);
	dst = src - morp;
}


void SkinPolarity::transToRedImage(cv::Mat &src, cv::Mat &dst)
{
	cv::Mat luv;
	cv::cvtColor(src, luv, CV_RGB2Luv);
	std::vector<cv::Mat> luv_channels;
	cv::split(luv, luv_channels);
	cv::Mat img_l = luv_channels[0];
	cv::Mat img_u = luv_channels[1];
	cv::Mat img_v = luv_channels[2];

	//dip::enhanceContrast(img_v, img_v, 1.2, -30);
	cv::Mat img_b, img_g, img_r;
	img_v.convertTo(img_b, -1, 1.9, -70 * 0.9);
	img_b.convertTo(img_g, -1, 1, -3);
	img_v.convertTo(img_r, -1, 1, 70);
	cv::Mat rgb_img;
	std::vector<cv::Mat> bgr_channels;
	bgr_channels.push_back(img_b);
	bgr_channels.push_back(img_g);
	bgr_channels.push_back(img_r);
	cv::merge(bgr_channels, rgb_img);
	cv::Mat red;
	rgb_img.convertTo(red, -1, 1.6, -86 * 0.6);
	dip::sharpUseUSM(red, dst, 100, 5);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_red_img.bmp", dst);
#endif
}

void SkinPolarity::createSpotMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	float pec = 0.1;
	cv::Point pt;
	std::vector<cv::Point> pts(30);
	std::vector<std::vector<cv::Point>> contours;
	pts[0].x = (m_pts[21].x + m_pts[22].x) / 2 - vy.x *0.6;
	pts[0].y = (m_pts[21].y + m_pts[22].y) / 2 - vy.y*0.6;

	pts[1] = m_pts[18] - vy*0.4;
	pts[2] = m_pts[17] - vy*pec;
	pts[3] = m_pts[18] - vy*pec;
	pts[4] = m_pts[19] - vy*pec;
	pts[5] = m_pts[20] - vy*pec;
	pts[6] = m_pts[21] - vy*pec + vx*pec;
	pts[7] = cv::Point(m_pts[21].x, m_pts[39].y) + vy*pec;
	pts[8] = m_pts[1] + vx*pec;
	pts[9] = m_pts[4] + vx*pec;
	pts[10] = m_pts[5] - vy*pec;
	pts[11] = cv::Point(m_pts[40].x, m_pts[31].y);
	pts[12] = cv::Point(m_pts[40].x, m_pts[30].y);
	pts[13] = cv::Point(m_pts[21].x, m_pts[29].y);
	pts[14] = cv::Point(m_pts[21].x, m_pts[30].y);
	pts[15] = m_pts[30] + vy*0.1;

	pts[16] = cv::Point(m_pts[22].x, m_pts[30].y);
	pts[17] = cv::Point(m_pts[22].x, m_pts[29].y);
	pts[18] = cv::Point(m_pts[47].x, m_pts[30].y);
	pts[19] = cv::Point(m_pts[47].x, m_pts[35].y);
	pts[20] = m_pts[11] - vy*pec;
	pts[21] = m_pts[12] - vx*pec;
	pts[22] = m_pts[15] - vx*pec;
	pts[23] = cv::Point(m_pts[22].x, m_pts[42].y) + vy*pec;
	pts[24] = m_pts[22] - vx*pec - vy*pec;
	pts[25] = m_pts[23] - vy*pec;
	pts[26] = m_pts[24] - vy*pec;
	pts[27] = m_pts[25] - vy*pec;
	pts[28] = m_pts[26] - vy*pec;
	pts[29] = m_pts[25] - vy*0.3;
	contours.push_back(pts);
	m_mask_spot = cv::Mat::zeros(m_src.size(), CV_8UC1);
	cv::drawContours(m_mask_spot, contours, 0, cv::Scalar(255), -1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_spot.bmp", m_mask_spot);
#endif
}

double SkinPolarity::meanMat(cv::Mat &img, int thresh)
{
	cv::Mat mask;
	mask = img - cv::Scalar(thresh);
	cv::Scalar avg = mean(img, mask);
	return avg[0];
}


void SkinPolarity::preprocess()
{
	createSpotMask();
	transToBrownImage(m_src, m_brown_img);
	transToRedImage(m_src, m_red_img);
}


/*spot detection*/
void SkinPolarity::brownAnalysis(double thresh, double area_l, double area_h)
{
	//binary
	cv::Mat binary;
	cv::threshold(m_brown_gray, binary, thresh, 255, CV_8UC1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "spot_1_log_binary.jpg", binary);
#endif

	//spot acne img
	cv::Mat spot_img;
	binary.copyTo(spot_img, m_mask_spot);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "spot_2_crop.jpg", spot_img);
#endif

	//morphology
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	cv::morphologyEx(spot_img, spot_img, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
	element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::morphologyEx(spot_img, spot_img, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "spot_3_crop_morp.jpg", spot_img);
#endif

	//draw spots
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(spot_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	m_brown_img.copyTo(m_brown_dst);
	m_brown_spot_num = 0;
	for (int j = 0; j < contours.size(); ++j) {
		double area = cv::contourArea(contours[j], false);
		if (area >= area_l && area < area_h) {
			cv::drawContours(m_brown_dst, contours, j, cv::Scalar(0, 255, 0));
			m_brown_spot_num++;
		}
	}
}

void SkinPolarity::redAnalysis(double log_sigma, double binary_thresh, double mean_thresh, double area_l, double area_h)
{
	//cv::Mat gray;
	//cv::cvtColor(m_red_img, gray, CV_BGR2GRAY);
	std::vector<cv::Mat> bgr_channels;
	cv::split(m_red_img, bgr_channels);
	cv::Mat gray = bgr_channels.at(2);
	//gray.convertTo(gray, -1, 1.2, 10);

	//cv::Mat morp;
	//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
	//cv::morphologyEx(gray, morp, cv::MORPH_BLACKHAT, element);
	//gray = gray - morp;
	//cv::imwrite(file_path + "red_morp_black.jpg", bgr_channels[2]);

	//LOG image
	cv::Mat log_img;
	createLOGImg(gray, log_sigma, log_img);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "red_0_log_img.jpg", log_img);
#endif

	//binary
	cv::Mat binary;
	cv::threshold(log_img, binary, binary_thresh, 255, CV_8UC1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "red_1_log_binary.jpg", binary);
#endif

	//spot acne img
	cv::Mat spot_img;
	binary.copyTo(spot_img, m_mask_spot);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "red_2_crop.jpg", spot_img);
#endif

	//morphology
//	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
//	cv::morphologyEx(spot_img, spot_img, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
//	element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
//	cv::morphologyEx(spot_img, spot_img, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
//#ifdef SKIN_ANALYSIS_DEBUG
//	cv::imwrite(file_path + "red_3_crop_morp.jpg", spot_img);
//#endif

	//draw spots
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(spot_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	m_red_img.copyTo(m_red_dst);
	m_red_spot_num = 0;
	for (int j = 0; j < contours.size(); ++j) {
		double area = cv::contourArea(contours[j], false);
		//cv::Point2f center;
		//float radius = 0;
		//cv::minEnclosingCircle(cv::Mat(contours[j]), center, radius);
		//float percentage = area / (CV_PI*radius*radius);

		cv::Rect min_rect = cv::boundingRect(cv::Mat(contours[j]));
		cv::Mat roi = log_img(min_rect);
		double avg = meanMat(roi, binary_thresh);

		if (area >= area_l && area < area_h/* && percentage < 0.5*/ && avg > mean_thresh) {
			cv::drawContours(m_red_dst, contours, j, cv::Scalar(255, 288, 0), -1);
			m_red_spot_num++;
		}
	}

}



