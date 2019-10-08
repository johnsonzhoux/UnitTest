#include "StdAfx.h"
#include "image_process.h"
#include <iostream>

namespace dip {

void sharpColorImg(cv::Mat &src, cv::Mat &dst, cv::Size sz)
{
	std::vector<cv::Mat> channels;
	cv::split(src, channels);

	for (int i = 0; i < channels.size(); ++i) {
		cv::Mat tophat;
		cv::Mat blackhat;
		cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, sz);
		cv::morphologyEx(channels[i], tophat, cv::MORPH_TOPHAT, element);
		cv::morphologyEx(channels[i], blackhat, cv::MORPH_BLACKHAT, element);
		channels[i] = channels[i] + tophat - blackhat;
	}
	cv::merge(channels, dst);
}

void sharpGrayImg(cv::Mat &src, cv::Mat &dst, cv::Size sz)
{
	cv::Mat tophat;
	cv::Mat blackhat;
	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, sz);
	cv::morphologyEx(src, tophat, cv::MORPH_TOPHAT, element);
	cv::morphologyEx(src, blackhat, cv::MORPH_BLACKHAT, element);
	dst = src + tophat - blackhat;
}

void sharpGrayImgUseLap(cv::Mat &src, cv::Mat &dst, bool is_diag)
{
	cv::Mat kernel = cv::Mat::zeros(3, 3, CV_16SC1);
	if (is_diag) {
		for (int j = 0; j < 3; ++j) {
			for (int i = 0; i < 3; ++i) {
				kernel.at<short>(j, i) = -1;
			}
		}
		kernel.at<short>(1, 1) = 8;
	}
	else {
		kernel.at<short>(0, 1) = -1;
		kernel.at<short>(1, 0) = -1;
		kernel.at<short>(1, 1) = 4;
		kernel.at<short>(1, 2) = -1;
		kernel.at<short>(2, 1) = -1;
	}
	cv::Mat filter;
	cv::filter2D(src, filter, CV_8UC1, kernel);
	dst = src + filter;
}


void sharpUseUSM(cv::Mat &src, cv::Mat &dst, int amount, int radius, int thresh, int filter_type)
{
	src.copyTo(dst);
	cv::Mat blur;
	if (filter_type == FILTER_GAUSSIAN) {
		cv::GaussianBlur(src, blur, cv::Size(radius * 2 + 1, radius * 2 + 1), 0);
	}
	else {
		cv::blur(src, blur, cv::Size(radius * 2 + 1, radius * 2 + 1));
	}

	int value = 0;
	for (int j = 0; j < src.rows; ++j) {
		uchar *sptr = src.ptr<uchar>(j);
		uchar *bptr = blur.ptr<uchar>(j);
		uchar *dptr = dst.ptr<uchar>(j);
		for (int i = 0; i < src.cols*src.channels(); ++i) {
			value = static_cast<int>(sptr[i]) - static_cast<int>(bptr[i]);			
			if (std::abs(value) > thresh) {
				value = sptr[i] + amount*value / 100;
				dptr[i] = cv::saturate_cast<uchar>(value);
			}
		}
	}
}

void sharpGrayImgUseSobel(cv::Mat &src, cv::Mat &dst)
{
	cv::Mat kernel_x = cv::Mat::zeros(3, 3, CV_16SC1);
	kernel_x.at<short>(0, 0) = -1;
	kernel_x.at<short>(0, 1) = -2;
	kernel_x.at<short>(0, 2) = -1;
	kernel_x.at<short>(2, 0) = 1;
	kernel_x.at<short>(2, 1) = 2;
	kernel_x.at<short>(2, 2) = 1;

	cv::Mat kernel_y = cv::Mat::zeros(3, 3, CV_16SC1);
	kernel_y.at<short>(0, 0) = -1;
	kernel_y.at<short>(1, 0) = -2;
	kernel_y.at<short>(2, 0) = -1;
	kernel_y.at<short>(0, 2) = 1;
	kernel_y.at<short>(1, 2) = 2;
	kernel_y.at<short>(2, 2) = 1;

	cv::Mat gx, gy;
	cv::filter2D(src, gx, CV_8UC1, kernel_x);
	cv::filter2D(src, gy, CV_8UC1, kernel_y);
	//dst = src + gx + gy;
	cv::addWeighted(gx, 0.5, gy, 0.5, 0, dst);
}


void enhanceBlackRegion(cv::Mat &src, cv::Mat &dst, cv::Size sz)
{
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, sz);
	cv::Mat morp;
	cv::morphologyEx(src, morp, cv::MORPH_BLACKHAT, element);
	dst = src - morp;
}

void enhanceWhiteRegion(cv::Mat &src, cv::Mat &dst, cv::Size sz)
{
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, sz);
	cv::Mat morp;
	cv::morphologyEx(src, morp, cv::MORPH_TOPHAT, element);
	dst = src + morp;
}


/*color transfer*/
void transfer(cv::Mat &img, cv::Mat src_mean, cv::Mat src_stddev, cv::Mat dst_mean, cv::Mat dst_stddev)
{
	for (int j = 0; j < img.rows; ++j) {
		cv::Vec3b *ptr = img.ptr<cv::Vec3b>(j);
		for (int i = 0; i < img.cols; ++i) {
			ptr[i][0] = cv::saturate_cast<uchar>((ptr[i][0] - src_mean.at<double>(0, 0)) / src_stddev.at<double>(0, 0)*dst_stddev.at<double>(0, 0) + dst_mean.at<double>(0, 0));
			ptr[i][1] = cv::saturate_cast<uchar>((ptr[i][1] - src_mean.at<double>(1, 0)) / src_stddev.at<double>(1, 0)*dst_stddev.at<double>(1, 0) + dst_mean.at<double>(1, 0));
			ptr[i][2] = cv::saturate_cast<uchar>((ptr[i][2] - src_mean.at<double>(2, 0)) / src_stddev.at<double>(2, 0)*dst_stddev.at<double>(2, 0) + dst_mean.at<double>(2, 0));
		}
	}
}

/*transform src to dst*/
void colorTransfer(cv::Mat &src, cv::Mat &dst, cv::Mat &out)
{
	cv::Mat src_lab, dst_lab;
	cv::cvtColor(src, src_lab, CV_BGR2Lab);
	cv::cvtColor(dst, dst_lab, CV_BGR2Lab);

	//cv::imwrite("src.jpg", src_lab);

	cv::Mat src_mean, src_stddev, dst_mean, dst_stddev;
	cv::meanStdDev(src_lab, src_mean, src_stddev);
	cv::meanStdDev(dst_lab, dst_mean, dst_stddev);
	transfer(src_lab, src_mean, src_stddev, dst_mean, dst_stddev);
	cv::cvtColor(src_lab, out, CV_Lab2BGR);
}

void reduceGrayHighLight(cv::Mat &gray, cv::Mat &dst, float k)
{
	//multiply, C = (AxB)/255, refer http://www.360doc.com/content/15/0623/12/15361233_480084978.shtml
	cv::Mat mask;
	cv::multiply(gray, gray, mask, 1.0 / 255);
	cv::Mat inv_mask;
	cv::subtract(cv::Scalar(255), mask, inv_mask);

	//enhance part
	cv::Mat second = gray.mul(k);

	//addWeight
	cv::Mat tmp0, tmp1;
	cv::multiply(second, mask, tmp0, 1.0 / 255);
	cv::multiply(gray, inv_mask, tmp1, 1.0 / 255);
	cv::add(tmp0, tmp1, dst);
}

void reduceHighLight(cv::Mat &src, cv::Mat &dst, float k)
{
	cv::Mat gray;
	cv::cvtColor(src, gray, CV_BGR2GRAY);

	//multiply,  C = (AxB)/255, refer http://www.360doc.com/content/15/0623/12/15361233_480084978.shtml
	cv::Mat mask;
	cv::multiply(gray, gray, mask, 1.0 / 255);

	cv::Mat masks;
	cv::cvtColor(mask, masks, CV_GRAY2BGR);
	cv::Mat inv_masks;
	cv::subtract(cv::Scalar(255, 255, 255), masks, inv_masks);

	//enhance part
	cv::Mat second = src.mul(k);

	//addWeight
	cv::Mat tmp0, tmp1;
	cv::multiply(second, masks, tmp0, 1.0 / 255);
	cv::multiply(src, inv_masks, tmp1, 1.0 / 255);
	cv::add(tmp0, tmp1, dst);
}

void increaseGrayDark(cv::Mat &gray, cv::Mat &dst, float gama)
{
	cv::Mat inv_gray;
	cv::subtract(cv::Scalar(255), gray, inv_gray);

	//screen, (255-A)x(255-B)/255
	cv::Mat mask;
	cv::multiply(inv_gray, inv_gray, mask, 1.0 / 255);
	cv::Mat inv_mask;
	cv::subtract(cv::Scalar(255), mask, inv_mask);

	//enhance part
	cv::Mat second_tmp, second;
	gray.convertTo(second_tmp, CV_32FC1, 1.0 / 255);
	cv::pow(second_tmp, gama, second_tmp);
	second_tmp.convertTo(second, CV_8UC1, 255);

	//addWeight
	cv::Mat tmp0, tmp1;
	cv::multiply(second, mask, tmp0, 1.0 / 255);
	cv::multiply(gray, inv_mask, tmp1, 1.0 / 255);
	cv::add(tmp0, tmp1, dst);
}

void increaseDark(cv::Mat &src, cv::Mat &dst, float gama)
{
	cv::Mat gray;
	cv::cvtColor(src, gray, CV_BGR2GRAY);
	cv::Mat inv_gray;
	cv::subtract(cv::Scalar(255), gray, inv_gray);

	//multiply
	cv::Mat mask;
	cv::multiply(inv_gray, inv_gray, mask, 1.0 / 255);
	cv::Mat masks;
	cv::cvtColor(mask, masks, CV_GRAY2BGR);
	cv::Mat inv_masks;
	cv::subtract(cv::Scalar(255, 255, 255), masks, inv_masks);

	//enhance part
	cv::Mat second_tmp, second;
	src.convertTo(second_tmp, CV_32FC3, 1.0 / 255);
	cv::pow(second_tmp, gama, second_tmp);
	second_tmp.convertTo(second, CV_8UC3, 255);

	//addWeight
	cv::Mat tmp0, tmp1;
	cv::multiply(second, masks, tmp0, 1.0 / 255);
	cv::multiply(src, inv_masks, tmp1, 1.0 / 255);
	cv::add(tmp0, tmp1, dst);
}

void fillHole(cv::Mat &src, cv::Mat &dst)
{
	cv::Mat tmp = cv::Mat::zeros(src.rows + 2, src.cols + 2, CV_8UC1);
	src.copyTo(tmp(cv::Range(1, src.rows + 1), cv::Range(1, src.cols + 1)));
	cv::floodFill(tmp, cv::Point(0, 0), cv::Scalar(255));
	cv::Mat cut_img;
	tmp(cv::Range(1, src.rows + 1), cv::Range(1, src.cols + 1)).copyTo(cut_img);
	dst = src | ~cut_img;
}

//dst_img must be init before use
void homoFilter(cv::Mat &src_img, cv::Mat &dst_img)
{
	/* refer https://blog.csdn.net/bluecol/article/details/45788803
	 * and https://blog.csdn.net/lilingyu520/article/details/46654265
	 */

	cv::Mat src, dst;
	src_img.convertTo(src, CV_64FC1);
	dst = cv::Mat::zeros(src.size(), CV_64FC1);

	//1. ln
	for (int i = 0; i < src.rows; i++) {
		double* ptr = src.ptr<double>(i);
		for (int j = 0; j < src.cols; j++) {
			ptr[j] = log(ptr[j] + 0.0001);
		}
	}

	//spectrum
	//2. dct
	cv::Mat mat_dct = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);
	dct(src, mat_dct);

	//3. linear filter
	cv::Mat Huv;
	double gammaH = 1.5;
	double gammaL = 0.5;
	double c = 1;
	double D0 = (src.rows / 2)*(src.rows / 2) + (src.cols / 2)*(src.cols / 2);//filter radius, 
	double D = 0;
	Huv = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);
	for (int i = 0; i < src.rows; i++) {
		double * ptr = Huv.ptr<double>(i);
		for (int j = 0; j < src.cols; j++) {
			D = pow((i), 2.0) + pow((j), 2.0);
			ptr[j] = (gammaH - gammaL)*(1 - exp(-c*D / D0)) + gammaL;
		}
	}
	//std::cout << Huv.ptr<double>(0)[0] << std::endl;
	Huv.ptr<double>(0)[0] = 1.02;//adjust global brightness
	mat_dct = mat_dct.mul(Huv);

	//4. idct
	idct(mat_dct, dst);

	//5. exp
	for (int i = 0; i < dst.rows; i++) {
		double* ptr = dst.ptr<double>(i);
		for (int j = 0; j < dst.cols; j++) {
			ptr[j] = exp(ptr[j]);
		}
	}

	dst.convertTo(dst_img, dst_img.type());

}

void gamaAdjust(cv::Mat &src, cv::Mat &dst, float gama)
{
	cv::Mat tmp;
	if (src.channels() == 3) {
		src.convertTo(tmp, CV_32FC3, 1.0 / 255);
	}
	else {
		src.convertTo(tmp, CV_32FC1, 1.0 / 255);
	}
	cv::pow(tmp, gama, tmp);
	tmp.convertTo(dst, src.type(), 255);
}

double meanMat(cv::Mat &img, int thresh)
{
	cv::Mat mask;
	mask = img - cv::Scalar(thresh);
	cv::Scalar avg = mean(img, mask);
	return avg[0];
}

void imageExtract(cv::Mat &src, cv::Mat &out_mask, int channel, int thresh)
{
	int c0, c1, c2;//c0 is must extract channel
	if (channel == 0) {
		c0 = 0;
		c1 = 1;
		c2 = 2;
	}
	else if (channel == 1) {
		c0 = 1;
		c1 = 0;
		c2 = 2;
	}
	else {
		c0 = 2;
		c1 = 0;
		c2 = 1;
	}

	out_mask = cv::Mat::zeros(src.size(), CV_8UC1);
	for (int j = 0; j < src.rows; ++j) {
		cv::Vec3b *sptr = src.ptr<cv::Vec3b>(j);
		uchar *mptr = out_mask.ptr<uchar>(j);
		for (int i = 0; i < src.cols; ++i) {
			int color[3];
			color[0] = sptr[i][0];
			color[1] = sptr[i][1];
			color[2] = sptr[i][2];
			if ((color[c0] - color[c1] > thresh)
				&& (color[c0] - color[c2] > thresh)) {
				mptr[i] = 255;
			}
		}
	}

}



}//end namespace

