#ifndef IMAGE_PROCESS_H
#define IMAGE_PROCESS_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace dip {

#define BLUE_CHANNEL 0
#define GREEN_CHANNEL 1
#define RED_CHANNEL 2


enum FilterTypes {
	FILTER_GAUSSIAN = 0, 
	FILTER_MEAN = 1, 
};


/**
* @brief: Use tophat and blackhat to sharp color image.
* @param [in] src: Input color image
* @param [out] dst: Output color image
* @param [in] sz: Kernel size of tophat and blackhat
*/
void sharpColorImg(cv::Mat &src, cv::Mat &dst, cv::Size sz);

/**
* @brief: Use tophat and blackhat to sharp gray image.
* @param [in] src: Input gray image
* @param [out] dst: Output gray image
* @param [in] sz: Kernel size of tophat and blackhat
*/
void sharpGrayImg(cv::Mat &src, cv::Mat &dst, cv::Size sz);

/**
* @brief: Use laplace to sharp gray image.
* @param [in] src: Input gray image
* @param [out] dst: Output gray image
* @param [in] is_diag: Whether use diagonal information
*/
void sharpGrayImgUseLap(cv::Mat &src, cv::Mat &dst, bool is_diag);

/**
* @brief: Use USM to sharp image
* @param [in] src: Input image
* @param [out] dst: Output image
* @param [in] amount: The degree of sharp
* @param [in] radius: Filter's radius
* @param [in] thresh: The difference between original value and filtered value is bigger than thresh will sharp.
* @param [in] filter_type: The type of filter, see FilterTypes. 
*/
void sharpUseUSM(cv::Mat &src, cv::Mat &dst, int amount, int radius, int thresh = 0, int filter_type = FILTER_MEAN);

/**
* @brief: Use sobel to sharp gray image.
* @param [in] src: Input gray image
* @param [out] dst: Output gray image
*/
void sharpGrayImgUseSobel(cv::Mat &src, cv::Mat &dst);

/**
* @brief: Use blackhat to enhance black region.
* @param [in] src: Input image
* @param [out] dst: Output image
* @param [in] sz: Kernel size of blackhat
*/
void enhanceBlackRegion(cv::Mat &src, cv::Mat &dst, cv::Size sz);

/**
* @brief: Use tophat to enhance white region.
* @param [in] src: Input image
* @param [out] dst: Output image
* @param [in] sz: Kernel size of tophat
*/
void enhanceWhiteRegion(cv::Mat &src, cv::Mat &dst, cv::Size sz);

/**
* @brief: Use mean and stddev to realize color transfer.
* @param [in/out] img: Color image
* @param [in] src_mean: Mean of src image
* @param [in] src_stddev: Stddev of src image
* @param [in] dst_mean: Mean of dst image
* @param [in] dst_stddev: Stddev of dst image
*/
void transfer(cv::Mat &img, cv::Mat src_mean, cv::Mat src_stddev, cv::Mat dst_mean, cv::Mat dst_stddev);

/**
* @brief: Realize color transfer from src image to dst image, this function call transfer function.
* @param [in] src: Color image
* @param [in] dst: target style image
* @param [out] out: Output image
*/
void colorTransfer(cv::Mat &src, cv::Mat &dst, cv::Mat &out);

/**
* @brief: Reduce hightlight for gray image.
* @param [in] src: Input gray image
* @param [out] dst: Output gray image
* @param [in] k: The range is [0, 1], more little more dark
*/
void reduceGrayHighLight(cv::Mat &gray, cv::Mat &dst, float k);

 /**
 * @brief: Reduce hightlight for color image.
 * @param [in] src: Input color image
 * @param [out] dst: Output color image
 * @param [in] k: The range is [0, 1], more little more dark
 */
void reduceHighLight(cv::Mat &src, cv::Mat &dst, float k);

/**
* @brief: Increase darkness for gray image.
* @param [in] src: Input gray image
* @param [out] dst: Output gray image
* @param [in] gama: The range is [0, 1], more little more light
*/
void increaseGrayDark(cv::Mat &gray, cv::Mat &dst, float gama);

/**
* @brief: Increase darkness for color image.
* @param [in] src: Input color image
* @param [out] dst: Output color image
* @param [in] gama: The range is [0, 1], more little more light
*/
void increaseDark(cv::Mat &src, cv::Mat &dst, float gama);

/**
* @brief: Fill dark hole for binary image.
* @param [in] src: Input binary image
* @param [out] dst: Output binary image
*/
void fillHole(cv::Mat &src, cv::Mat &dst);

/**
* @brief: Homomorphic filter.
* @param [in] src_img: Input image
* @param [out] dst_img: Output image
* @note: src_img size must be even
*/
void homoFilter(cv::Mat &src_img, cv::Mat &dst_img);

/**
* @brief: Use gama curve to adjust image.
* @param [in] src: Input image
* @param [out] dst: Output image
* @param [in] gama: The range is [0, 1], more little more light
*/
void gamaAdjust(cv::Mat &src, cv::Mat &dst, float gama);

/**
* @brief: Get mean value of gray image.
* @param [in] src: Input gray image
* @param [in] thresh: The pixel value bigger than thresh will be used to compute mean value
* @param return: mean value
*/
double meanMat(cv::Mat &img, int thresh);

 /**
 * @brief: Extract specific channel mask
 * @param [in] src: Input color image
 * @param [out] out_mask: Output mask
 * @param [in] channel: channel number, the value can be BLUE_CHANNEL¡¢GREEN_CHANNEL¡¢RED_CHANNEL
 * @param [in] thresh: the biggest must bigger than other two channel value
 */
void imageExtract(cv::Mat &src, cv::Mat &out_mask, int channel, int thresh = 10);


}//end namespace
#endif