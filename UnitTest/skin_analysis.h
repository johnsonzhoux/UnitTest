#ifndef SKIN_ANALYSIS_H
#define SKIN_ANALYSIS_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#define EPS 2.2204e-16
#define EPS 2.2204e-8

#define COMPLEXION_WHITE_0 0
#define COMPLEXION_WHITE_1 1
#define COMPLEXION_NATURE 2
#define COMPLEXION_WHEAT 3
#define COMPLEXION_DIM 4
#define COMPLEXION_DARK 5
 
//#define SKIN_TYPE_MIXTURE 0
//#define SKIN_TYPE_OILL 1
//#define SKIN_TYPE_NEUTRAL 2
//#define SKIN_TYPE_DRY 3


#define SKIN_ANALYSIS_DEBUG

const std::string file_path = "tmp/";


class SkinAnalysis {
public:
	//default constructor
	SkinAnalysis();
	//constructor list
	SkinAnalysis(cv::Mat src, std::vector<cv::Point> pts, double sigma_spot, int spot_binary_thresh, int spot_avg_thresh,
		double sigma_pore, int pore_thresh, int oill_thresh, int wrinkle_thresh,  cv::Vec3b acne_color_l, cv::Vec3b acne_color_h,
		double blackhead_color_diff, double spot_area_l, double spot_area_h, double acne_area_l, double acne_area_h, double blackhead_area_l, double blackhead_area_h);
	//destructor
	~SkinAnalysis();
	//设置输入图片
	void setInputImg(cv::Mat &src) { src.copyTo(m_src); }
	//设置68个人脸关键点信息
	void setInputFacePoints(std::vector<cv::Point> &pts);

	/**
	* @brief 斑点及痘痘检测的参数设置
	* @param sigma 设置得越大，可以检测更大的斑点或痘痘
	* @param binary_thresh 对LOG算子检测后的灰度图进行二值化的阈值
	* @param avg_thresh 平均值大于avg_thresh的斑点或痘痘将保留下来
	*/
	inline void setSpotThresh(double sigma, int binary_thresh, int avg_thresh) { m_sigma_spot = sigma; m_spot_binary_thresh = binary_thresh; m_spot_avg_thresh = avg_thresh; }
	
	/**
	* @brief 黑头及毛孔检测参数设置
	* @param sigma 检测黑头或毛孔大小参数设置
	* @param thresh 对LOG算子检测后的灰度图进行二值化的阈值
	*/
	inline void setPoreThresh(double sigma, int thresh) { m_sigma_pore = sigma; m_pore_thresh = thresh; }

	/**
	* @brief 油性检测阈值设置
	* @param thresh 顶帽操作后检测为油性的阈值
	*/
	inline void setOillThresh(int thresh) { m_oill_thresh = thresh; }//be also used to skinTypeDetect
	
	/**
	* @brief 皱纹检测阈值设置
	* @param thresh Frangi滤波后，检测为皱纹的阈值
	*/
	inline void setWrinkleThresh(int thresh) { m_wrinkle_thresh = thresh; }
	
	/**
	* @brief 设置痘痘在hsv空间的颜色范围
	* @param low 检测为痘痘的颜色下限
	* @param high 检测为痘痘的颜色上限
	*/
	inline void setAcneColorRange(cv::Vec3b low, cv::Vec3b high) { m_acne_color_l = low; m_acne_color_h = high; }

	/**
	* @brief 设置判定为黑头的灰度值差异
	* @param diff 检测为黑头的阈值为 thresh = mean-diff
	*/
	inline void setBlackheadColorDiff(double diff) { m_blackhead_color_diff = diff; }

	/**
	* @brief 设置检测为斑点的面积范围
	* @param low 检测为斑点的最小面积
	* @param high 检测为斑点的最大面积
	*/
	inline void setSpotAreaRange(double low, double high) { m_spot_area_l = low; m_spot_area_h = high; }

	/**
	* @brief 设置检测为痘痘的面积范围
	* @param low 检测为痘痘的最小面积
	* @param high 检测为痘痘的最大面积
	*/
	inline void setAcneAreaRange(double low, double high) { m_acne_area_l = low; m_acne_area_h = high; }

	/**
	* @brief 设置检测为黑头或毛孔的面积范围
	* @param low 检测为黑头或毛孔的最小面积
	* @param high 检测为黑头或毛孔的最大面积
	*/
	inline void setBlackheadAreaRange(double low, double high) { m_blackhead_area_l = low; m_blackhead_area_h = high; }

	/**
	* @brief 设置显示纹理特征点面积大小的阈值
	* @param thresh 面积阈值，大于该阈值显示
	*/
	inline void setTextureThresh(int thresh) { m_texture_thresh = thresh; }

	//void printParams();

	//图片预处理，包括创建灰度图，生成mask等
	void preprocess();

	//进行皮肤分析
	void analysis();

public:
	//output
	int m_spot_num;//斑点的个数
	int m_acne_num;//痘痘的个数
	int m_blackhead_num;//黑头的个数
	int m_pore_num;//毛孔的个数
	double m_oill_percent;//油性的百分比
	//int m_skin_type;//肤质
	double m_oill_center_percent;//油性区域在人脸中间区域的百分比
	double m_oill_face_percent;//油性区域在脸颊的百分比
	int m_wrinkle_len;//皱纹的总长度
	int m_marionette_len;//木偶纹的长度
	int m_nasolabial_len;//法令纹的长度
	int m_frown_len;//川字纹的长度
	int m_forehead_len;//额纹的长度
	int m_eye_wrinkle_len;//眼皱纹的长度
	int m_complexion_number;//肤色类型 0-->black, 1-->white, 2-->red, 3-->brown, 4-->yellow
	float m_blackeye_level;//黑眼圈水平
	int m_texture_bulge_num;//纹理检测凸出部分的个数
	int m_texture_sunken_num;//纹理检测凹陷部分的个数

	cv::Mat m_spot_img;//斑点检测结果图
	cv::Mat m_acne_img;//痘痘检测结果图
	cv::Mat m_blackhead_img;//黑头检测结果图
	cv::Mat m_pore_img;//毛孔检测结果图
	cv::Mat m_oill_img;//油性检测结果图
	cv::Mat m_wrinkle_img;//皱纹检测结果图
	cv::Mat m_texture_img;//纹理检测结果图

public:
	void createGrayImg();//创建灰度图
	void createCGrayImg();//创建自定义灰度图
	void splitImg();//将原始图片的三个通道分离出来
	void createHSVImg();//创建hsv颜色空间图片
	void createSpotMask();//创建用于斑点检测的mask
	void createPoreMask();//创建用于毛孔检测的mask
	void createWrinkleMask();//创建用于皱纹检测的mask
	void createTextureMask();//创建用于纹理检测的msk
	void createAllFaceMask();//创建全脸检测mask
	void createMarionetteMask();//木偶纹
	void crateNasolabialMask();//法令纹
	void createFrownMask();//川字纹
	void createForeheadMask();//额纹
	void createEyeWrinkleMask();//眼皱纹
	void createSkinTypeMask();//创建肤质类型mask，油性、中性、干性、混合型
	cv::Scalar meanHSV(cv::Mat &hsv, cv::Mat &mask);//求解hsv颜色空间图片的均值

	/**
	* @brief 创建LOG算子核
	* @param size 核大小，需为奇数，如3、5、7
	* @param sigma 高斯函数的标准差
	*/
	cv::Mat createLOGKernel(int size, double sigma);

	void createLOGImg(cv::Mat &src, double sigma, cv::Mat &dst, int padding);//anew set input image?
	void sharpImg(cv::Mat &gray, cv::Mat &dst);
	void reduceHighLight(cv::Mat &src, cv::Mat &dst, float k);
	void increaseDark(cv::Mat &src, cv::Mat &dst, float gama);
	void acneAndSpotDetect(cv::Mat &gray_img);//痘痘和斑点检测
	void blackheadAndPoreDetect(cv::Mat &gray_img);//黑头和毛孔检测
	void oillDetect(cv::Mat &gray_img);

	/*wrinkle detection*/
	void hessian2D(cv::Mat &image, float sigma, cv::Mat &Dxx, cv::Mat &Dxy, cv::Mat &Dyy);
	cv::Mat logical(cv::Mat &M);
	void eig2image(cv::Mat &Dxx, cv::Mat &Dxy, cv::Mat &Dyy, cv::Mat &Lambda1, cv::Mat &Lambda2, cv::Mat &Ix, cv::Mat &Iy);
	std::vector<float> createSigma(float start, float step, float end);
	void maxAmplitude(std::vector<cv::Mat> &imgs, cv::Mat &dst);
	void frangiFilter2D(cv::Mat &I, cv::Mat &dst, std::vector<float> sigmas, float c0, float c1);
	void frangiFilter2D(cv::Mat &I, cv::Mat &dst, float sigma, float c0, float c1);
	cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1);
	void wrinkleDetect(cv::Mat &gray_img);

	void complexionClassify();
	void blackEyeRecognize();

	void emboss(cv::Mat &src, cv::Mat &dst, cv::Mat &refer, float ratio);
	void textureDetect();

private:
	//input
	cv::Mat m_src;//rgb image
	cv::Mat m_mask_as;//mask of acne and spot
	cv::Mat m_mask_bp;//mask of blackhead and pore
	cv::Mat m_mask_oill;//oill mask
	cv::Mat m_mask_wrinkle;//wrinkle mask
	cv::Mat m_mask_texture;
	cv::Mat m_mask_complexion;
	cv::Mat m_mask_all;
	cv::Mat m_mask_marionette;//木偶纹
	cv::Mat m_mask_nasolabial;//法令纹
	cv::Mat m_mask_frown;//川字纹
	cv::Mat m_mask_forehead;
	cv::Mat m_mask_eye_wrinkle;
	cv::Mat m_mask_skintype_center;
	cv::Mat m_mask_skintype_face;
	double m_sigma_spot;
	int m_spot_binary_thresh;
	int m_spot_avg_thresh;
	double m_sigma_pore;
	int m_pore_thresh;
	int m_oill_thresh;
	int m_wrinkle_thresh;
	cv::Vec3b m_acne_color_l;
	cv::Vec3b m_acne_color_h;
	int m_blackhead_color_diff;
	double m_spot_area_l;
	double m_spot_area_h;
	double m_acne_area_l;
	double m_acne_area_h;
	double m_blackhead_area_l;//also used to pore
	double m_blackhead_area_h;//also used to pore
	double m_texture_thresh;

	std::vector<cv::Point> m_pts;//face feature points
	
	//tmp
	cv::Mat m_gray_img;
	cv::Mat m_cgray_img;
	cv::Mat m_blue_img;
	cv::Mat m_green_img;
	cv::Mat m_red_img;
	cv::Mat m_hsv_img;
	std::vector<std::vector<cv::Point>> m_wrinkle_contours;
};





#endif



