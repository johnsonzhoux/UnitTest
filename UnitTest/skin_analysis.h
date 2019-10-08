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
	//��������ͼƬ
	void setInputImg(cv::Mat &src) { src.copyTo(m_src); }
	//����68�������ؼ�����Ϣ
	void setInputFacePoints(std::vector<cv::Point> &pts);

	/**
	* @brief �ߵ㼰�������Ĳ�������
	* @param sigma ���õ�Խ�󣬿��Լ�����İߵ�򶻶�
	* @param binary_thresh ��LOG���Ӽ���ĻҶ�ͼ���ж�ֵ������ֵ
	* @param avg_thresh ƽ��ֵ����avg_thresh�İߵ�򶻶�����������
	*/
	inline void setSpotThresh(double sigma, int binary_thresh, int avg_thresh) { m_sigma_spot = sigma; m_spot_binary_thresh = binary_thresh; m_spot_avg_thresh = avg_thresh; }
	
	/**
	* @brief ��ͷ��ë�׼���������
	* @param sigma ����ͷ��ë�״�С��������
	* @param thresh ��LOG���Ӽ���ĻҶ�ͼ���ж�ֵ������ֵ
	*/
	inline void setPoreThresh(double sigma, int thresh) { m_sigma_pore = sigma; m_pore_thresh = thresh; }

	/**
	* @brief ���Լ����ֵ����
	* @param thresh ��ñ��������Ϊ���Ե���ֵ
	*/
	inline void setOillThresh(int thresh) { m_oill_thresh = thresh; }//be also used to skinTypeDetect
	
	/**
	* @brief ���Ƽ����ֵ����
	* @param thresh Frangi�˲��󣬼��Ϊ���Ƶ���ֵ
	*/
	inline void setWrinkleThresh(int thresh) { m_wrinkle_thresh = thresh; }
	
	/**
	* @brief ���ö�����hsv�ռ����ɫ��Χ
	* @param low ���Ϊ��������ɫ����
	* @param high ���Ϊ��������ɫ����
	*/
	inline void setAcneColorRange(cv::Vec3b low, cv::Vec3b high) { m_acne_color_l = low; m_acne_color_h = high; }

	/**
	* @brief �����ж�Ϊ��ͷ�ĻҶ�ֵ����
	* @param diff ���Ϊ��ͷ����ֵΪ thresh = mean-diff
	*/
	inline void setBlackheadColorDiff(double diff) { m_blackhead_color_diff = diff; }

	/**
	* @brief ���ü��Ϊ�ߵ�������Χ
	* @param low ���Ϊ�ߵ����С���
	* @param high ���Ϊ�ߵ��������
	*/
	inline void setSpotAreaRange(double low, double high) { m_spot_area_l = low; m_spot_area_h = high; }

	/**
	* @brief ���ü��Ϊ�����������Χ
	* @param low ���Ϊ��������С���
	* @param high ���Ϊ������������
	*/
	inline void setAcneAreaRange(double low, double high) { m_acne_area_l = low; m_acne_area_h = high; }

	/**
	* @brief ���ü��Ϊ��ͷ��ë�׵������Χ
	* @param low ���Ϊ��ͷ��ë�׵���С���
	* @param high ���Ϊ��ͷ��ë�׵�������
	*/
	inline void setBlackheadAreaRange(double low, double high) { m_blackhead_area_l = low; m_blackhead_area_h = high; }

	/**
	* @brief ������ʾ���������������С����ֵ
	* @param thresh �����ֵ�����ڸ���ֵ��ʾ
	*/
	inline void setTextureThresh(int thresh) { m_texture_thresh = thresh; }

	//void printParams();

	//ͼƬԤ�������������Ҷ�ͼ������mask��
	void preprocess();

	//����Ƥ������
	void analysis();

public:
	//output
	int m_spot_num;//�ߵ�ĸ���
	int m_acne_num;//�����ĸ���
	int m_blackhead_num;//��ͷ�ĸ���
	int m_pore_num;//ë�׵ĸ���
	double m_oill_percent;//���Եİٷֱ�
	//int m_skin_type;//����
	double m_oill_center_percent;//���������������м�����İٷֱ�
	double m_oill_face_percent;//�������������յİٷֱ�
	int m_wrinkle_len;//���Ƶ��ܳ���
	int m_marionette_len;//ľż�Ƶĳ���
	int m_nasolabial_len;//�����Ƶĳ���
	int m_frown_len;//�����Ƶĳ���
	int m_forehead_len;//���Ƶĳ���
	int m_eye_wrinkle_len;//�����Ƶĳ���
	int m_complexion_number;//��ɫ���� 0-->black, 1-->white, 2-->red, 3-->brown, 4-->yellow
	float m_blackeye_level;//����Ȧˮƽ
	int m_texture_bulge_num;//������͹�����ֵĸ���
	int m_texture_sunken_num;//�����ⰼ�ݲ��ֵĸ���

	cv::Mat m_spot_img;//�ߵ�����ͼ
	cv::Mat m_acne_img;//���������ͼ
	cv::Mat m_blackhead_img;//��ͷ�����ͼ
	cv::Mat m_pore_img;//ë�׼����ͼ
	cv::Mat m_oill_img;//���Լ����ͼ
	cv::Mat m_wrinkle_img;//���Ƽ����ͼ
	cv::Mat m_texture_img;//��������ͼ

public:
	void createGrayImg();//�����Ҷ�ͼ
	void createCGrayImg();//�����Զ���Ҷ�ͼ
	void splitImg();//��ԭʼͼƬ������ͨ���������
	void createHSVImg();//����hsv��ɫ�ռ�ͼƬ
	void createSpotMask();//�������ڰߵ����mask
	void createPoreMask();//��������ë�׼���mask
	void createWrinkleMask();//�����������Ƽ���mask
	void createTextureMask();//���������������msk
	void createAllFaceMask();//����ȫ�����mask
	void createMarionetteMask();//ľż��
	void crateNasolabialMask();//������
	void createFrownMask();//������
	void createForeheadMask();//����
	void createEyeWrinkleMask();//������
	void createSkinTypeMask();//������������mask�����ԡ����ԡ����ԡ������
	cv::Scalar meanHSV(cv::Mat &hsv, cv::Mat &mask);//���hsv��ɫ�ռ�ͼƬ�ľ�ֵ

	/**
	* @brief ����LOG���Ӻ�
	* @param size �˴�С����Ϊ��������3��5��7
	* @param sigma ��˹�����ı�׼��
	*/
	cv::Mat createLOGKernel(int size, double sigma);

	void createLOGImg(cv::Mat &src, double sigma, cv::Mat &dst, int padding);//anew set input image?
	void sharpImg(cv::Mat &gray, cv::Mat &dst);
	void reduceHighLight(cv::Mat &src, cv::Mat &dst, float k);
	void increaseDark(cv::Mat &src, cv::Mat &dst, float gama);
	void acneAndSpotDetect(cv::Mat &gray_img);//�����Ͱߵ���
	void blackheadAndPoreDetect(cv::Mat &gray_img);//��ͷ��ë�׼��
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
	cv::Mat m_mask_marionette;//ľż��
	cv::Mat m_mask_nasolabial;//������
	cv::Mat m_mask_frown;//������
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



