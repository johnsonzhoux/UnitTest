#include "StdAfx.h"
#include "skin_analysis.h"
#include "image_process.h"
#include <iostream>
#include <string>
#include <sstream>

//default constructor
SkinAnalysis::SkinAnalysis() {
	m_src = cv::Mat(); 
	m_sigma_spot = 0;
	m_spot_binary_thresh = 0;
	m_spot_avg_thresh = 0;
	m_sigma_pore = 0;
	m_pore_thresh = 0;
	m_spot_binary_thresh = 0;
	m_spot_avg_thresh = 0;
	m_oill_thresh = 0;
	m_wrinkle_thresh = 0;
	m_acne_color_l = cv::Vec3b(0, 0, 0); 
	m_acne_color_h = cv::Vec3b(0, 0, 0); 
	m_blackhead_color_diff = 0.0;
	m_spot_area_l = 0; 
	m_spot_area_h = 0;
	m_acne_area_l = 0; 
	m_acne_area_h = 0; 
	m_blackhead_area_l = 0; 
	m_blackhead_area_h = 0;
	m_pts.clear();
	m_wrinkle_contours.clear();

	//init output value 
	m_spot_num = 0;
	m_acne_num = 0;
	m_blackhead_num = 0;
	m_pore_num = 0;
	m_oill_percent = 0.0;
	m_oill_center_percent = 0.0;
	m_oill_face_percent = 0.0;
	m_wrinkle_len = 0;
	m_marionette_len = 0;
	m_nasolabial_len = 0;
	m_frown_len = 0;
	m_forehead_len = 0;
	m_eye_wrinkle_len = 0;
	m_complexion_number = 0;
	m_blackeye_level = 0;
}

//constructor list
SkinAnalysis::SkinAnalysis(cv::Mat src, std::vector<cv::Point> pts, double sigma_spot, int spot_binary_thresh, int spot_avg_thresh,
		double sigma_pore, int pore_thresh, int oill_thresh, int wrinkle_thresh, cv::Vec3b acne_color_l, cv::Vec3b acne_color_h,
		double blackhead_color_diff, double spot_area_l, double spot_area_h, double acne_area_l, double acne_area_h, double blackhead_area_l, double blackhead_area_h)

{
	m_src = src;
	m_pts.assign(pts.begin(), pts.end());
	m_sigma_spot = sigma_spot;
	m_spot_binary_thresh = spot_binary_thresh;
	m_spot_avg_thresh = spot_avg_thresh;
	m_sigma_pore = sigma_pore;
	m_pore_thresh = pore_thresh;
	m_oill_thresh = oill_thresh;
	m_wrinkle_thresh = wrinkle_thresh;
	m_acne_color_l = acne_color_l;
	m_acne_color_h = acne_color_h;
	m_spot_area_l = spot_area_l;
	m_spot_area_h = spot_area_h;
	m_acne_area_l = acne_area_l;
	m_acne_area_h = acne_area_h;
	m_blackhead_area_l = blackhead_area_l;
	m_blackhead_area_h = blackhead_area_h;
	
}

//destructor
SkinAnalysis::~SkinAnalysis()
{

}

void SkinAnalysis::setInputFacePoints(std::vector<cv::Point> &pts)
{
	m_pts.assign(pts.begin(), pts.end());
}


//void SkinAnalysis::printParams()
//{
//	std::cout << "sigma_l = " << m_sigma_l << "\n"
//		<< "sigma_h = " << m_sigma_h << "\n"
//
//		<< "blackhead_thresh = " << m_logl_thresh << "\n"
//		<< "spot_thresh = " << m_logh_thresh << "\n"
//		<< "m_oill_thresh = " << m_oill_thresh << "\n"
//
//		<< "acne_color_l = " << m_acne_color_l << "\n"
//		<< "acne_color_h = " << m_acne_color_h << "\n"
//		<< "m_blackhead_color_diff = " << m_blackhead_color_diff << "\n"
//		
//		<< "spot area range = [ " << m_spot_area_l << ", " << m_spot_area_h << " ]\n"
//		<< "acne area range = [ " << m_acne_area_l << ", " << m_acne_area_h << " ]\n"
//		<< "blackhead area range = [ " << m_blackhead_area_l << ", " << m_blackhead_area_h << " ]"
//		<< std::endl;
//}

void SkinAnalysis::createGrayImg()
{
	cv::cvtColor(m_src, m_gray_img, CV_BGR2GRAY);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_gray_img.jpg", m_gray_img);
#endif
}

void SkinAnalysis::createCGrayImg()
{
	//std::vector<cv::Mat> channels;
	//cv::split(m_src, channels);
	//cv::Mat b = channels[0];
	//cv::Mat g = channels[1];
	//cv::Mat r = channels[2];
	//cv::Mat tmp;
	//cv::addWeighted(b, 1.0 / 3, g, 1.0 / 3, 0, tmp);
	//cv::addWeighted(tmp, 1.0, r, 1.0 / 3, 0, m_cgray_img);

	//usm sharp and convert to gray
	cv::Mat usm_img;
	dip::sharpUseUSM(m_src, usm_img, 150, 3);
	cv::cvtColor(usm_img, m_cgray_img, CV_BGR2GRAY);

#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "usm_img.jpg", usm_img);
	cv::imwrite(file_path + "m_cgray_img.jpg", m_cgray_img);
#endif
}

void SkinAnalysis::splitImg()
{
	std::vector<cv::Mat> channels;
	cv::split(m_src, channels);
	m_blue_img = channels[0];
	m_green_img = channels[1];
	m_red_img = channels[2];
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_blue_img.jpg", m_blue_img);
	cv::imwrite(file_path + "m_green_img.jpg", m_green_img);
	cv::imwrite(file_path + "m_red_img.jpg", m_red_img);
#endif
}

void SkinAnalysis::createHSVImg()
{
	cv::cvtColor(m_src, m_hsv_img, CV_BGR2HSV);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_hsv_img.jpg", m_hsv_img);
#endif
}

void SkinAnalysis::createSpotMask()
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
	pts[0] = (m_pts[21] + m_pts[22]) *0.5 - vy*0.6;
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
	pts[24] = m_pts[22] - vx*pec -vy*pec;
	pts[25] = m_pts[23] - vy*pec;
	pts[26] = m_pts[24] - vy*pec;
	pts[27] = m_pts[25] - vy*pec;
	pts[28] = m_pts[26] -vy*pec;
	pts[29] = m_pts[25] - vy*0.3;
	contours.push_back(pts);
	m_mask_as = cv::Mat::zeros(m_src.size(), CV_8UC1);
	cv::drawContours(m_mask_as, contours, 0, cv::Scalar(255), -1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_as.bmp", m_mask_as);
#endif
}


void SkinAnalysis::createPoreMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	float pec = 0.25;
	cv::Point pt;
	std::vector<cv::Point> pts(14);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = m_pts[28];
	pts[1] = m_pts[1] + vx*pec;
	pts[2] = m_pts[4] + vx*pec;
	pts[3] = m_pts[5] - vy*pec;
	pts[4] = cv::Point(m_pts[40].x, m_pts[31].y);
	pts[5] = cv::Point(m_pts[21].x, m_pts[29].y);
	pts[6] = cv::Point(m_pts[21].x, m_pts[30].y);
	pts[7] = m_pts[30] + vy*0.1;
	pts[8] = cv::Point(m_pts[22].x, m_pts[30].y);
	pts[9] = cv::Point(m_pts[22].x, m_pts[29].y);
	pts[10] = cv::Point(m_pts[47].x, m_pts[35].y);
	pts[11] = m_pts[11] - vy*pec;
	pts[12] = m_pts[12] - vx*pec;
	pts[13] = m_pts[15] - vx*pec;
	contours.push_back(pts);
	m_mask_bp = cv::Mat::zeros(m_src.size(), CV_8UC1);
	cv::drawContours(m_mask_bp, contours, 0, cv::Scalar(255), -1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_bp.bmp", m_mask_bp);
#endif
}

void SkinAnalysis::createWrinkleMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	float pec = 0.1;
	cv::Point pt;
	std::vector<cv::Point> pts(12);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = (m_pts[21] + m_pts[22]) *0.5 - vy*0.6;
	pts[1] = m_pts[19] - vy*0.3;
	pts[2] = m_pts[18] - vy*pec;
	pts[3] = m_pts[19] - vy*pec;
	pts[4] = m_pts[20] - vy*pec;
	pts[5] = m_pts[21] - vy*pec;
	pts[6] = m_pts[27];
	pts[7] = m_pts[22] - vy*pec;
	pts[8] = m_pts[23] - vy*pec;
	pts[9] = m_pts[24] - vy*pec;
	pts[10] = m_pts[25] - vy*pec;
	pts[11] = m_pts[24] - vy*0.3;
	contours.push_back(pts);

	pts.clear();
	pts.resize(4);
	pts[0] = m_pts[39] + vy*0.05;
	pts[1] = m_pts[1] + vx*pec;
	pts[2] = m_pts[3] + vx*pec;
	pts[3] = cv::Point(m_pts[21].x, m_pts[28].y);
	contours.push_back(pts);

	pts.clear();
	pts.resize(4);
	pts[0] = m_pts[42] + vy*0.05;
	pts[1] = m_pts[15] - vx*pec;
	pts[2] = m_pts[13] - vx*pec;
	pts[3] = cv::Point(m_pts[22].x, m_pts[28].y);
	contours.push_back(pts);

	m_mask_wrinkle = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); ++i) {
		cv::drawContours(m_mask_wrinkle, contours, i, cv::Scalar(255), -1);
	}
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_wrinkle.bmp", m_mask_wrinkle);
#endif
}


void SkinAnalysis::createTextureMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	float pec = 0.1;
	cv::Point pt;
	std::vector<cv::Point> pts(7);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = cv::Point(m_pts[39].x, m_pts[28].y);
	pts[1] = cv::Point(m_pts[39].x, m_pts[29].y);
	pts[2] = cv::Point(m_pts[40].x, m_pts[31].y);
	pts[3] = m_pts[48] - vy*pec -vx*pec;
	pts[4] = m_pts[4] + vx*pec;
	pts[5] = m_pts[3] + vx*pec;
	pts[6] = m_pts[2] + vx*pec;
	contours.push_back(pts);

	pts.clear();
	pts.resize(7);
	pts[0] = cv::Point(m_pts[42].x, m_pts[28].y);
	pts[1] = cv::Point(m_pts[42].x, m_pts[29].y);
	pts[2] = cv::Point(m_pts[47].x, m_pts[35].y);
	pts[3] = m_pts[54] - vy*pec + vx*pec;
	pts[4] = m_pts[12] - vx*pec;
	pts[5] = m_pts[13] - vx*pec;
	pts[6] = m_pts[14] - vx*pec;
	contours.push_back(pts);

	m_mask_texture = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); ++i) {
		cv::drawContours(m_mask_texture, contours, i, cv::Scalar(255), -1);
	}
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_texture.bmp", m_mask_texture);
#endif
}

void SkinAnalysis::createAllFaceMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	float pec = 0.1;
	cv::Point pt;
	std::vector<cv::Point> pts(44);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = (m_pts[21] + m_pts[22]) *0.5 - vy*0.6;
	pts[1] = m_pts[18] - vy*0.4;
	pts[2] = m_pts[17] - vy*pec;
	pts[3] = m_pts[18] - vy*pec;
	pts[4] = m_pts[19] - vy*pec;
	pts[5] = m_pts[20] - vy*pec;
	pts[6] = m_pts[21] - vy*pec + vx*pec;
	pts[7] = cv::Point(m_pts[21].x, m_pts[39].y) + vy*pec;
	pts[8] = m_pts[1] + vx*pec*2;
	pts[9] = m_pts[2] + vx*pec*2;
	pts[10] = m_pts[3] + vx*pec;
	pts[11] = m_pts[4] + vx*pec;
	pts[12] = m_pts[5] + vx*pec;
	pts[13] = m_pts[6] - vy*pec;
	pts[14] = m_pts[7]- vy*pec;
	pts[15] = m_pts[58] + vy*pec;
	pts[16] = m_pts[59] + vy*pec;
	pts[17] = m_pts[48] - vx*pec*3;

	pts[18] = cv::Point(m_pts[40].x, m_pts[31].y);
	pts[19] = cv::Point(m_pts[40].x, m_pts[30].y);
	pts[20] = cv::Point(m_pts[21].x, m_pts[29].y);
	pts[21] = cv::Point(m_pts[21].x, m_pts[30].y);
	pts[22] = m_pts[30] + vy*0.1;
	pts[23] = cv::Point(m_pts[22].x, m_pts[30].y);
	pts[24] = cv::Point(m_pts[22].x, m_pts[29].y);
	pts[25] = cv::Point(m_pts[47].x, m_pts[30].y);
	pts[26] = cv::Point(m_pts[47].x, m_pts[35].y);

	pts[27] = m_pts[54] + vx*pec*3;
	pts[28] = m_pts[55] + vy*pec;
	pts[29] = m_pts[56] + vy*pec;
	pts[30] = m_pts[9] - vy*pec;
	pts[31] = m_pts[10] - vy*pec;
	pts[32] = m_pts[11] - vx*pec;
	pts[33] = m_pts[12] - vx*pec;
	pts[34] = m_pts[13] - vx*pec;
	pts[35] = m_pts[14] - vx*pec*2;
	pts[36] = m_pts[15] - vx*pec*2;
	pts[37] = cv::Point(m_pts[22].x, m_pts[42].y) + vy*pec;
	pts[38] = m_pts[22] - vx*pec - vy*pec;
	pts[39] = m_pts[23] - vy*pec;
	pts[40] = m_pts[24] - vy*pec;
	pts[41] = m_pts[25] - vy*pec;
	pts[42] = m_pts[26] - vy*pec;
	pts[43] = m_pts[25] - vy*0.3;
	contours.push_back(pts);
	m_mask_all = cv::Mat::zeros(m_src.size(), CV_8UC1);
	cv::drawContours(m_mask_all, contours, 0, cv::Scalar(255), -1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_all.bmp", m_mask_all);
#endif
}


void SkinAnalysis::createMarionetteMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	cv::Point pt;
	std::vector<cv::Point> pts(5);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = cv::Point((m_pts[5].x + m_pts[6].x) / 2, (m_pts[33].y + m_pts[51].y) / 2);
	pts[1] = cv::Point(m_pts[49].x, (m_pts[33].y + m_pts[51].y) / 2);
	pts[2] = m_pts[48] - vx*0.01;
	pts[3] = cv::Point(m_pts[59].x, m_pts[57].y);
	pts[4] = cv::Point((m_pts[5].x + m_pts[6].x) / 2, m_pts[57].y);
	contours.push_back(pts);
	m_wrinkle_contours.push_back(pts);

	pts.clear();
	pts.resize(5);
	pts[0] = cv::Point(m_pts[53].x, (m_pts[33].y + m_pts[51].y) / 2);
	pts[1] = cv::Point((m_pts[10].x + m_pts[11].x) / 2, (m_pts[33].y + m_pts[51].y) / 2);
	pts[2] = cv::Point((m_pts[10].x + m_pts[11].x) / 2, m_pts[57].y);
	pts[3] = cv::Point(m_pts[55].x, m_pts[57].y);
	pts[4] = m_pts[54] + vx*0.01;
	contours.push_back(pts);
	m_wrinkle_contours.push_back(pts);

	m_mask_marionette = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); ++i) {
		cv::drawContours(m_mask_marionette, contours, i, cv::Scalar(255), -1);
	}
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_marionette.bmp", m_mask_marionette);
#endif
}

void SkinAnalysis::crateNasolabialMask()
{
	cv::Point pt;
	std::vector<cv::Point> pts(4);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = cv::Point(m_pts[39].x, m_pts[29].y);
	pts[1] = cv::Point(m_pts[32].x, m_pts[29].y);
	pts[2] = cv::Point(m_pts[48].x, (m_pts[33].y + m_pts[51].y) / 2);
	pts[3] = cv::Point((m_pts[5].x + m_pts[6].x) / 2, (m_pts[33].y + m_pts[51].y) / 2);
	contours.push_back(pts);
	m_wrinkle_contours.push_back(pts);

	pts.clear();
	pts.resize(4);
	pts[0] = cv::Point(m_pts[34].x, m_pts[29].y);
	pts[1] = cv::Point(m_pts[42].x, m_pts[29].y);
	pts[2] = cv::Point((m_pts[10].x + m_pts[11].x) / 2, (m_pts[33].y + m_pts[51].y) / 2);
	pts[3] = cv::Point(m_pts[54].x, (m_pts[33].y + m_pts[51].y) / 2);
	contours.push_back(pts);
	m_wrinkle_contours.push_back(pts);

	m_mask_nasolabial = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); ++i) {
		cv::drawContours(m_mask_nasolabial, contours, i, cv::Scalar(255), -1);
	}
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_nasolabial.bmp", m_mask_nasolabial);
#endif
}

void SkinAnalysis::createFrownMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	cv::Point pt;
	std::vector<cv::Point> pts(8);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = m_pts[20] - vy*0.08;
	pts[1] = m_pts[21] - vy*0.08;
	pts[2] = cv::Point(m_pts[21].x, m_pts[27].y);
	pts[3] = cv::Point(m_pts[22].x, m_pts[27].y);
	pts[4] = m_pts[22] - vy*0.08;
	pts[5] = m_pts[23] - vy*0.08;
	pts[6] = m_pts[23] - vy*0.15;//注意和额纹一致
	pts[7] = m_pts[20] - vy*0.15;//注意和额纹一致
	contours.push_back(pts);
	m_wrinkle_contours.push_back(pts);

	m_mask_frown = cv::Mat::zeros(m_src.size(), CV_8UC1);
	cv::drawContours(m_mask_frown, contours, 0, cv::Scalar(255), -1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_frown.bmp", m_mask_frown);
#endif
}



void SkinAnalysis::createForeheadMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	cv::Point pt;
	std::vector<cv::Point> pts(9);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = m_pts[17] - vy*0.2;
	pts[1] = m_pts[20] - vy*0.15;
	pts[2] = m_pts[23] - vy*0.15;
	pts[3] = m_pts[26] - vy*0.2;
	//pts[4] = m_pts[25] - vy*0.45;
	//pts[5] = m_pts[24] - vy*0.7;
	//pts[6] = cv::Point(m_pts[27].x, (m_pts[24].y + m_pts[19].y) / 2) - vy*0.75;
	//pts[7] = m_pts[19] - vy*0.7;
	//pts[8] = m_pts[18] - vy*0.45;
	pts[4] = m_pts[25] - vy*0.3;
	pts[5] = m_pts[24] - vy*0.45;
	pts[6] = cv::Point(m_pts[27].x, (m_pts[24].y + m_pts[19].y) / 2) - vy*0.5;
	pts[7] = m_pts[19] - vy*0.45;
	pts[8] = m_pts[18] - vy*0.3;
	contours.push_back(pts);
	m_wrinkle_contours.push_back(pts);

	m_mask_forehead = cv::Mat::zeros(m_src.size(), CV_8UC1);
	cv::drawContours(m_mask_forehead, contours, 0, cv::Scalar(255), -1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_frown.bmp", m_mask_forehead);
#endif

}


void SkinAnalysis::createEyeWrinkleMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	cv::Point pt;
	std::vector<cv::Point> pts(7);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = cv::Point(m_pts[21].x, m_pts[39].y);
	pts[1] = m_pts[40] + vy*0.08;
	pts[2] = m_pts[36] + vy*0.1;
	pts[3] = cv::Point(m_pts[17].x, m_pts[41].y);
	pts[4] = cv::Point(m_pts[17].x, m_pts[1].y);
	pts[5] = cv::Point(m_pts[41].x, (m_pts[1].y + m_pts[29].y) / 2);
	pts[6] = cv::Point(m_pts[39].x, m_pts[28].y)/* - vy*0.08*/;
	contours.push_back(pts);
	m_wrinkle_contours.push_back(pts);

	pts.clear();
	pts.resize(7);
	pts[0] = cv::Point(m_pts[22].x, m_pts[42].y);
	pts[1] = m_pts[47] + vy*0.08;
	pts[2] = m_pts[45] + vy*0.1;
	pts[3] = cv::Point(m_pts[26].x, m_pts[46].y);
	pts[4] = cv::Point(m_pts[26].x, m_pts[15].y);

	pts[5] = cv::Point(m_pts[46].x, (m_pts[15].y + m_pts[29].y) / 2);
	pts[6] = cv::Point(m_pts[42].x, m_pts[28].y)/* - vy*0.08*/;
	contours.push_back(pts);
	m_wrinkle_contours.push_back(pts);

	m_mask_eye_wrinkle = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); ++i) {
		cv::drawContours(m_mask_eye_wrinkle, contours, i, cv::Scalar(255), -1);
	}
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_eye_wrinkle.bmp", m_mask_eye_wrinkle);
#endif
}

void SkinAnalysis::createSkinTypeMask()
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	float pec = 0.1;
	cv::Point pt;
	std::vector<cv::Point> pts(16);
	std::vector<std::vector<cv::Point>> contours;
	pts[0] = (m_pts[21] + m_pts[22]) *0.5 - vy*0.6;
	pts[1] = m_pts[18] - vy*0.4;
	pts[2] = m_pts[17] - vy*pec;
	pts[3] = m_pts[18] - vy*pec;
	pts[4] = m_pts[19] - vy*pec;
	pts[5] = m_pts[20] - vy*pec;
	pts[6] = m_pts[21] - vy*pec + vx*pec;
	//pts[7] = cv::Point(m_pts[21].x, m_pts[39].y) + vy*pec;//
	//pts[8] = m_pts[1] + vx*pec;
	//pts[9] = m_pts[4] + vx*pec;
	//pts[10] = m_pts[5] - vy*pec;
	//pts[11] = cv::Point(m_pts[40].x, m_pts[31].y);
	//pts[12] = cv::Point(m_pts[40].x, m_pts[30].y);
	//pts[13] = cv::Point(m_pts[21].x, m_pts[29].y);
	pts[7] = cv::Point(m_pts[21].x, m_pts[30].y);
	pts[8] = m_pts[30] + vy*0.1;

	pts[9] = cv::Point(m_pts[22].x, m_pts[30].y);
	//pts[17] = cv::Point(m_pts[22].x, m_pts[29].y);
	//pts[18] = cv::Point(m_pts[47].x, m_pts[30].y);
	//pts[19] = cv::Point(m_pts[47].x, m_pts[35].y);
	//pts[20] = m_pts[11] - vy*pec;
	//pts[21] = m_pts[12] - vx*pec;
	//pts[22] = m_pts[15] - vx*pec;
	//pts[23] = cv::Point(m_pts[22].x, m_pts[42].y) + vy*pec;
	pts[10] = m_pts[22] - vx*pec - vy*pec;
	pts[11] = m_pts[23] - vy*pec;
	pts[12] = m_pts[24] - vy*pec;
	pts[13] = m_pts[25] - vy*pec;
	pts[14] = m_pts[26] - vy*pec;
	pts[15] = m_pts[25] - vy*0.3;
	contours.push_back(pts);

	pts.clear();
	pts.resize(6);
	pts[0] = m_pts[59] + vy*0.08;
	pts[1] = m_pts[57] + vy*0.08;
	pts[2] = m_pts[55] + vy*0.08;
	pts[3] = m_pts[10] - vy*0.08;
	pts[4] = m_pts[8] - vy*0.08;
	pts[5] = m_pts[6] - vy*0.08;
	contours.push_back(pts);

	m_mask_skintype_center = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); ++i) {
		cv::drawContours(m_mask_skintype_center, contours, i, cv::Scalar(255), -1);
	}

	//two face
	contours.clear();
	pts.clear();
	pts.resize(7);
	pts[0] = cv::Point(m_pts[21].x, m_pts[39].y) + vy*pec;//
	pts[1] = m_pts[1] + vx*pec;
	pts[2] = m_pts[4] + vx*pec;
	pts[3] = m_pts[5] - vy*pec;
	pts[4] = cv::Point(m_pts[40].x, m_pts[31].y);
	pts[5] = cv::Point(m_pts[40].x, m_pts[30].y);
	pts[6] = cv::Point(m_pts[21].x, m_pts[29].y);
	contours.push_back(pts);

	pts.clear();
	pts.resize(7);
	pts[0] = cv::Point(m_pts[22].x, m_pts[29].y);
	pts[1] = cv::Point(m_pts[47].x, m_pts[30].y);
	pts[2] = cv::Point(m_pts[47].x, m_pts[35].y);
	pts[3] = m_pts[11] - vy*pec;
	pts[4] = m_pts[12] - vx*pec;
	pts[5] = m_pts[15] - vx*pec;
	pts[6] = cv::Point(m_pts[22].x, m_pts[42].y) + vy*pec;
	contours.push_back(pts);

	m_mask_skintype_face = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); ++i) {
		cv::drawContours(m_mask_skintype_face, contours, i, cv::Scalar(255), -1);
	}

#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_skintype_center.bmp", m_mask_skintype_center);
	cv::imwrite(file_path + "m_mask_skintype_face.bmp", m_mask_skintype_face);
#endif
}




void SkinAnalysis::preprocess()
{
	createGrayImg();
	createCGrayImg();
	splitImg();
	createHSVImg();	
	createSpotMask();
	createPoreMask();
	//皱纹分区mask
	createMarionetteMask();
	crateNasolabialMask();
	createFrownMask();
	createForeheadMask();
	createEyeWrinkleMask();
	//createWrinkleMask();
	m_mask_wrinkle = cv::Mat::zeros(m_src.size(), CV_8UC1);
	m_mask_wrinkle += m_mask_marionette;
	m_mask_wrinkle += m_mask_nasolabial;
	m_mask_wrinkle += m_mask_frown;
	m_mask_wrinkle += m_mask_forehead;
	m_mask_wrinkle += m_mask_eye_wrinkle;
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_mask_wrinkle.bmp", m_mask_wrinkle);
#endif
	createSkinTypeMask();
	createTextureMask();
	createAllFaceMask();
	m_mask_as.copyTo(m_mask_oill);
	m_mask_bp.copyTo(m_mask_complexion);//m_mask_as.copyTo(m_mask_complexion);
}

cv::Scalar SkinAnalysis::meanHSV(cv::Mat &hsv, cv::Mat &mask)
{
	double sum_h = 0;
	double sum_s = 0;
	double sum_v = 0;
	int cnt = 0;
	cv::Scalar mean(0, 0, 0);
	for (int j = 0; j < hsv.rows; ++j) {
		cv::Vec3b *sptr = hsv.ptr<cv::Vec3b>(j);
		uchar *mptr = mask.ptr<uchar>(j);
		for (int i = 0; i < hsv.cols; ++i) {
			if (mptr[i]) {
				cnt++;
				double h = 0;
				double s = 0;
				double v = 0;
				//h = [0,180]
				//if (sptr[i][0] > 90) {
				if (sptr[i][0] > 156) {
					h = static_cast<double>(sptr[i][0]) - 181.0;
				}
				else {
					h = static_cast<double>(sptr[i][0]);
				}
				s = sptr[i][1];
				v = sptr[i][2];
				sum_h += h;
				sum_s += s;
				sum_v += v;
			}
		}
	}
	if (cnt == 0) {
		return mean;
	}
	else {
		sum_h = sum_h / cnt;
		sum_s = sum_s / cnt;
		sum_v = sum_v / cnt;
		if (sum_h < 0) {
			sum_h += 181.0;
		}
		mean = cv::Scalar(sum_h, sum_s, sum_v);
	}
	return mean;
}

//参考：https://blog.csdn.net/abcd1992719g/article/details/27071273
cv::Mat SkinAnalysis::createLOGKernel(int size, double sigma)
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



void SkinAnalysis::createLOGImg(cv::Mat &src, double sigma, cv::Mat &dst, int padding = 10)
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

void SkinAnalysis::sharpImg(cv::Mat &gray, cv::Mat &dst)
{
	//filter
	cv::Mat filter;
	cv::medianBlur(gray, filter, 3);
	//cv::blur(m_gray_img, filter, cv::Size(3, 3));//used for spots
//#ifdef SKIN_ANALYSIS_DEBUG
//	cv::imwrite(file_path + "filter.jpg", filter);
//#endif

	//enhance contrast
	cv::Mat tophat;
	cv::Mat blackhat;
	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)/*cv::Size(9, 9)*/);
	cv::morphologyEx(filter, tophat, cv::MORPH_TOPHAT, element);
	cv::morphologyEx(filter, blackhat, cv::MORPH_BLACKHAT, element);
	//dst = filter + tophat - blackhat;//enhance light and shadow
	dst = filter - blackhat;//enhance shadow
	//dst = blackhat;
//#ifdef SKIN_ANALYSIS_DEBUG
//	cv::imwrite(file_path + "sharp_img.jpg", dst);
//#endif
}


/*unify light*/
void SkinAnalysis::reduceHighLight(cv::Mat &gray, cv::Mat &dst, float k)
{
	//multiply
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

void SkinAnalysis::increaseDark(cv::Mat &gray, cv::Mat &dst, float gama)
{
	cv::Mat inv_gray;
	cv::subtract(cv::Scalar(255), gray, inv_gray);

	//multiply
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




/*acne and spot detection*/
void SkinAnalysis::acneAndSpotDetect(cv::Mat &gray_img)
{
	//spot and acne LOG image
	cv::Mat log_img;
	createLOGImg(gray_img, 6, log_img);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "as_0_log_img.jpg", log_img);
#endif
	cvNamedWindow("log", 0);
	imshow("log", log_img);

	//binary
	cv::Mat logh_binary;
	cv::threshold(log_img, logh_binary, 10, 255, CV_8UC1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "as_1_logh_binary.jpg", logh_binary);
#endif

	//spot acne img
	cv::Mat sa_img;
	logh_binary.copyTo(sa_img/*, m_mask_all*/);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "as_2_crop.jpg", sa_img);
#endif
	cvNamedWindow("logh_binary", 0);
	imshow("logh_binary", sa_img);
	cvWaitKey(0);
	//morphology
//	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
//	cv::morphologyEx(sa_img, sa_img, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
//	element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
//	cv::morphologyEx(sa_img, sa_img, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
//#ifdef SKIN_ANALYSIS_DEBUG
//	cv::imwrite(file_path + "as_3_crop_morp.jpg", sa_img);
//#endif
		
	//draw spots
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(sa_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	m_src.copyTo(m_acne_img);
	m_src.copyTo(m_spot_img);

	m_acne_num = 0;
	m_spot_num = 0;
	for (int j = 0; j < contours.size(); ++j) {
		cv::Rect min_rect = cv::boundingRect(cv::Mat(contours[j]));
		cv::Mat roi = log_img(min_rect);
		double avg = dip::meanMat(roi, m_spot_binary_thresh);
		if (avg >m_spot_avg_thresh) {
			cv::Mat hsv_roi(m_hsv_img, min_rect);
			cv::Mat mask_roi(m_mask_as, min_rect);
			cv::Scalar hsv_mean = meanHSV(hsv_roi, mask_roi);

			//area
			double area = cv::contourArea(contours[j], false);
			cv::Point2f center;
			float radius = 0;
			cv::minEnclosingCircle(cv::Mat(contours[j]), center, radius);
			float percentage = area / (CV_PI*radius*radius);

			//if is acne
			//if ((hsv_mean[0] < 15 || hsv_mean[0] > 160) && hsv_mean[1] > 50 && hsv_mean[2] > 150) {
			if ((hsv_mean[0] < m_acne_color_h[0] || hsv_mean[0] > m_acne_color_l[0]) && hsv_mean[1] > m_acne_color_l[1] && hsv_mean[2] > m_acne_color_l[2]
				&& percentage > 0.5) {
				//area constraint
				if (area >= m_acne_area_l && area < m_acne_area_h) {
					cv::circle(m_acne_img, center, radius + 2, cv::Scalar(0, 0, 255), 2);
					m_acne_num++;
				}
			}
			//spots
			else {
				//area constraint
				if (area >= m_spot_area_l && area < m_spot_area_h) {
					for (int i = 0; i < contours[j].size(); ++i) {
						//m_spot_img.at<cv::Vec3b>(contours[j][i].y, contours[j][i].x) = cv::Vec3b(245, 252, 44);
						cv::drawContours(m_spot_img, contours, j, cv::Scalar(245, 252, 44), 2);
					}
					m_spot_num++;
				}
			}
		}
		
	}
//#ifdef SKIN_ANALYSIS_DEBUG
//	cv::imwrite(file_path + "m_acne_img.jpg", m_acne_img);
//	cv::imwrite(file_path + "m_spot_img.jpg", m_spot_img);
//#endif
}

/*blackhead detection*/
void SkinAnalysis::blackheadAndPoreDetect(cv::Mat &gray_img)
{
	//unify lightness
	cv::Mat light;
	reduceHighLight(gray_img, light, 0.75);
	cv::Mat dark;
	increaseDark(gray_img, dark, 0.65);
	cv::Mat u_gray;
	cv::addWeighted(light, 0.5, dark, 0.5, 0, u_gray);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "bk_0_unify_light.jpg", u_gray);
#endif

	//sharp image
	cv::Mat sharp_img;
	sharpImg(u_gray, sharp_img);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "bk_1_sharp_img.jpg", sharp_img);
#endif

	//get blackhead LOG image
	cv::Mat log_l_img;
	createLOGImg(sharp_img, m_sigma_pore, log_l_img);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "bk_2_log_l_img.jpg", log_l_img);
#endif

	//binary
	cv::Mat logl_binary;
	cv::threshold(log_l_img, logl_binary, m_pore_thresh, 255, CV_8UC1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "bk_3_logl_binary.jpg", logl_binary);
#endif

	//crop
	cv::Mat bk_img;
	logl_binary.copyTo(bk_img, m_mask_bp);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "bk_4_crop.jpg", bk_img);
#endif

	//morphology
	//cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
	//cv::morphologyEx(bk_img, bk_img, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
	//cv::morphologyEx(bk_img, bk_img, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);

	//draw blackhead
	cv::Scalar bk_mean = cv::mean(u_gray, m_mask_bp);
	double bk_thresh = bk_mean[0] - m_blackhead_color_diff;//检测为黑头的阈值
	if (bk_thresh < 0) {
		bk_thresh = 0;
	}
	else if (bk_thresh > 255) {
		bk_thresh = 255;
	}
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(bk_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	m_src.copyTo(m_blackhead_img);
	m_src.copyTo(m_pore_img);
	m_blackhead_num = 0;
	m_pore_num = 0;
	for (int j = 0; j < contours.size(); ++j) {
		cv::Rect min_rect = cv::boundingRect(cv::Mat(contours[j]));
		cv::Mat gray_roi(u_gray, min_rect);
		cv::Mat mask_roi(m_mask_bp, min_rect);
		cv::Scalar roi_mean = cv::mean(gray_roi, mask_roi);

		double len = cv::arcLength(contours[j], true);
		double area = cv::contourArea(contours[j], false);
		cv::Point2f center;
		float radius = 0;
		cv::minEnclosingCircle(cv::Mat(contours[j]), center, radius);
		float percentage = area / (CV_PI*radius*radius);

		if ((len == 0 && area == 0)
			|| (len <= area+5) && len <= 10
			|| percentage > 0.4) {
			if (area >= m_blackhead_area_l && area < m_blackhead_area_h) {
				cv::circle(m_pore_img, center, radius + 2, cv::Scalar(150, 21, 62));
				m_pore_num++;
				if (roi_mean[0] < bk_thresh) {
					cv::circle(m_blackhead_img, center, radius + 2, cv::Scalar(150, 21, 62));
					m_blackhead_num++;
				}
			}
		}
	}
//#ifdef SKIN_ANALYSIS_DEBUG
//	cv::imwrite(file_path + "m_blackhead_img.jpg", m_blackhead_img);
//	cv::imwrite(file_path + "m_pore_img.jpg", m_pore_img);
//#endif
}

void SkinAnalysis::oillDetect(cv::Mat &gray_img)
{
	//filter
	cv::Mat filter;
	//cv::medianBlur(gray_img, filter, 3);
	cv::blur(m_gray_img, filter, cv::Size(3, 3));//used for spots
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "oill_0_filter.jpg", filter);
#endif

	//extract light region
	cv::Mat tophat;
	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)/*cv::Size(9, 9)*/);
	cv::morphologyEx(filter, tophat, cv::MORPH_TOPHAT, element);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "oill_1_hat.jpg", tophat);
#endif

	//binary
	cv::Mat binary;
	cv::threshold(tophat, binary, m_oill_thresh, 255, CV_8UC1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "oill_2_binary.jpg", binary);
#endif

#if 0
	//crop
	cv::Mat oill_img;
	binary.copyTo(oill_img, mask);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "oill_3_crop.jpg", oill_img);
#endif

	//morphology
//	element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
//	cv::morphologyEx(oill_img, oill_img, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
//#ifdef SKIN_ANALYSIS_DEBUG
//	cv::imwrite(file_path + "oill_img_morp.jpg", oill_img);
//#endif

	//draw spots
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(oill_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	m_src.copyTo(m_oill_img);
	double oill_area = 0;
	for (int j = 0; j < contours.size(); ++j) {
		double area = cv::contourArea(contours[j], false);
		for (int i = 0; i < contours[j].size(); ++i) {
			m_oill_img.at<cv::Vec3b>(contours[j][i].y, contours[j][i].x) = cv::Vec3b(0, 255, 0);
		}
		oill_area += area;
	}
	int sum = cv::countNonZero(mask);
	double percent = (oill_area / sum)*100.0;
#endif
	m_src.copyTo(m_oill_img);


	//crop
	cv::Mat oill_center_img;
	binary.copyTo(oill_center_img, m_mask_skintype_center);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "oill_3_crop_center.jpg", oill_center_img);
#endif

	//draw spots
	std::vector<std::vector<cv::Point>> center_contours;
	cv::findContours(oill_center_img, center_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	
	double oill_center_area = 0;
	for (int j = 0; j < center_contours.size(); ++j) {
		double area = cv::contourArea(center_contours[j], false);
		for (int i = 0; i < center_contours[j].size(); ++i) {
			m_oill_img.at<cv::Vec3b>(center_contours[j][i].y, center_contours[j][i].x) = cv::Vec3b(0, 255, 0);
		}
		oill_center_area += area;
	}
	int center_sum = cv::countNonZero(m_mask_skintype_center);
	m_oill_center_percent = (oill_center_area / center_sum)*100.0;

	//crop
	cv::Mat oill_face_img;
	binary.copyTo(oill_face_img, m_mask_skintype_face);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "oill_3_crop_face.jpg", oill_face_img);
#endif

	//draw spots
	std::vector<std::vector<cv::Point>> face_contours;
	cv::findContours(oill_face_img, face_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	double oill_face_area = 0;
	for (int j = 0; j < face_contours.size(); ++j) {
		double area = cv::contourArea(face_contours[j], false);
		for (int i = 0; i < face_contours[j].size(); ++i) {
			m_oill_img.at<cv::Vec3b>(face_contours[j][i].y, face_contours[j][i].x) = cv::Vec3b(0, 255, 0);
		}
		oill_face_area += area;
	}
	int face_sum = cv::countNonZero(m_mask_skintype_face);
	m_oill_face_percent = (oill_face_area / face_sum)*100.0;


//#ifdef SKIN_ANALYSIS_DEBUG
//	std::cout << "oill_area = " << oill_area << "/" << sum << ", percent = " << m_oill_percent << std::endl;
//	cv::imwrite(file_path + "m_oill_img.jpg", m_oill_img);
//#endif
}



/*Retinal vessel extraction by matched filter with first-order derivative of Gaussian(MF-FDOG)*/

/* hessian2D
* Dxx、Dxy、Dyy is double type
*/
void SkinAnalysis::hessian2D(cv::Mat &image, float sigma, cv::Mat &Dxx, cv::Mat &Dxy, cv::Mat &Dyy)
{
	//Make kernel coordinates
	cv::Mat X = cv::Mat::zeros(2 * 3 * sigma + 1, 2 * 3 * sigma + 1, CV_32F);//3*sigma must be less to max double
	for (int j = 0; j < X.rows; ++j) {
		float *ptr = X.ptr<float>(j);
		float v = -1 * 3 * sigma + j;
		for (int i = 0; i < X.cols; ++i) {
			ptr[i] = v;
		}
	}
	cv::Mat Y;
	Y = X.t();

	//Build the gaussian 2nd derivatives filters
	cv::Mat DGaussxx = cv::Mat::zeros(X.size(), X.type());
	cv::Mat DGaussxy = cv::Mat::zeros(X.size(), X.type());
	cv::Mat DGaussyy = cv::Mat::zeros(X.size(), X.type());

	//DGaussxx
	float c0 = 1 / (2 * CV_PI*(std::pow(sigma, 4)));
	cv::Mat X2;
	cv::pow(X, 2, X2);
	cv::Mat mat0;
	X2.convertTo(mat0, CV_32F, 1 / (sigma*sigma), -1);
	cv::Mat Y2;
	cv::pow(Y, 2, Y2);
	cv::Mat mat1;
	mat1 = X2 + Y2;
	cv::Mat mat2;
	mat1.convertTo(mat2, CV_32F, -1 / (2 * sigma*sigma));
	cv::Mat mat3;
	cv::exp(mat2, mat3);
	cv::Mat mat4;
	mat4 = mat0.mul(mat3);
	mat4.convertTo(DGaussxx, -1, c0);

	//DGaussxy
	float c1 = 1 / (2 * CV_PI*(std::pow(sigma, 6)));
	cv::Mat mat5;
	mat5 = X.mul(Y);
	cv::Mat mat6;
	mat6 = mat5.mul(mat3);
	mat6.convertTo(DGaussxy, -1, c1);

	//DGaussyy
	DGaussyy = DGaussxx.t();

	//filter
	cv::filter2D(image, Dxx, CV_32F, DGaussxx, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);//how to handle border
	cv::filter2D(image, Dxy, CV_32F, DGaussxy, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	cv::filter2D(image, Dyy, CV_32F, DGaussyy, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
}

cv::Mat SkinAnalysis::logical(cv::Mat &M)
{
	cv::Mat D = cv::Mat::zeros(M.size(), CV_32F);
	for (int j = 0; j < M.rows; ++j) {
		float *sptr = M.ptr<float>(j);
		float *dptr = D.ptr<float>(j);
		for (int i = 0; i < M.cols; ++i) {
			if (sptr[i] > 0) {
				dptr[i] = 1;
			}
		}
	}
	return D;
}

/* eig2image
* input: Dxx、Dxy、Dyy
* output:Lambda1、Lambda2、Ix、Iy
*
*/
void SkinAnalysis::eig2image(cv::Mat &Dxx, cv::Mat &Dxy, cv::Mat &Dyy, cv::Mat &Lambda1, cv::Mat &Lambda2, cv::Mat &Ix, cv::Mat &Iy)
{
	//Compute the eigenvectors of J, v1 and v2
	cv::Mat mat0;
	cv::pow(Dxx - Dyy, 2, mat0);
	cv::Mat mat1;
	cv::pow(Dxy, 2, mat1);
	cv::Mat mat2;
	mat2 = 4 * mat1;
	cv::Mat tmp;
	cv::sqrt(mat0 + mat2, tmp);

	cv::Mat v2x;
	cv::Mat v2y;
	v2x = 2 * Dxy;
	v2y = Dyy - Dxx + tmp;

	//Normalize
	cv::Mat mag;
	cv::Mat v2x2;
	cv::Mat v2y2;
	cv::pow(v2x, 2, v2x2);
	cv::pow(v2y, 2, v2y2);
	cv::sqrt(v2x2 + v2y2, mag);
	cv::divide(v2x, mag, v2x);//??
	cv::divide(v2y, mag, v2y);//??

	//The eigenvectors are orthogonal
	cv::Mat v1x;
	cv::Mat v1y;
	v1x = -1 * v2y;
	v1y = v2x;

	//Compute the eigenvalues
	cv::Mat mu1 = 0.5*(Dxx + Dyy + tmp);
	cv::Mat mu2 = 0.5*(Dxx + Dyy - tmp);

	//Sort eigen values by absolute value abs(Lambda1)<abs(Lambda2)
	cv::Mat check;
	cv::Mat mu1s2 = cv::abs(mu1) - cv::abs(mu2);
	check = logical(mu1s2);

	Lambda1 = mu1.mul(cv::Scalar(1.0) - check) + mu2.mul(check);
	Lambda2 = mu1.mul(check) + mu2.mul(cv::Scalar(1.0) - check);
	Ix = v1x.mul(cv::Scalar(1.0) - check) + v2x.mul(check);
	Iy = v1y.mul(check) + v2y.mul(cv::Scalar(1.0) - check);
}

std::vector<float> SkinAnalysis::createSigma(float start, float step, float end)
{
	std::vector<float> sigma;
	while (start <= end + 1e-8) {
		float s = start;// exp(start);
		sigma.push_back(s);
		start += step;
	}
	return sigma;
}


void SkinAnalysis::maxAmplitude(std::vector<cv::Mat> &imgs, cv::Mat &dst)
{
	imgs[0].copyTo(dst);
	for (int k = 1; k < imgs.size(); ++k) {
		for (int j = 0; j < imgs[k].rows; ++j) {
			float *sptr = imgs[k].ptr<float>(j);
			float *dptr = dst.ptr<float>(j);
			for (int i = 0; i < imgs[k].cols; ++i) {
				if (sptr[i] > dptr[i]) {
					dptr[i] = sptr[i];
				}
			}
		}
	}
}

void SkinAnalysis::frangiFilter2D(cv::Mat &I, cv::Mat &dst, std::vector<float> sigmas, float c0, float c1)
{
	float beta = 2 * c0*c0;
	float c = 2 * c1*c1;
	std::vector<cv::Mat> ALLfiltered;

	for (int k = 0; k < sigmas.size(); ++k) {
		float sigma = sigmas[k];
#ifdef SKIN_ANALYSIS_DEBUG
		std::cout << "sigma = " << sigma << std::endl;
#endif // SKIN_ANALYSIS_DEBUG

		//Make 2D hessian
		cv::Mat Dxx, Dxy, Dyy;
		hessian2D(I, sigma, Dxx, Dxy, Dyy);

		//Correct for scale
		Dxx = Dxx*sigma*sigma;
		Dxy = Dxy*sigma*sigma;
		Dyy = Dyy*sigma*sigma;

		cv::Mat Lambda1;
		cv::Mat Lambda2;
		cv::Mat Ix;
		cv::Mat Iy;
		eig2image(Dxx, Dxy, Dyy, Lambda2, Lambda1, Ix, Iy);

		//Compute some similarity measures
		for (int j = 0; j < Lambda1.rows; ++j) {
			float *ptr = Lambda1.ptr<float>(j);
			for (int i = 0; i < Lambda1.cols; ++i) {
				if (ptr[i] < EPS && ptr[i] > -1 * EPS) {
					ptr[i] = EPS;
				}
			}
		}
		cv::Mat Rb;
		cv::Mat div;
		cv::divide(Lambda2, Lambda1, div);
		cv::pow(div, 2, Rb);
		cv::Mat S2;
		S2 = Lambda1.mul(Lambda1) + Lambda2.mul(Lambda2);

		//Compute the output image
		cv::Mat m0;
		cv::exp((-1.0 / beta)*Rb, m0);
		cv::Mat m1;
		cv::exp((-1.0 / c)*S2, m1);
		cv::Mat m2;
		cv::subtract(cv::Scalar(1.0), m1, m2);

		cv::Mat Ifiltered;
		Ifiltered = m0.mul(m2);

		//% see pp. 45
		cv::Mat lz;
		lz = logical(Lambda1);
		Ifiltered = Ifiltered.mul(lz);

		ALLfiltered.push_back(Ifiltered);
	}
	maxAmplitude(ALLfiltered, dst);
#ifdef SKIN_ANALYSIS_DEBUG
	double min_val;
	double max_val;
	cv::Point min_loc;
	cv::Point max_loc;
	cv::minMaxLoc(dst, &min_val, &max_val, &min_loc, &max_loc);
	std::cout << "min = " << min_val << " max_val = " << max_val << std::endl;
#endif	
}

void SkinAnalysis::frangiFilter2D(cv::Mat &I, cv::Mat &dst, float sigma, float c0, float c1)
{
	float beta = 2 * c0*c0;
	float c = 2 * c1*c1;
	std::vector<cv::Mat> ALLfiltered;

#ifdef SKIN_ANALYSIS_DEBUG
		std::cout << "sigma = " << sigma << std::endl;
#endif // SKIN_ANALYSIS_DEBUG

	//Make 2D hessian
	cv::Mat Dxx, Dxy, Dyy;
	hessian2D(I, sigma, Dxx, Dxy, Dyy);

	//Correct for scale
	Dxx = Dxx*sigma*sigma;
	Dxy = Dxy*sigma*sigma;
	Dyy = Dyy*sigma*sigma;

	cv::Mat Lambda1;
	cv::Mat Lambda2;
	cv::Mat Ix;
	cv::Mat Iy;
	eig2image(Dxx, Dxy, Dyy, Lambda2, Lambda1, Ix, Iy);

	//Compute some similarity measures
	for (int j = 0; j < Lambda1.rows; ++j) {
		float *ptr = Lambda1.ptr<float>(j);
		for (int i = 0; i < Lambda1.cols; ++i) {
			if (ptr[i] < EPS && ptr[i] > -1 * EPS) {
				ptr[i] = EPS;
			}
		}
	}
	cv::Mat Rb;
	cv::Mat div;
	cv::divide(Lambda2, Lambda1, div);
	cv::pow(div, 2, Rb);
	cv::Mat S2;
	S2 = Lambda1.mul(Lambda1) + Lambda2.mul(Lambda2);

	//Compute the output image
	cv::Mat m0;
	cv::exp((-1.0 / beta)*Rb, m0);
	cv::Mat m1;
	cv::exp((-1.0 / c)*S2, m1);
	cv::Mat m2;
	cv::subtract(cv::Scalar(1.0), m1, m2);

	cv::Mat Ifiltered;
	Ifiltered = m0.mul(m2);

	//% see pp. 45
	cv::Mat lz;
	lz = logical(Lambda1);
	dst = Ifiltered.mul(lz);

#ifdef SKIN_ANALYSIS_DEBUG
	double min_val;
	double max_val;
	cv::Point min_loc;
	cv::Point max_loc;
	cv::minMaxLoc(dst, &min_val, &max_val, &min_loc, &max_loc);
	std::cout << "min = " << min_val << " max_val = " << max_val << std::endl;
#endif	
}




/**
* @brief 对输入图像进行细化,骨骼化
* @param src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
* @param maxIterations限制迭代次数，如果不进行限制，默认为-1，代表不限制迭代次数，直到获得最终结果
* @return 为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
*/
cv::Mat SkinAnalysis::thinImage(const cv::Mat & src, const int maxIterations)
{
	assert(src.type() == CV_8UC1);
	cv::Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //记录迭代次数  
	while (true) {
		count++;
		if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达  
			break;
		std::vector<uchar *> mFlag; //用于标记需要删除的点  
		//对点标记  
		for (int i = 0; i < height; ++i) {
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j) {
				//如果满足四个条件，进行标记  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6) {
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0) {
						//标记  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i) {
			**i = 0;
		}

		//直到没有点满足，算法结束  
		if (mFlag.empty()) {
			break;
		}
		else {
			mFlag.clear();//将mFlag清空  
		}

		//对点标记  
		for (int i = 0; i < height; ++i) {
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j) {
				//如果满足四个条件，进行标记  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6) {
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0) {
						//标记  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i) {
			**i = 0;
		}

		//直到没有点满足，算法结束  
		if (mFlag.empty()) {
			break;
		}
		else {
			mFlag.clear();//将mFlag清空  
		}
	}
	return dst;
}




#if 0//用一个mask进行皱纹检测
void SkinAnalysis::wrinkleDetect(cv::Mat &gray_img)
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	int roi_x = m_pts[0].x;
	int roi_y = (m_pts[21].y - h > 0) ? m_pts[21].y - h : 0;
	cv::Mat gray_roi = gray_img(cv::Rect(roi_x, roi_y, 5 * w, 3 * h));
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_0_roi.jpg", gray_roi);
#endif
	
	//filter
	cv::Mat filter;
	cv::medianBlur(gray_roi, filter, 5);
	cv::GaussianBlur(filter, filter, cv::Size(3, 3), 0);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_1_filter.jpg", filter);
#endif

	//sharp image
	cv::Mat sharp_img;
	cv::Mat tophat;
	cv::Mat blackhat;
	cv::Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7));
	cv::morphologyEx(filter, tophat, cv::MORPH_TOPHAT, element);
	cv::morphologyEx(filter, blackhat, cv::MORPH_BLACKHAT, element);
	sharp_img = filter + tophat - blackhat;
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_2_sharp.jpg", sharp_img);
#endif

	//frangi filter
	cv::Mat f_gray;
	sharp_img.convertTo(f_gray, CV_32FC1);
	cv::Mat f_frangi;
	//std::vector<double> sigmas = createSigma(1, 0.5, 2);
	//frangiFilter2D(f_gray, f_frangi, sigmas, 0.5, 15);
	std::vector<float> sigmas = createSigma(1, 1, 3);
	frangiFilter2D(f_gray, f_frangi, sigmas, 1, 15);
	cv::Mat frangi;
	f_frangi.convertTo(frangi, CV_8UC1, 255);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_3_frangi.jpg", frangi);
#endif

	//restore image
	cv::Mat frangi_full = cv::Mat(gray_img.size(), CV_8UC1);
	cv::Mat frangi_full_roi = frangi_full(cv::Rect(roi_x, roi_y, 5 * w, 3 * h));
	frangi.copyTo(frangi_full_roi);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_4_frangi_full.jpg", frangi_full);
#endif

	//binary
	cv::Mat binary;
	cv::threshold(frangi_full, binary, m_wrinkle_thresh, 255, CV_8UC1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_5_binary.jpg", binary);
#endif

	//crop
	cv::Mat crop;
	binary.copyTo(crop, m_mask_wrinkle);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_6_crop.jpg", crop);
#endif

	//morphology
	element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	//cv::morphologyEx(crop, crop, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
	cv::morphologyEx(crop, crop, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_7_crop_morp.jpg", crop);
#endif

	//filtration
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(crop, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cv::Mat filt = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int j = 0; j < contours.size(); ++j) {
		double len = cv::arcLength(contours[j], true);
		double area = cv::contourArea(contours[j], false);
		cv::Point2f center;
		float radius = 0;
		cv::minEnclosingCircle(cv::Mat(contours[j]), center, radius);
		float percentage = area / (CV_PI*radius*radius);
		if (len > 30 && area <= 15 || percentage < 0.3 && area > 15) {
			cv::drawContours(filt, contours, j, cv::Scalar(255), -1);
		}
	}
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_8_filt.jpg", filt);
#endif
	
	//thin image
	cv::threshold(filt, filt, 10, 1, cv::THRESH_BINARY);
	cv::Mat thin = thinImage(filt);
	thin = thin * 255;
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_9_thin.jpg", thin);
#endif
	
	//draw image
	m_wrinkle_len = cv::countNonZero(thin);
	m_src.copyTo(m_wrinkle_img);
	cv::Mat color = cv::Mat(m_src.size(), CV_8UC3, cv::Scalar(0, 255, 0));
	cv::Mat w_mask;
	thin.copyTo(w_mask);
	//element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	//cv::morphologyEx(w_mask, w_mask, cv::MORPH_DILATE, element, cv::Point(-1, -1), 1);
	color.copyTo(m_wrinkle_img, w_mask);
//#ifdef SKIN_ANALYSIS_DEBUG
//	cv::imwrite(file_path + "m_wrinkle_img.bmp", m_wrinkle_img);
//#endif

}
#endif

//分区域皱纹检测
void SkinAnalysis::wrinkleDetect(cv::Mat &gray_img)
{
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	int roi_x = m_pts[0].x;
	int roi_y = (m_pts[21].y - h > 0) ? m_pts[21].y - h : 0;
	//std::cout << "roi_x = " << roi_x << ", roi_y = " << roi_y << "w = " << 5*w << ", h = " << 3*h << std::endl;
	//cv::Mat gray_roi = gray_img(cv::Rect(roi_x, roi_y, 5 * w, 3 * h));
	int roi_w = roi_x + 5 * w - gray_img.cols > 0 ? gray_img.cols - (roi_x+1) : 5 * w;
	int roi_h = roi_y + 3 * h - gray_img.rows > 0 ? gray_img.rows - (roi_y + 1) : 3 * h;
	cv::Mat gray_roi = gray_img(cv::Rect(roi_x, roi_y, roi_w, roi_h));
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_0_roi.jpg", gray_roi);
#endif

	//filter
	cv::Mat filter;
	cv::medianBlur(gray_roi, filter, 5);
	cv::GaussianBlur(filter, filter, cv::Size(3, 3), 0);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_1_filter.jpg", filter);
#endif

	//sharp image
	cv::Mat sharp_img;
	cv::Mat tophat;
	cv::Mat blackhat;
	cv::Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7));
	cv::morphologyEx(filter, tophat, cv::MORPH_TOPHAT, element);
	cv::morphologyEx(filter, blackhat, cv::MORPH_BLACKHAT, element);
	sharp_img = filter + tophat - blackhat;
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_2_sharp.jpg", sharp_img);
#endif

	//frangi filter
	cv::Mat f_gray;
	sharp_img.convertTo(f_gray, CV_32FC1);
	//cv::Mat f_frangi;
	//std::vector<double> sigmas = createSigma(1, 0.5, 2);
	//frangiFilter2D(f_gray, f_frangi, sigmas, 0.5, 15);
	//std::vector<float> sigmas = createSigma(1, 1, 3);
	//frangiFilter2D(f_gray, f_frangi, sigmas, 1, 15);
	//std::vector<float> sigmas = createSigma(8, 7, 8);
	//frangiFilter2D(f_gray, f_frangi, sigmas, 1, 15);
	//cv::Mat frangi;
	//f_frangi.convertTo(frangi, CV_8UC1, 255);

	cv::Mat f_frangi_eye;
	cv::Mat f_frangi;
	float eye_sigma = 1.5;
	float other_sigma = 8;
	frangiFilter2D(f_gray, f_frangi_eye, eye_sigma, 10, 15);
	frangiFilter2D(f_gray, f_frangi, other_sigma, 1, 15);
	cv::Mat frangi_eye;
	cv::Mat frangi;
	f_frangi_eye.convertTo(frangi_eye, CV_8UC1, 255);
	f_frangi.convertTo(frangi, CV_8UC1, 255);
	cv::Mat eye_mask_roi = m_mask_eye_wrinkle(cv::Rect(roi_x, roi_y, roi_w, roi_h));
	frangi_eye.copyTo(frangi, eye_mask_roi);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_3_frangi.jpg", frangi);
#endif

	//restore image
	cv::Mat frangi_full = cv::Mat(gray_img.size(), CV_8UC1);
	cv::Mat frangi_full_roi = frangi_full(cv::Rect(roi_x, roi_y, roi_w, roi_h));
	frangi.copyTo(frangi_full_roi);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_4_frangi_full.jpg", frangi_full);
#endif

	//binary
	cv::Mat binary;
	cv::threshold(frangi_full, binary, m_wrinkle_thresh, 255, CV_8UC1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_5_binary.jpg", binary);
#endif

	//crop
	cv::Mat crop;
	binary.copyTo(crop, m_mask_wrinkle);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_6_crop.jpg", crop);
#endif

	//morphology
	element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	//cv::morphologyEx(crop, crop, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1);
	cv::morphologyEx(crop, crop, cv::MORPH_CLOSE, element, cv::Point(-1, -1), 1);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_7_crop_morp.jpg", crop);
#endif

	//filtration
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(crop, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	cv::Mat filt = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int j = 0; j < contours.size(); ++j) {
		double len = cv::arcLength(contours[j], true);
		double area = cv::contourArea(contours[j], false);
		cv::Point2f center;
		float radius = 0;
		cv::minEnclosingCircle(cv::Mat(contours[j]), center, radius);
		float percentage = area / (CV_PI*radius*radius);
		if (len > 30 && area <= 15 || percentage < 0.3 && area > 15) {
			cv::drawContours(filt, contours, j, cv::Scalar(255), -1);
		}
	}
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_8_filt.jpg", filt);
#endif

	//thin image
	cv::threshold(filt, filt, 10, 1, cv::THRESH_BINARY);
	cv::Mat thin = thinImage(filt);
	thin = thin * 255;
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "wrinkle_9_thin.jpg", thin);
#endif

	
	m_wrinkle_len = cv::countNonZero(thin);//总长度
	//wrinkle len
	cv::Mat tmp_mat;
	thin.copyTo(tmp_mat, m_mask_marionette);
	m_marionette_len = cv::countNonZero(tmp_mat);
	tmp_mat = cv::Mat::zeros(m_src.size(), CV_8UC1);
	thin.copyTo(tmp_mat, m_mask_nasolabial);
	m_nasolabial_len = cv::countNonZero(tmp_mat);
	tmp_mat = cv::Mat::zeros(m_src.size(), CV_8UC1);
	thin.copyTo(tmp_mat, m_mask_frown);
	m_frown_len = cv::countNonZero(tmp_mat);
	tmp_mat = cv::Mat::zeros(m_src.size(), CV_8UC1);
	thin.copyTo(tmp_mat, m_mask_forehead);
	m_forehead_len = cv::countNonZero(tmp_mat);
	tmp_mat = cv::Mat::zeros(m_src.size(), CV_8UC1);
	thin.copyTo(tmp_mat, m_mask_eye_wrinkle);
	m_eye_wrinkle_len = cv::countNonZero(tmp_mat);

	//draw image
	m_src.copyTo(m_wrinkle_img);

	cv::Mat color = cv::Mat(m_src.size(), CV_8UC3, cv::Scalar(0, 255, 0));
	cv::Mat w_mask;
	thin.copyTo(w_mask);
	//element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	//cv::morphologyEx(w_mask, w_mask, cv::MORPH_DILATE, element, cv::Point(-1, -1), 1);
	color.copyTo(m_wrinkle_img, w_mask);
	for (int i = 0; i < m_wrinkle_contours.size(); ++i) {
		cv::drawContours(m_wrinkle_img, m_wrinkle_contours, i, cv::Scalar(255), 1);
	}
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_wrinkle_img.bmp", m_wrinkle_img);
#endif
}

void SkinAnalysis::complexionClassify()
{
	//hsv mean value
	cv::Mat crop;
	m_hsv_img.copyTo(crop, m_mask_complexion);
	cv::Scalar mean = meanHSV(crop, m_mask_complexion);
#ifdef SKIN_ANALYSIS_DEBUG
	std::cout << "color = " << mean << std::endl;
#endif

	//classify
	float h = mean[0];
	float s = mean[1];
	float v = mean[2];
	if (s < 43) {
		m_complexion_number = COMPLEXION_WHITE_0;
	}
	else if (s >= 43 && s < 63) {
		m_complexion_number = COMPLEXION_WHITE_1;
	}
	else if (s >= 63 && s < 88 && v >= 215) {
		m_complexion_number = COMPLEXION_NATURE;
	}
	else if (s >= 63 && s < 88 && v < 215) {
		m_complexion_number = COMPLEXION_WHEAT;
	}
	else if (s >= 88 && s < 98) {
		m_complexion_number = COMPLEXION_DIM;
	}
	else {
		m_complexion_number = COMPLEXION_DARK;
	}
}

void SkinAnalysis::blackEyeRecognize()
{
	CV_Assert(m_pts.size() == 68);
	//three court five eyes
	int h = ((m_pts[7].y - m_pts[21].y) + (m_pts[9].y - m_pts[22].y)) / 4;
	int w = (m_pts[16].x - m_pts[0].x) / 5;
	cv::Point vy(0, h);
	cv::Point vx(w, 0);

	std::vector<cv::Point> pts;
	pts.assign(m_pts.begin(), m_pts.end());
	//modify
	pts[36].x -= w*0.3;
	pts[36].y += h*0.05;
	pts[41].y += h*0.06;
	pts[40].y += h*0.06;
	pts[39].y += h*0.05;

	pts[45].x += w*0.3;
	pts[45].y += h*0.05;
	pts[46].y += h*0.06;
	pts[47].y += h*0.06;
	pts[42].y += h*0.05;


	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Point> contour;
	cv::Point pt;
	//left eye
	pt = pts[36] + (pts[36] - pts[37])*0.3;
	contour.push_back(pt);
	pt = pts[36];
	pt.y += (pts[41].y - pts[36].y) * 2;
	contour.push_back(pt);
	pt = pts[40];
	pt.y += (pts[40].y - pts[38].y)*0.8;
	contour.push_back(pt);
	contour.push_back(pts[39]);
	contour.push_back(pts[40]);
	contour.push_back(pts[41]);
	contour.push_back(pts[36]);
	contours.push_back(contour);
	//right eye
	contour.clear();
	pt = pts[45] + (pts[45] - pts[44])*0.3;
	contour.push_back(pt);
	pt = pts[45];
	pt.y += (pts[46].y - pts[45].y) * 2;
	contour.push_back(pt);
	pt = pts[47];
	pt.y += (pts[47].y - pts[43].y)*0.8;
	contour.push_back(pt);
	contour.push_back(pts[42]);
	contour.push_back(pts[47]);
	contour.push_back(pts[46]);
	contour.push_back(pts[45]);
	contours.push_back(contour);
	//left face
	contour.clear();
	pt = pts[1];
	pt.x += (pts[31].x - pts[1].x)*0.2;
	contour.push_back(pt);
	pt = pts[41];
	pt.y += (pts[5].y - pts[41].y)*0.2;
	contour.push_back(pt);
	pt = pts[31];
	pt.x -= (pts[31].x - pts[1].x)*0.2;
	contour.push_back(pt);
	pt = pts[5];
	pt.y -= (pts[5].y - pts[41].y)*0.5;
	contour.push_back(pt);
	contours.push_back(contour);
	//right face
	contour.clear();
	pt = pts[35];
	pt.x += (pts[15].x - pts[35].x)*0.2;
	contour.push_back(pt);
	pt = pts[46];
	pt.y += (pts[11].y - pts[46].y)*0.2;
	contour.push_back(pt);
	pt = pts[15];
	pt.x -= (pts[15].x - pts[35].x)*0.2;
	contour.push_back(pt);
	pt = pts[11];
	pt.y -= (pts[11].y - pts[46].y)*0.5;
	contour.push_back(pt);
	contours.push_back(contour);

	//calculate mask
	cv::Mat eye_mask = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int i = 0; i < 2; ++i) {
		cv::drawContours(eye_mask, contours, i, cv::Scalar(255), -1);
	}
	cv::Mat face_mask = cv::Mat::zeros(m_src.size(), CV_8UC1);
	for (int i = 2; i < contours.size(); ++i) {
		cv::drawContours(face_mask, contours, i, cv::Scalar(255), -1);
	}
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "blackeye_eye_mask.jpg", eye_mask);
	cv::imwrite(file_path + "blackeye_face_mask.jpg", face_mask);
#endif

	cv::Scalar eye_mean = meanHSV(m_hsv_img, eye_mask);
	cv::Scalar face_mean = meanHSV(m_hsv_img, face_mask);
#ifdef SKIN_ANALYSIS_DEBUG
	std::cout << "eye_mean = " << eye_mean << std::endl;
	std::cout << "face_mean = " << face_mean << std::endl;
#endif
	m_blackeye_level = (face_mean[2] - eye_mean[2]) / face_mean[2];
}


void SkinAnalysis::emboss(cv::Mat &src, cv::Mat &dst, cv::Mat &refer, float ratio)
{
	CV_Assert(src.channels() == refer.channels());
	src.copyTo(dst);
	int chns = src.channels();
	int border = 1;
	int value = 0;
	for (int j = border; j < src.rows - border; ++j) {
		uchar *sptr0 = src.ptr<uchar>(j - 1);
		uchar *sptr = src.ptr<uchar>(j);
		uchar *sptr1 = src.ptr<uchar>(j+1);
		uchar *rptr = refer.ptr<uchar>(j);
		uchar *dptr = dst.ptr<uchar>(j);
		for (int i = border; i < src.cols - border; ++i) {
			for (int k = 0; k < chns; ++k) {
				//value = (sptr[chns*i + k] - sptr0[chns*(i + 1) + k])*ratio + rptr[chns*i + k];
				value = (sptr1[chns*(i-1) + k] - sptr0[chns*(i + 1) + k])*ratio + rptr[chns*i + k];
				//value = (sptr[chns*(i+1) + k] - sptr[chns*(i - 1) + k])*ratio + rptr[chns*i + k];
				//value = (sptr1[chns*i + k] - sptr0[chns*(i) + k])*ratio + rptr[chns*i + k];
				dptr[chns*i + k] = cv::saturate_cast<uchar>(value);
			}
		}
	}
}

void SkinAnalysis::textureDetect()
{
	cv::Mat refer;
	cv::blur(m_src, refer, cv::Size(95, 95));
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "texture_0_refer.jpg", refer);
#endif

	cv::Mat dia;
	emboss(m_src, dia, refer, 7.0);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "texture_1_dia.jpg", dia);
#endif

	cv::Mat filter;
	cv::GaussianBlur(dia, filter, cv::Size(5, 5), 0);
	//cv::blur(dia, filter, cv::Size(5, 5));
	//cv::medianBlur(dia, filter, 3);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "texture_2_filter.jpg", filter);
#endif

	cv::Mat refer_gray;
	cv::cvtColor(refer, refer_gray, CV_BGR2GRAY);
	cv::Mat filter_gray;
	cv::cvtColor(filter, filter_gray, CV_BGR2GRAY);

	/*bulge*/
	cv::Mat bulge;
	bulge = filter_gray - refer_gray;//extract light part
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "texture_3_bulge.jpg", bulge);
#endif

	cv::Mat bulge_filter;
	cv::blur(bulge, bulge_filter, cv::Size(5, 5));
	//cv::GaussianBlur(bulge, bulge_filter, cv::Size(5, 5), 0);
	cv::imwrite(file_path + "texture_4_filter.bmp", bulge_filter);

	cv::Mat bulge_binary;
	cv::threshold(bulge_filter, bulge_binary, 0, 255, CV_THRESH_OTSU);
	//cv::threshold(bulge_filter, bulge_binary, 30, 255, CV_THRESH_BINARY);
	cv::imwrite(file_path + "texture_5_binary.bmp", bulge_binary);

	cv::Mat bulge_morp;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(bulge_binary, bulge_morp, cv::MORPH_OPEN, element);
	cv::imwrite(file_path + "texture_6_morp.bmp", bulge_morp);

	cv::Mat bulge_crop;
	bulge_morp.copyTo(bulge_crop, m_mask_texture);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(bulge_crop, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	

	/*sunken*/
	cv::Mat sunken;
	sunken = refer_gray - filter_gray;//extract dark part
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "texture_s_0.jpg", sunken);
#endif

	cv::Mat sunken_filter;
	cv::blur(sunken, sunken_filter, cv::Size(5, 5));
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "texture_s_1_filter.bmp", sunken_filter);
#endif
	cv::Mat sunken_binary;
	cv::threshold(sunken_filter, sunken_binary, 0, 255, CV_THRESH_OTSU);
	//cv::threshold(sunken_filter, sunken_binary, 30, 255, CV_THRESH_BINARY);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "texture_s_2_binary.bmp", sunken_binary);
#endif

	cv::Mat sunken_morp;
	/*cv::Mat */element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(sunken_binary, sunken_morp, cv::MORPH_OPEN, element);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "texture_s_3_morp.bmp", sunken_morp);
#endif

	cv::Mat sunken_crop;
	sunken_morp.copyTo(sunken_crop, m_mask_texture);

	std::vector<std::vector<cv::Point>> contours1;
	cv::findContours(sunken_crop, contours1, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);


	/*draw*/
	m_src.copyTo(m_texture_img);
	filter.copyTo(m_texture_img, m_mask_texture);
#ifdef SKIN_ANALYSIS_DEBUG
	cv::imwrite(file_path + "m_texture_img_before.bmp", m_texture_img);
#endif

	m_texture_bulge_num = 0;
	m_texture_sunken_num = 0;
	for (int i = 0; i < contours.size(); ++i) {
		double area = cv::contourArea(cv::Mat(contours[i]), false);
		if (area > m_texture_thresh) {
			m_texture_bulge_num++;
			cv::drawContours(m_texture_img, contours, i, cv::Scalar(149, 253, 244), -1);
		}
	}
	for (int i = 0; i < contours1.size(); ++i) {
		double area = cv::contourArea(cv::Mat(contours1[i]), false);
		if (area > m_texture_thresh) {
			m_texture_sunken_num++;
			cv::drawContours(m_texture_img, contours1, i, cv::Scalar(162, 149, 37), -1);
		}
	}

//#ifdef SKIN_ANALYSIS_DEBUG
//	cv::imwrite(file_path + "m_texture_img.bmp", m_texture_img);
//#endif
}

void SkinAnalysis::analysis()
{
	acneAndSpotDetect(m_blue_img);
	blackheadAndPoreDetect(m_gray_img);
	oillDetect(m_gray_img);
	wrinkleDetect(m_cgray_img);
	complexionClassify();
	blackEyeRecognize();
	textureDetect();
}