#ifndef CMSALCUT_H_
#define CMSALCUT_H_

#include "Backend/Vendor/graph.h"
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <string>
#include "CmGaussianMModel.h"

using namespace cv;
using namespace std;

typedef const Mat CMat;

class CmCut
{
public:
	enum TrimapValue {TrimapBackground = 0, TrimapUnknown = 128, TrimapForeground = 255};

	CmCut(CMat &img3f);
	~CmCut(void);

	static Mat CutObjs(CMat &img3f, CMat &sal1f, float t1 = 0.2f, float t2 = 0.9f,
		CMat &borderMask = Mat(), int wkSize = 20);

	static void Demo(std::string& image_path, std::string& result_path);

public:
	void initialize(CMat &sal1f, float t1, float t2);
	void initialize(CMat &sal1u); // Background = 0, unknown = 128, foreground = 255
	void fitGMMs();
	int refineOnce();
	void drawResult(Mat& maskForeGround) {compare(_segVal1f, 0.5, maskForeGround, CMP_GT);}

private:
	int updateHardSegmentation();
	void initGraph();
	static int ExpandMask(CMat &fMask, Mat &mask1u, CMat &bdReg1u, int expandRatio = 5);

private:
	int _w, _h;
	Mat _imgBGR3f, _imgLab3f;
	Mat _trimap1i;
	Mat _segVal1f;

	float _lambda;
	float _beta;
	float _L;
	GraphF *_graph;
	Mat_<Vec4f> _NLinks;
	int _directions[4];

	CmGaussianMModel _bGMM, _fGMM;
	Mat _bGMMidx1i, _fGMMidx1i;
};


#endif /* CMSALCUT2_H_ */
