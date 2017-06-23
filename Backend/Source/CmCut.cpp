#include "Backend/Include/CmCut.h"
#include "Backend/Include/CmPreProcess.h"
#include "Backend/Include/CmSaliency.h"

#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)

using namespace cv;
using namespace std;

typedef const Mat CMat;
typedef unsigned char byte;

Point const DIRECTION8[9] = {
        Point(1,  0),
        Point(1,  1),
        Point(0,  1),
        Point(-1, 1),
        Point(-1, 0),
        Point(-1,-1),
        Point(0, -1),
        Point(1, -1),
        Point(0, 0),
};

double const SQRT2 = sqrt(2.0);
float vecSqrDist(const Vec3f &v1, const Vec3f &v2)
{	float s = 0;

	for (int i=0; i<3; i++)
	  s += sqrt(v1[i] - v2[i]);
     return s;
}

CmCut::CmCut(CMat &img3f)
	:_fGMM(5), _bGMM(5), _w(img3f.cols), _h(img3f.rows), _lambda(50)
{
	CV_Assert(img3f.data != NULL && img3f.type() == CV_32FC3);
	_imgBGR3f = img3f;
	cvtColor(_imgBGR3f, _imgLab3f, CV_BGR2Lab);
	_trimap1i = Mat::zeros(_h, _w, CV_32S);
	_segVal1f = Mat::zeros(_h, _w, CV_32F);
	_graph = NULL;

	_L = 8 * _lambda + 1;
	_beta = 0; {
		int edges = 0;
		double result = 0;
		for (int y = 0; y < _h; ++y) {
			const Vec3f* img = _imgLab3f.ptr<Vec3f>(y);
			for (int x = 0; x < _w; ++x){
				Point pnt(x, y);
				for (int i = 0; i < 4; i++)	{
					Point pntN = pnt + DIRECTION8[i];
					if (CHK_IND(pntN))
						result += vecSqrDist(_imgLab3f.at<Vec3f>(pntN), img[x]), edges++;
				}
			}
		}
		_beta = (float)(0.5 * edges/result);
	}
	_NLinks.create(_h, _w); {
		static const float dW[4] = {1, (float)(1/SQRT2), 1, (float)(1/SQRT2)};
		for (int y = 0; y < _h; y++) {
			Vec4f *nLink = _NLinks.ptr<Vec4f>(y);
			const Vec3f* img = _imgLab3f.ptr<Vec3f>(y);
			for (int x = 0; x < _w; x++, nLink++) {
				Point pnt(x, y);
				const Vec3f& c1 = img[x];
				for (int i = 0; i < 4; i++)	{
					Point pntN = pnt + DIRECTION8[i];
					if (CHK_IND(pntN))
						(*nLink)[i] = _lambda * dW[i] * exp(-_beta * vecSqrDist(_imgLab3f.at<Vec3f>(pntN), c1));
				}
			}
		}
	}

	for (int i = 0; i < 4; i++)
		_directions[i] = DIRECTION8[i].x + DIRECTION8[i].y * _w;
}

CmCut::~CmCut(void)
{
	if (_graph)
		delete _graph;
}

Mat CmCut::CutObjs(CMat &_img3f, CMat &_sal1f, float t1, float t2, CMat &_border1u, int wkSize)
{
	Mat border1u = _border1u;
	if (border1u.data == NULL || border1u.size != _img3f.size){
		int bW = cvRound(0.02 * _img3f.cols), bH = cvRound(0.02 * _img3f.rows);
		border1u.create(_img3f.rows, _img3f.cols, CV_8U);
		border1u = 255;
		border1u(Rect(bW, bH, _img3f.cols - 2*bW, _img3f.rows - 2*bH)) = (int)0;
	}
	Mat sal1f, wkMask;
	_sal1f.copyTo(sal1f);
	sal1f.setTo(0, border1u);

	cv::Rect rect(0, 0, _img3f.cols, _img3f.rows);
	if (wkSize > 0){
		threshold(sal1f, sal1f, t1, 1, THRESH_TOZERO);
		sal1f.convertTo(wkMask, CV_8U, 255);
		threshold(wkMask, wkMask, 70, 255, THRESH_TOZERO);
		wkMask = CmPreProcess::GetNZRegionsLS(wkMask, 0.005);
		if (wkMask.data == NULL)
			return Mat();
		rect = CmPreProcess::GetMaskRange(wkMask, wkSize);
		sal1f = sal1f(rect);
		border1u = border1u(rect);
		wkMask = wkMask(rect);
	}
	CMat img3f = _img3f(rect);

	Mat fMask;
	CmCut salCut(img3f);
	salCut.initialize(sal1f, t1, t2);
	const int outerIter = 4;
	for (int j = 0; j < outerIter; j++)	{
		salCut.fitGMMs();
		int changed = 1000, times = 8;
		while (changed > 50 && times--) {
			changed = salCut.refineOnce();
		}
		salCut.drawResult(fMask);

		fMask = CmPreProcess::GetNZRegionsLS(fMask);
		if (fMask.data == NULL)
			return Mat();

		if (j == outerIter - 1 || ExpandMask(fMask, wkMask, border1u, 5) < 10)
			break;

		salCut.initialize(wkMask);
		fMask.copyTo(wkMask);
	}

	Mat resMask = Mat::zeros(_img3f.size(), CV_8U);
	fMask.copyTo(resMask(rect));
	return resMask;
}

void CmCut::initialize(CMat &sal1f, float t1, float t2)
{
	CV_Assert(sal1f.type() == CV_32F && sal1f.size == _imgBGR3f.size);
	sal1f.copyTo(_segVal1f);

	for (int y = 0; y < _h; y++) {
		int* triVal = _trimap1i.ptr<int>(y);
		const float *segVal = _segVal1f.ptr<float>(y);
		for (int x = 0; x < _w; x++) {
			triVal[x] = segVal[x] < t1 ? TrimapBackground : TrimapUnknown;
			triVal[x] = segVal[x] > t2 ? TrimapForeground : triVal[x];
		}
	}
}

void CmCut::initialize(CMat &sal1u) // Background = 0, unknown = 128, foreground = 255
{
	CV_Assert(sal1u.type() == CV_8UC1 && sal1u.size == _imgBGR3f.size);
	for (int y = 0; y < _h; y++) {
		int* triVal = _trimap1i.ptr<int>(y);
		const unsigned char *salVal = sal1u.ptr<unsigned char>(y);
		float *segVal = _segVal1f.ptr<float>(y);
		for (int x = 0; x < _w; x++) {
			triVal[x] = salVal[x] < 70 ? TrimapBackground : TrimapUnknown;
			triVal[x] = salVal[x] > 200 ? TrimapForeground : triVal[x];
			segVal[x] = salVal[x] < 70 ? 0 : 1.f;
		}
	}
}


void CmCut::fitGMMs()
{
	_fGMM.BuildGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);
	Mat _segVal1f2 = 1-_segVal1f;
	_bGMM.BuildGMMs(_imgBGR3f, _bGMMidx1i, _segVal1f2);
}

int CmCut::refineOnce()
{
	if (_fGMM.GetSumWeight() < 50 || _bGMM.GetSumWeight() < 50)
		return 0;

	_fGMM.RefineGMMs(_imgBGR3f, _fGMMidx1i, _segVal1f);

	_bGMM.BuildGMMs(_imgBGR3f, _bGMMidx1i, (Mat)(1 - _segVal1f));

	initGraph();
	if (_graph)
		_graph->maxflow();

	return updateHardSegmentation();
}

int CmCut::updateHardSegmentation()
{
	int changed = 0;
	for (int y = 0, id = 0; y < _h; ++y) {
		float* segVal = _segVal1f.ptr<float>(y);
		int* triMapD = _trimap1i.ptr<int>(y);
		for (int x = 0; x < _w; ++x, id++) {
			float oldValue = segVal[x];
			if (triMapD[x] == TrimapBackground)
				segVal[x] = 0.f;
			else if (triMapD[x] == TrimapForeground)
				segVal[x] = 1.f;
			else
				segVal[x] = _graph->what_segment(id) == GraphF::SOURCE ? 1.f : 0.f;
			changed += abs(segVal[x] - oldValue) > 0.1 ? 1 : 0;
		}
	}
	return changed;
}

void CmCut::initGraph()
{
	if (_graph == NULL)
		_graph = new GraphF(_w * _h, 4 * _w * _h);
	else
		_graph->reset();
	_graph->add_node(_w * _h);

	for (int y = 0, id = 0; y < _h; ++y) {
		int* triMapD = _trimap1i.ptr<int>(y);
		const float* img = _imgBGR3f.ptr<float>(y);
		for(int x = 0; x < _w; x++, img += 3, id++) {
			float back, fore;
			if (triMapD[x] == TrimapUnknown ) {
				fore = -log(_bGMM.P(img));
				back = -log(_fGMM.P(img));
			}
			else if (triMapD[x] == TrimapBackground )
				fore = 0, back = _L;
			else
				fore = _L,	back = 0;
			_graph->add_tweights(id, fore, back);

			Point pnt(x, y);
			const Vec4f& nLink = _NLinks(pnt);
			for (int i = 0; i < 4; i++)	{
				Point nPnt = pnt + DIRECTION8[i];
				if (CHK_IND(nPnt)) break;
					_graph->add_edge(id, id + _directions[i], nLink[i], nLink[i]);
			}
		}
	}
}



int CmCut::ExpandMask(CMat &fMask, Mat &mask1u, CMat &bdReg1u, int expandRatio)
{
	compare(fMask, mask1u, mask1u, CMP_NE);
	int changed = cvRound(sum(mask1u).val[0] / 255.0);

	Mat bigM, smalM;
	dilate(fMask, bigM, Mat(), Point(-1, -1), expandRatio);
	erode(fMask, smalM, Mat(), Point(-1, -1), expandRatio);
	static const double erodeSmall = 255 * 50;
	if (sum(smalM).val[0] < erodeSmall)
		smalM = fMask;
	mask1u = bigM * 0.5 + smalM * 0.5;
	mask1u.setTo(0, bdReg1u);
	return changed;
}

void CmCut::Demo(std::string& image_path, std::string& result_path)
{

    Mat img3f = imread(image_path);
    Mat sal;
    img3f.convertTo(img3f, CV_32FC3, 1.0/255);
    sal = CmSaliency::GetRC(img3f);
    Mat cutMat;
    float t = 0.9f;
    int maxIt = 4;
    GaussianBlur(sal, sal, Size(9, 9), 0);
    normalize(sal, sal, 0, 1, NORM_MINMAX);
    while (cutMat.empty() && maxIt--){
        cutMat = CmCut::CutObjs(img3f, sal, 0.1f, t);
        t -= 0.2f;
    }
    vector<int> compression_params;
    compression_params.push_back((int)CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(9);

    result_path = "/Users/magdalenapeksa/Dev/openCV_project/build/krasnal2.png";

    try {
        imwrite(result_path, cutMat, compression_params);
    } catch (runtime_error& ex) {
    }

}
