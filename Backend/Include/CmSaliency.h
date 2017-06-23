#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;
typedef const Mat CMat;
typedef pair<float, int> CostfIdx;

struct CmSaliency
{
	static Mat GetRC(CMat &img3f);
	static Mat GetRC(CMat &img3f, CMat &idx1i, int regNum, double sigmaDist = 0.4);
	static Mat GetRC(CMat &img3f, double sigmaDist, double segK, int segMinSize, double segSigma);
	static void SmoothByHist(CMat &img3f, Mat &sal1f, float delta);
	static void SmoothByRegion(Mat &sal1f, CMat &idx1i, int regNum, bool bNormalize = true);

private:
	static void SmoothSaliency(CMat &colorNum1i, Mat &sal1f, float delta, const vector<vector<CostfIdx>> &similar);

	struct Region{
		Region() { pixNum = 0; ad2c = Point2d(0, 0);}
		int pixNum;
		vector<CostfIdx> freIdx;
		Point2d centroid;
		Point2d ad2c;
	};
	static void BuildRegions(CMat& regIdx1i, vector<Region> &regs, CMat &colorIdx1i, int colorNum);
	static void RegionContrast(const vector<Region> &regs, CMat &color3fv, Mat& regSal1d, double sigmaDist);
	static int Quantize(CMat& img3f, Mat &idx1i, Mat &_color3f, Mat &_colorNum, double ratio = 0.95, const int colorNums[3] = DefaultNums);
	static const int DefaultNums[3];

	static Mat GetBorderReg(CMat &idx1i, int regNum, double ratio = 0.02, double thr = 0.3);
};
