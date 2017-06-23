#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace cv;

typedef vector<int> vecI;
typedef const Mat CMat;

struct CmPreProcess {
	static Rect GetMaskRange(CMat &mask1u, int ext = 0, int thresh = 10);
	static int GetNZRegions(const Mat_<unsigned char> &label1u, Mat_<int> &regIdx1i, vecI &idxSum);
	static Mat GetNZRegionsLS(CMat &mask1u, double ignoreRatio = 0.02);
};




