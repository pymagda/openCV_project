#include "Backend/Include/CmPreProcess.h"
#include <list>
#include <queue>

#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)
typedef vector<double> vecD;
typedef vector<int> vecI;
using namespace cv;
using namespace std;
extern Point const DIRECTION8[9];

const double EPS = 1e-200;
typedef const Mat CMat;
#define ForPoints2(pnt, xS, yS, xE, yE) for (Point pnt(0, (yS)); pnt.y != (yE); pnt.y++) for (pnt.x = (xS); pnt.x != (xE); pnt.x++)

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

Rect CmPreProcess::GetMaskRange(CMat &mask1u, int ext, int thresh)
{
	int maxX = INT_MIN, maxY = INT_MIN, minX = INT_MAX, minY = INT_MAX, rows = mask1u.rows, cols = mask1u.cols;
	for (int r = 0; r < rows; r++)	{
		const unsigned char* data = mask1u.ptr<unsigned char>(r);
		for (int c = 0; c < cols; c++)
			if (data[c] > thresh) {
				maxX = max(maxX, c);
				minX = min(minX, c);
				maxY = max(maxY, r);
				minY = min(minY, r);
			}
	}

	maxX = maxX + ext + 1 < cols ? maxX + ext + 1 : cols;
	maxY = maxY + ext + 1 < rows ? maxY + ext + 1 : rows;
	minX = minX - ext > 0 ? minX - ext : 0;
	minY = minY - ext > 0 ? minY - ext : 0;

	return Rect(minX, minY, maxX - minX, maxY - minY);
}

int CmPreProcess::GetNZRegions(const Mat_<unsigned char> &label1u, Mat_<int> &regIdx1i, vecI &idxSum)
{
	vector<pair<int, int>> counterIdx;
	int _w = label1u.cols, _h = label1u.rows, maxIdx = -1;
	regIdx1i.create(label1u.size());
	regIdx1i = -1;

	for (int y = 0; y < _h; y++){		
		int *regIdx = regIdx1i.ptr<int>(y);
		const unsigned char  *label = label1u.ptr<unsigned char>(y);
		for (int x = 0; x < _w; x++) {
			if (regIdx[x] != -1 || label[x] == 0)
				continue;
			
			pair <int,int> counterReg = std::make_pair(0,++maxIdx);
			Point pt(x, y);
			queue<Point, list<Point>> neighbs;
			regIdx[x] = maxIdx;
			neighbs.push(pt);

			while(neighbs.size()){
				pt = neighbs.front();
				neighbs.pop();
				counterReg.first += label1u(pt);

				Point nPt(pt.x, pt.y - 1);
				if (nPt.y >= 0 && regIdx1i(nPt) == -1 && label1u(nPt) > 0){
					regIdx1i(nPt) = maxIdx;
					neighbs.push(nPt);  
				}

				nPt.y = pt.y + 1; // lower
				if (nPt.y < _h && regIdx1i(nPt) == -1 && label1u(nPt) > 0){
					regIdx1i(nPt) = maxIdx;
					neighbs.push(nPt);  
				}

				nPt.y = pt.y, nPt.x = pt.x - 1; // Left
				if (nPt.x >= 0 && regIdx1i(nPt) == -1 && label1u(nPt) > 0){
					regIdx1i(nPt) = maxIdx;
					neighbs.push(nPt);  
				}

				nPt.x = pt.x + 1;  // Right
				if (nPt.x < _w && regIdx1i(nPt) == -1 && label1u(nPt) > 0)	{
					regIdx1i(nPt) = maxIdx;
					neighbs.push(nPt);  
				}				
			}

			// Add current region to regions
			counterIdx.push_back(counterReg);
		}
	}
	sort(counterIdx.begin(), counterIdx.end(), greater<pair<int, int>>());
	int idxNum = (int)counterIdx.size();
	vector<int> newIdx(idxNum);
	idxSum.resize(idxNum);
	for (int i = 0; i < idxNum; i++){
		idxSum[i] = counterIdx[i].first;
		newIdx[counterIdx[i].second] = i;
	}
	
	for (int y = 0; y < _h; y++){
		int *regIdx = regIdx1i.ptr<int>(y);
		for (int x = 0; x < _w; x++)
			if (regIdx[x] >= 0)
				regIdx[x] = newIdx[regIdx[x]];
	}
	return idxNum;
}

Mat CmPreProcess::GetNZRegionsLS(CMat &mask1u, double ignoreRatio)
{
	CV_Assert(mask1u.type() == CV_8UC1 && mask1u.data != NULL);
	ignoreRatio *= mask1u.rows * mask1u.cols * 255;
	Mat_<int> regIdx1i;
	vecI idxSum;
	Mat resMask;
	CmPreProcess::GetNZRegions(mask1u, regIdx1i, idxSum);
	if (idxSum.size() >= 1 && idxSum[0] > ignoreRatio)
		compare(regIdx1i, 0, resMask, CMP_EQ);
	return resMask;
}
