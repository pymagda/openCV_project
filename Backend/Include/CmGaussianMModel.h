#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;

struct CmGaussian
{
    double mean[3];
    double covar[3][3];
    double det;
    double inv[3][3];
    double w;
    double eValues[3];
    double eVectors[3][3];
};

class CmGaussianMModel
{
public:
    typedef Vec<float, 3> Sample;
    CmGaussianMModel(int K, double thrV = 0.01);
    ~CmGaussianMModel(void);
    float P(const float c[3]) const ;
    double P(int i, const float c[3]) const ;
    void BuildGMMs(const Mat& sampleDf, Mat& component1i, const Mat& w1f = Mat());
    int RefineGMMs(const Mat& sampleDf, Mat& components1i, const Mat& w1f = Mat(), bool needReAssign = true);

    double GetSumWeight() const {return _sumW;}

protected:
    int _K, _MaxK;
    double _sumW;
    double _ThrV;
    CmGaussian* _Guassians;

    void AssignEachPixel(const Mat& sampleDf, Mat &component1i);
};

class CmGaussianFitter
{
public:
    CmGaussianFitter() {Reset();}
    inline void Add(const float* _c) ;
    inline void Add(const float* _c, float _weight) ;
    void Reset() {memset(this, 0, sizeof(CmGaussianFitter));}
    void BuildGuassian(CmGaussian& g, double totalCount, bool computeEigens = false) const;
    inline double Count(){return count;}
private:
    double s[3];
    double p[3][3] ;
    double count;
};

void CmGaussianFitter::Add(const float* _c)
{
    double c[3];
    for (int i = 0;  i < 3; i++)
        c[i] = _c[i];

    for (int i = 0; i < 3; i++) {
        s[i] += c[i];
        for (int j = 0; j < 3; j++)
            p[i][j] += c[i] * c[j];
    }
    count++;
}

void CmGaussianFitter::Add(const float* _c, float _weight)
{
    double c[3];
    for (int i = 0;  i < 3; i++)
        c[i] = _c[i];
    double weight = _weight;

    for (int i = 0; i < 3; i++) {
        s[i] += c[i] * weight;
        for (int j = 0; j < 3; j++)
            p[i][j] += c[i] * c[j] * weight;
    }
    count += weight;
}
inline void CmGaussianFitter::BuildGuassian(CmGaussian& g, double totalCount, bool computeEigens) const
{
    const double Epsilon = 1e-7/(3*3);

    if (count < Epsilon)
        g.w = 0;
    else {
        for (int i = 0; i < 3; i++)
            g.mean[i] = s[i]/count;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++)
                g.covar[i][j] = p[i][j]/count - g.mean[i] * g.mean[j];
            g.covar[i][i] += Epsilon;
        }
        Mat covar(3, 3, CV_64FC1, g.covar);
        Mat inv(3, 3, CV_64FC1, g.inv);
        invert(covar, inv, CV_LU);
        g.det = determinant(covar);
        g.w = count/totalCount;

        if (computeEigens)  {
            Mat eVals(3, 1, CV_64FC1, g.eValues);
            Mat eVecs(3, 1, CV_64FC1, g.eVectors);
            Matx<double, 3, 3> tmp;
            SVD::compute(covar, eVals, eVecs, tmp);
        }
    }
}
inline CmGaussianMModel::CmGaussianMModel(int K, double thrV)
        : _K(K), _ThrV(thrV), _MaxK(K)
{
    _Guassians = new CmGaussian[_K];
}

inline CmGaussianMModel::~CmGaussianMModel(void)
{
    if (_Guassians)
        delete []_Guassians;
}

inline float CmGaussianMModel::P(const float c[3]) const
{
    double r = 0;
    if (_Guassians)
        for (int i = 0; i < _K; i++)
            r += _Guassians[i].w * P(i, c);
    return (float)r;
}

inline double CmGaussianMModel::P(int i, const float c[3]) const
{
    double result = 0;
    CmGaussian& guassian = _Guassians[i];
    if (guassian.w > 0) {
        double v[3];
        for (int t = 0; t < 3; t++)
            v[t] = c[t] - guassian.mean[t];

        if (guassian.det > 0)   {
            double (&inv)[3][3] = guassian.inv;
            double d = 0;
            for(int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    d += v[i] * inv[i][j] * v[j];
            result = (double)(0.0635 / sqrt(guassian.det) * exp(-0.5f * d));   // 1/(2*pi)^1.5 = 0.0635
        }
        else {
            if (guassian.w < 1e-3)
                return 0;
        }
    }
    return result;
}

inline void CmGaussianMModel::BuildGMMs(const Mat& sampleDf, Mat& component1i, const Mat& w1f)
{
    bool weighted = w1f.data != NULL;
    int rows = sampleDf.rows, cols = sampleDf.cols;
    component1i = Mat::zeros(sampleDf.size(), CV_32S);{
        if (sampleDf.isContinuous() && component1i.isContinuous() && (!weighted || w1f.isContinuous()))
            cols *= sampleDf.rows, rows = 1;
        _sumW = weighted ? sum(w1f).val[0] : rows * cols;
    }
    CmGaussianFitter* fitters = new CmGaussianFitter[_K];
    for (int y = 0; y < rows; y++)  {
        int* components = component1i.ptr<int>(y);
        const float* img = sampleDf.ptr<float>(y);
        const float* w = weighted ? w1f.ptr<float>(y) : NULL;
        if (weighted){
            for (int x = 0; x < cols; x++, img += 3)
                fitters[0].Add(img, w[x]);
        }else{
            for (int x = 0; x < cols; x++, img += 3)
                fitters[0].Add(img);
        }
    }
    fitters[0].BuildGuassian(_Guassians[0], _sumW, true);
    int nSplit = 0;
    for (int i = 1; i < _K; i++) {
        if (_Guassians[nSplit].eValues[0] < _ThrV){
            _K = i;
            delete []fitters;
            return;
        }
        fitters[nSplit] = CmGaussianFitter();
        CmGaussian& sG = _Guassians[nSplit];
        double split = 0;
        for (int t = 0; t < 3; t++)
            split += sG.eVectors[t][0] * sG.mean[t];
        for (int y = 0; y < rows; y++)  {
            int* components = component1i.ptr<int>(y);
            const float* img = sampleDf.ptr<float>(y);
            if (weighted){
                const float* w = w1f.ptr<float>(y);
                for (int x = 0; x < cols; x++, img += 3) {// for each pixel
                    if (components[x] != nSplit)
                        continue;
                    double tmp = 0;
                    for (int t = 0; t < 3; t++)
                        tmp += sG.eVectors[t][0] * img[t];
                    if (tmp > split)
                        components[x] = i, fitters[i].Add(img, w[x]);
                    else
                        fitters[nSplit].Add(img, w[x]);
                }
            }else{
                for (int x = 0; x < cols; x++, img += 3) {// for each pixel
                    if (components[x] != nSplit)
                        continue;
                    double tmp = 0;
                    for (int t = 0; t < 3; t++)
                        tmp += sG.eVectors[t][0] * img[t];
                    if (tmp > split)
                        components[x] = i, fitters[i].Add(img);
                    else
                        fitters[nSplit].Add(img);
                }
            }
        }
        fitters[nSplit].BuildGuassian(_Guassians[nSplit], _sumW, true);
        fitters[i].BuildGuassian(_Guassians[i], _sumW, true);
        nSplit = 0;
        for (int j = 0; j <= i; j++)
            if (_Guassians[j].eValues[0] > _Guassians[nSplit].eValues[0])
                nSplit = j;
    }
    delete []fitters;
}

inline int CmGaussianMModel::RefineGMMs(const Mat& sampleDf, Mat& components1i, const Mat& w1f, bool needReAssign)
{
    bool weighted = w1f.data != NULL;
    int rows = sampleDf.rows, cols = sampleDf.cols; {
        if (sampleDf.isContinuous() && components1i.isContinuous() && (!weighted || w1f.isContinuous()))
            cols *= sampleDf.rows, rows = 1;
    }

    if (needReAssign)
        AssignEachPixel(sampleDf, components1i);
    CmGaussianFitter* fitters = new CmGaussianFitter[_K];
    for (int y = 0; y < rows; y++)  {
        const float* pixel = sampleDf.ptr<float>(y);
        const int* component = components1i.ptr<int>(y);
        if (weighted){
            const float* w = w1f.ptr<float>(y);
            for (int x = 0; x < cols; x++, pixel += 3)
                fitters[component[x]].Add(pixel, w[x]);
        }
        else
            for (int x = 0; x < cols; x++, pixel += 3)
                fitters[component[x]].Add(pixel);
    }


    int newK = 0;
    for (int i = 0; i < _K; i++)
        if (fitters[i].Count() > 0)
            fitters[i].BuildGuassian(_Guassians[newK++], _sumW, false);
    delete []fitters;
    _K = newK;
    AssignEachPixel(sampleDf, components1i);
    return _K;
}

inline void CmGaussianMModel::AssignEachPixel(const Mat& sampleDf, Mat &component1i)
{
    int rows = sampleDf.rows, cols = sampleDf.cols;
    if (sampleDf.isContinuous() && component1i.isContinuous())
        cols *= sampleDf.rows, rows = 1;

    for (int y = 0; y < rows; y++)  {
        const float* pixel = sampleDf.ptr<float>(y);
        int* component = component1i.ptr<int>(y);
        for (int x = 0; x < cols; x++, pixel += 3)  {
            int k = 0;
            double maxP = 0;
            for (int i = 0; i < _K; i++) {
                double posb = P(i, pixel);
                if (posb > maxP)
                    k = i, maxP = posb;
            }
            component[x] = k;
        }
    }
}