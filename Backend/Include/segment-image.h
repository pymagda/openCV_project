#ifndef SEGMENT_IMAGE
#define SEGMENT_IMAGE
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace cv;
typedef const Mat CMat;

#define THRESHOLD(size, c) (c/size)

typedef struct {
    int rank;
    int p;
    int size;
} uni_elt;

class universe {
public:
    universe(int elements);
    ~universe();
    int find(int x);
    void join(int x, int y);
    int size(int x) const { return elts[x].size; }
    int nu_sets() const { return num; }

private:
    uni_elt *elts;
    int num;
};

universe::universe(int elements) {
    elts = new uni_elt[elements];
    num = elements;
    for (int i = 0; i < elements; i++) {
        elts[i].rank = 0;
        elts[i].size = 1;
        elts[i].p = i;
    }
}

universe::~universe() {
    delete [] elts;
}

int universe::find(int x) {
    int y = x;
    while (y != elts[y].p)
        y = elts[y].p;
    elts[x].p = y;
    return y;
}

void universe::join(int x, int y) {
    if (elts[x].rank > elts[y].rank) {
        elts[y].p = x;
        elts[x].size += elts[y].size;
    } else {
        elts[x].p = y;
        elts[y].size += elts[x].size;
        if (elts[x].rank == elts[y].rank)
            elts[y].rank++;
    }
    num--;
}

typedef struct {
    float w;
    int a, b;
} edge;

bool operator<(const edge &a, const edge &b) {
    return a.w < b.w;
}

universe *segment_graph(int nu_vertices, int nu_edges, edge *edges, float c) {
    std::sort(edges, edges + nu_edges);

    universe *u = new universe(nu_vertices);

    float *threshold = new float[nu_vertices];
    for (int i = 0; i < nu_vertices; i++)
        threshold[i] = THRESHOLD(1,c);

    for (int i = 0; i < nu_edges; i++) {
        edge *pedge = &edges[i];

        int a = u->find(pedge->a);
        int b = u->find(pedge->b);
        if (a != b) {
            if ((pedge->w <= threshold[a]) &&
                (pedge->w <= threshold[b])) {
                u->join(a, b);
                a = u->find(a);
                threshold[a] = pedge->w + THRESHOLD(u->size(a), c);
            }
        }
    }

    delete threshold;
    return u;
}

template<typename T> inline T sqr(T x) { return x * x; }
static inline float diff(CMat &img3f, int x1, int y1, int x2, int y2)
{
	const Vec3f &p1 = img3f.at<Vec3f>(y1, x1);
	const Vec3f &p2 = img3f.at<Vec3f>(y2, x2);
	return sqrt(sqr(p1[0] - p2[0]) + sqr(p1[1] - p2[1]) + sqr(p1[2] - p2[2]));
}

int SegmentImage(CMat &_src3f, Mat &pImgInd, double sigma, double c, int min_size)
{
	int width(_src3f.cols), height(_src3f.rows);
	Mat smImg3f;
	GaussianBlur(_src3f, smImg3f, Size(), sigma, 0, BORDER_REPLICATE);

	edge *edges = new edge[width*height*4];
	int num = 0;
	{
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (x < width-1) {
					edges[num].a = y * width + x;
					edges[num].b = y * width + (x+1);
					edges[num].w = diff(smImg3f, x, y, x+1, y);
					num++;
				}

				if (y < height-1) {
					edges[num].a = y * width + x;
					edges[num].b = (y+1) * width + x;
					edges[num].w = diff(smImg3f, x, y, x, y+1);
					num++;
				}

				if ((x < width-1) && (y < height-1)) {
					edges[num].a = y * width + x;
					edges[num].b = (y+1) * width + (x+1);
					edges[num].w = diff(smImg3f, x, y, x+1, y+1);
					num++;
				}

				if ((x < width-1) && (y > 0)) {
					edges[num].a = y * width + x;
					edges[num].b = (y-1) * width + (x+1);
					edges[num].w = diff(smImg3f, x, y, x+1, y-1);
					num++;
				}
			}
		}
	}

	universe *u = segment_graph(width*height, num, edges, (float)c);

	for (int i = 0; i < num; i++) {
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
			u->join(a, b);
	}
	delete [] edges;

	map<int, int> marker;
	pImgInd.create(smImg3f.size(), CV_32S);

	int idxNum = 0;
	for (int y = 0; y < height; y++) {
		int *imgIdx = pImgInd.ptr<int>(y);
		for (int x = 0; x < width; x++) {
			int comp = u->find(y * width + x);
			if (marker.find(comp) == marker.end())
				marker[comp] = idxNum++;

			int idx = marker[comp];
			imgIdx[x] = idx;
		}
	}
	delete u;

	return idxNum;
}
#endif
