#include <opencv2/opencv.hpp>
typedef unsigned char uchar;
using namespace cv;
const int N = 2e6 + 10, M = N << 1, E = N + 2 * M, D = 257;

struct DFU {
	int f[N];
	int st0[N], top0;

	void init(int n);
	int Find(int u);
	void Union(int u, int v);
};

struct LTREE {
	int tot;
	int l[E], r[E], dis[E];
	int cur[E], id[E], val[E];
	int st[E][2], tt;//st[][0]=a,st[][1]=b->Merge
	int q[E], head, tail;//q[]->update

	int Merge(int a, int b);
	void top(int a, int &w, int &_id);
	void pop(int &a);
	int newNode(int k);
	void update(int a, int w);
};

struct Edge {
	int x, y, c;
	Edge();
	Edge(int _x, int _y, int _c);
};

struct Edge2 {
	int u, v, c;
	double w;
	Edge2();
	Edge2(int _u, int _v, int _c, double _w);
	bool operator <(const Edge2 &o)const;
};

struct Branch {
	int u, v, c, next;
	Branch();
	Branch(int _u, int _v, int _c, int _next);
};

struct DFU2 {
	int f[N << 1], g[N << 1];
	int st[N << 1], top;

	void init(int n);
	int Find(int u);
};

class MSA {
private:
	double max_dif_gra, max_dif_col, weight_col;

	uchar img3L[N * 3], img3R[N * 3];
	uchar m_img3L[N * 3], m_img3R[N * 3];
	uchar imgL[N], imgR[N];
	double graL[N], graR[N];
	double r_graL[N], r_graR[N];
	double c_graL[N], c_graR[N];
	double Exp[256];
	int n, m, Disp;
	int Time;

	int totE;
	int lk[N], totB;
	int in[N];
	int st[N], tt;
	int pre[N], key[N], inEdg[N];
	int root[N], topR;
	DFU W, S;
	LTREE heap;
	int Pre[N << 1], ie[N << 1], cnt;
	DFU2 bel;
	bool mark[N << 1];

	int queue[N], head, tail;
	int seq[N], fa[N];
	//double costL[N],costR[N],costUp[N],costA[N];
	//double costL[N*D],costR[N*D],costUp[N*D],costA[N*D];
	Mat costL, costR, costUp, costA;
	uchar disparity0[N], disparity1[N], tmp_disp[N];
	bool mask[N];
	bool nonocc[N];
	bool cuowu;
	int wei_edge[N];
	int m_edg[N];

	int reg_disp[N];
	Mat reg_cost, reg_costUp, reg_costA;

	Mat Dp, path;
	uchar best[N];

	inline int T(int u, int v);
	void init(Mat l, Mat r);
	void gradient(const uchar img[], double gra[]);
	void gradient_after_ctmf(const uchar img3[], double r_gra[], double c_gra[]);
	void getCost();
	void Insert(int v, int u, int c);
	//void build(int rt, const double gra[], const uchar img3[]);
	void build(int rt, const double r_gra[], const double c_gra[], const uchar img3[]);
	void add(int u, int v, int c);
	void Tarjan(int nn);

	bool vis[N];
	void check();
	void bb(int u, int fa);

	int anc[N];
	int gr_f1[N], gr_f2[N];//gr means getRoot
	void reg_getSeq();
	void reg_TreeDp();
	void getSeq0();
	void setRegion(Mat cost, const uchar img3[]);
	void getRoot();

	int sum[256];
	void baseSort(const uchar img3[]);
	void baseSort1(const uchar img3[]);
	void Kruskal(const uchar img3[]);
	void Kruskal1(const uchar img3[]);

	void getSeq();
	void get_weight_edge(bool left);
	//void TreeDp(const double cost[]);
	void TreeDp(Mat cost);
	//void TreeFilter(Mat cost);
	void WTA(uchar disparity[]);
	void LRcheck(uchar d1[], const uchar d2[], Mat cost);
	cv::Mat output(uchar d0[], int scale, bool Save);
	void setExp(double o);

	Mat pict;
	bool pic_used[N];
	int pic_f[N];
	int Find(int x);

	void paint();
	Mat seg;
	int col[N][3];

	struct node {
		int u, dis;
		node(){}
		node(int _u, int _dis) : u(_u), dis(_dis) {}
	};
	bool p_used[N];
	double p_dis[N];
	int p_pre[N];
	double dist(int i, int j, const uchar img3[]);
	void Prim(const uchar img3[]);
	int size[N], sum_col[N * 3];
	void getSize(const uchar img3[]);

	int rec[N];//test size


public:
	MSA(){}
	cv::Mat solve(Mat l, Mat r, int d, int scale, bool Save);
};
