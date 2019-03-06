
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "ctmf.h"
#include "MSA.h"
using namespace std;
using namespace cv;
Edge e[M];
Edge2 oriE[M], sortE[M];
Branch branch[M << 1];
int max_edg;

inline int MSA::T(int u, int v) {
	return u * m + v;
}

void MSA::init(Mat l, Mat r) {
	n = l.rows, m = l.cols;

	costL.create(1, n*m*Disp, CV_32F);
	costR.create(1, n*m*Disp, CV_32F);
	costUp.create(1, n*m*Disp, CV_32F);
	costA.create(1, n*m*Disp, CV_32F);
	//SumOfEdge = n*(m-1) + m*(n-1);
	max_dif_gra = 2.0;
	max_dif_col = 7.0;
	weight_col = 0.11;
	/*tansform 3 channel*/
	for (int i = 0, tot = 0; i < n; ++i) {
		uchar *pl = l.ptr<uchar>(i);
		uchar *pr = r.ptr<uchar>(i);
		for (int j = 0; j < 3 * m; ++j) {
			img3L[tot] = *pl++;
			img3R[tot] = *pr++;
			++tot;
		}
	}

	/*彩色转灰度*/
	for (int i = 0, tot = 0; i < n; ++i) {/// bgr
		for (int j = 0; j < m; ++j) {
			imgL[tot] = (int)(0.299*img3L[tot * 3 + 2] + 0.587*img3L[tot * 3 + 1] + 0.114*img3L[tot * 3] + 0.5);
			imgR[tot] = (int)(0.299*img3R[tot * 3 + 2] + 0.587*img3R[tot * 3 + 1] + 0.114*img3R[tot * 3] + 0.5);
			++tot;
		}
	}

	/*cale gradient*/
	gradient(imgL, graL);
	gradient(imgR, graR);
	getCost();

	ctmf(img3L, m_img3L, m, n, m * 3, m * 3, 1, 3, n*m * 3);
	ctmf(img3R, m_img3R, m, n, m * 3, m * 3, 1, 3, n*m * 3);

	gradient_after_ctmf(m_img3L, r_graL, c_graL);
	gradient_after_ctmf(m_img3R, r_graR, c_graR);
}

void MSA::gradient(const uchar img[], double gra[]) {
	for (int i = 0; i < n; ++i) {
		double plus = img[T(i, 1)], minus = img[T(i, 0)];
		gra[T(i, 0)] = plus - minus + 127.5;
		for (int j = 1; j < m - 1; ++j) {
			plus = img[T(i, j + 1)];
			gra[T(i, j)] = (plus - minus)*0.5 + 127.5;
			minus = img[T(i, j)];
		}
		gra[T(i, m - 1)] = (plus - minus) + 127.5;
	}
}

void MSA::getCost() {/// 初始代价是一个颜色值加梯度值的加权,是一个关于d的
	float *pL = costL.ptr<float>(0);
	float *pR = costR.ptr<float>(0);

	for (int i = 0; i < n; ++i) {
		int occ = T(i, 0);
		for (int j = 0; j < m; ++j) {
			for (int d = 0; d < Disp; ++d) {
				double dif_gra =
					min(abs(graL[T(i, j)] - ((j - d >= 0) ? graR[T(i, j - d)] : graR[occ])), max_dif_gra);
				double dif_col = 0.0;
				for (int k = 0; k < 3; ++k) {
					dif_col +=
						abs(img3L[T(i, j) * 3 + k] - ((j - d >= 0) ? img3R[T(i, j - d) * 3 + k] : img3R[occ * 3 + k]));
				}
				dif_col = min(dif_col / 3, max_dif_col);
				pL[T(i, j)*Disp + d] = weight_col * dif_col + (1 - weight_col) * dif_gra;
			}
		}
	}

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			for (int d = 0; d < Disp; ++d) {

				if (j + d<m) pR[T(i, j)*Disp + d] = pL[T(i, j + d)*Disp + d];
				else pR[T(i, j)*Disp + d] = pR[T(i, j)*Disp + d - 1];
			}
		}
	}
}

void MSA::gradient_after_ctmf(const uchar img3[], double r_gra[], double c_gra[]) {
	uchar img[N];
	for (int i = 0, tot = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			img[tot] = (int)(0.299*img3[tot * 3 + 2] + 0.587*img3[tot * 3 + 1] + 0.114*img3[tot * 3] + 0.5);
			++tot;
		}
	}
	for (int i = 0; i < n; ++i) {
		double plus = img[T(i, 1)], minus = img[T(i, 0)];
		r_gra[T(i, 0)] = plus - minus;
		for (int j = 1; j < m - 1; ++j) {
			plus = img[T(i, j + 1)];
			r_gra[T(i, j)] = (plus - minus)*0.5;
			minus = img[T(i, j)];
		}
		r_gra[T(i, m - 1)] = (plus - minus);
	}

	for (int j = 0; j < m; ++j) {
		double plus = img[T(1, j)], minus = img[T(0, j)];
		c_gra[T(0, j)] = plus - minus;
		for (int i = 1; i < n - 1; ++i) {
			plus = img[T(i + 1, j)];
			c_gra[T(i, j)] = (plus - minus) * 0.5;
			minus = img[T(i, j)];
		}
		c_gra[T(n - 1, j)] = (plus - minus);
	}
}

void MSA::Insert(int v, int u, int c) {
	e[totE] = Edge(u, v, c);
	in[v] = heap.Merge(in[v], heap.newNode(totE));
	++totE;
}

int MSA::Find(int x) {
	return (pic_f[x] == x) ? x : (pic_f[x] = Find(pic_f[x]));
}


void MSA::build(int rt, const double r_gra[], const double c_gra[], const uchar img3[]) {/// 有向边从低梯度指向高梯度, 如果梯度相等,建立双向边
	heap.tot = 0;
	totE = 0;
	memset(in, 0, sizeof(int)*(n*m + 1));
	for (int i = 0; i < n*m; ++i) {
		Insert(i, rt, 1e9);
	}
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m - 1; ++j) {
			int t1 = T(i, j), t2 = T(i, j + 1);
			int dif_col = 0;
			for (int k = 0; k < 3; ++k) {
				dif_col = max(dif_col, abs(img3[t1 * 3 + k] - img3[t2 * 3 + k]));
			}
			int dif_gra = abs(r_gra[t1]) - abs(r_gra[t2]);
			if (abs(dif_gra) <= 0) {
				Insert(t1, t2, dif_col);
				Insert(t2, t1, dif_col);
			}
			else if (dif_gra < 0) Insert(t2, t1, dif_col);
			else Insert(t1, t2, dif_col);
		}
	}
	for (int j = 0; j < m; ++j) {
		for (int i = 0; i < n - 1; ++i) {
			int t1 = T(i, j), t2 = T(i + 1, j);
			int dif_col = 0;
			for (int k = 0; k < 3; ++k) {
				dif_col = max(dif_col, abs(img3[t1 * 3 + k] - img3[t2 * 3 + k]));
			}
			int dif_gra = abs(c_gra[t1]) - abs(c_gra[t2]);
			if (abs(dif_gra) <= 0) {
				Insert(t1, t2, dif_col);
				Insert(t2, t1, dif_col);
			}
			else if (dif_gra < 0) Insert(t2, t1, dif_col);
			else Insert(t1, t2, dif_col);
		}
	}
}


void MSA::add(int u, int v, int c) {
	branch[totB] = Branch(u, v, c, lk[u]);
	lk[u] = totB++;
}


void MSA::Tarjan(int nn) {/// 实际上是根据颜色做的分割,因为点之间的边的权重就是颜色差
	memset(key, 0, sizeof(int)*(nn + 1));
	memset(pre, -1, sizeof(int)*(nn + 1));
	memset(Pre, -1, sizeof(int)*(nn + 1));
	memset(lk, -1, sizeof(int)*(nn + 1));
	tt = 0;
	for (int i = 0; i <= nn; ++i) {
		st[tt++] = i;
	}
	S.init(nn);
	W.init(nn);
	bel.init(2 * nn);
	memset(ie, -1, sizeof(int)*(2 * nn + 1));



	cnt = nn + 1;

	long long res = 0;

	while (tt) {
		int v = st[--tt];
		if (v != S.Find(v)) continue;

		int u, w;
		bool hasIn = false;
		int cur;
		while (in[v]) {
			heap.top(in[v], w, cur);
			heap.pop(in[v]);
			u = S.Find(e[cur].x);
			if (u != v) {
				hasIn = true;
				break;
			}
		}
		if (!hasIn) {
			if (v != nn) {
				printf("root is not nn ?\n");
			}
			continue;
		}

		res += w;
		pre[v] = u;
		key[v] = w;
		inEdg[v] = cur;

		int fv = bel.Find(e[cur].y);
		int ffv = bel.f[fv];
		ie[ffv] = cur;
		Pre[ffv] = fv;

		if (W.Find(u) == W.Find(v)) {
			bel.f[ffv] = cnt++;
			bel.g[ffv] = bel.f[ffv];
			if (key[v] > 0) {
				heap.update(in[v], key[v]);
				key[v] = 0;
			}
			for (int ek = inEdg[u], k = S.Find(pre[u]); k != v; ek = inEdg[k], k = S.Find(pre[k])) {

				int kk = e[ek].x;
				int fk = bel.Find(kk);
				int ffk = bel.f[fk];
				bel.f[ffk] = bel.f[ffv];
				bel.g[ffk] = bel.g[ffv];
				S.Union(v, k);
				if (key[k] > 0) {
					heap.update(in[k], key[k]);
					key[k] = 0;
				}
				in[v] = heap.Merge(in[v], in[k]);
			}
			if (key[u] > 0) {
				heap.update(in[u], key[u]);
				key[u] = 0;
			}

			int fu = bel.Find(e[cur].x);
			int ffu = bel.f[fu];
			bel.f[ffu] = bel.f[ffv];
			bel.g[ffu] = bel.g[ffv];
			S.Union(v, u);
			in[v] = heap.Merge(in[v], in[u]);
			st[tt++] = v;
		}
		else {
			W.Union(v, u);
		}
	}

	for (int i = 0; i < cnt; ++i) {
		bel.Find(i);
		pre[i] = -1;
	}
	memset(mark, 0, sizeof(bool)*cnt);
	for (int i = cnt - 1; i >= 0; --i) {
		if (i == nn) continue;
		if (mark[i]) continue;
		mark[i] = 1;
		int k = ie[i];
		int u = e[k].y;
		while (u != i) {
			mark[u] = 1;
			u = bel.g[u];
			if (u == bel.g[u]) break;
		}

		if (u == i) {
			pre[e[k].y] = e[k].x;
			key[e[k].y] = e[k].c;
			//ans2 += e[k].c;
		}
	}
	max_edg = 0;
	totB = 0;
	topR = 0;
	long long ans2 = 0;
	for (int i = 0; i < nn; ++i) {
		if (pre[i] == -1) {
			printf("Sth wrong!");
			continue;
		}
		if (pre[i] < nn) {
			add(pre[i], i, key[i]);
			add(i, pre[i], key[i]);
			ans2 += key[i];
			max_edg = max(max_edg, key[i]);
		}
		else {
			root[topR++] = i;
			//add(nn,i,0);
			ans2 += (int)1e9;
		}
	}

	printf("ans1 = %I64d\n", res);
	printf("ans2 = %I64d\n", ans2);
	printf("n = %d, m = %d\n", n, m);
	printf("totNode = %d\n", n*m);
	printf("totEdge = %d\n", totB / 2);
	printf("totRoot = %d\n", topR);
	printf("totEdge + totRoot = %d\n", totB / 2 + topR);
}



void MSA::getSeq0() {
	head = tail = -1;
	for (int i = 0; i < topR; ++i) {
		queue[++head] = root[i];
		fa[root[i]] = -1;
		anc[root[i]] = i;
		m_edg[i] = 0;
	}
	int tot = 0;
	while (tail < head) {
		int u = queue[++tail];
		seq[tot++] = u;
		for (int i = lk[u]; i > -1; i = branch[i].next) {
			int v = branch[i].v;
			if (v == fa[u]) continue;
			queue[++head] = v;
			fa[v] = u;
			anc[v] = anc[u];
			m_edg[anc[u]] = max(m_edg[anc[u]], branch[i].c);
		}
	}
	if (tot != n*m) {
		printf("the size of seq is wrong!\n");
	}
	/*tested*/
}

void MSA::getRoot() {
	//ȨΪ0�ı߻�ģ����û�бߵĸ�����Եü�һ
	for (int k = n*m - 1; k >= 0; --k) {
		int u = seq[k];
		gr_f1[u] = gr_f2[u] = 0;
		for (int i = lk[u]; i > -1; i = branch[i].next) {
			int v = branch[i].v;
			int c = branch[i].c + 1;
			if (v == fa[u]) continue;
			int tmp = gr_f1[v] + c;
			if (tmp > gr_f1[u]) {
				gr_f2[u] = gr_f1[u];
				gr_f1[u] = tmp;
			}
			else if (tmp > gr_f2[u]) {
				gr_f2[u] = tmp;
			}
		}
	}

	for (int k = 0; k < n*m; ++k) {
		int u = seq[k];
		for (int i = lk[u]; i > -1; i = branch[i].next) {
			int v = branch[i].v;
			int c = branch[i].c + 1;
			if (v == fa[u]) continue;
			int tmp = (gr_f1[u] == gr_f1[v] + c) ? gr_f2[u] + c : gr_f1[u] + c;
			if (tmp>gr_f1[v]) {
				gr_f2[v] = gr_f1[v];
				gr_f1[v] = tmp;
			}
			else if (tmp>gr_f2[v]) {
				gr_f2[v] = tmp;
			}
		}
	}
	for (int i = 0; i < n*m; ++i) {
		int k = anc[i];
		if (gr_f1[i] < gr_f1[root[k]]) {
			root[k] = i;
		}
	}

}

double MSA::dist(int u, int v, const uchar img3[]) {
	int u1 = u / m, u2 = u%m;
	int v1 = v / m, v2 = v%m;
	int u3 = (int)(0.299*img3[u * 3 + 2] + 0.587*img3[u * 3 + 1] + 0.114*img3[u * 3] + 0.5);
	int v3 = (int)(0.299*img3[v * 3 + 2] + 0.587*img3[v * 3 + 1] + 0.114*img3[v * 3] + 0.5);

	assert(size[u]>0 && size[v]>0);
	/*��dif_colӦ����max������sum/3*/
	int dif_col = 0;
	for (int t = 0; t < 3; ++t) {
		//dif_col = max(dif_col, abs((sum_col[u*3+t]+size[u]-1)/size[u]-(sum_col[v*3+t]+size[v]-1)/size[v]));
		//dif_col += abs((sum_col[u*3+t]+size[u]-1)/size[u]-(sum_col[v*3+t]+size[v]-1)/size[v]);
		dif_col = max(dif_col, abs(sum_col[u * 3 + t] / size[u] - sum_col[v * 3 + t] / size[v]));
	}
	int dif_size = abs(size[u] - size[v]);
	//double all_dis = sqrt((u1-v1)*(u1-v1)+(u2-v2)*(u2-v2) ) + dif_col;
	//double all_dis = abs(u1-v1) + abs(u2-v2) + dif_col;
	//double all_dis = abs(u1-v1)+abs(u2-v2)+dif_col;
	//return all_dis*(dif_col*dif_col)*(dif_size);
	/*����д����TeddyЧ������õ�*/
	double all_dis = sqrt((u1 - v1)*(u1 - v1) + (u2 - v2)*(u2 - v2));
	if (dif_col > 2) all_dis += min(dif_col * 2, 20);
	if (dif_size > 30) all_dis += min(dif_size, 50);

	assert(all_dis >= 0);
	return all_dis;
	/*
	double all_dis = sqrt((u1-v1)*(u1-v1)+(u2-v2)*(u2-v2));
	if (dif_col > 2) all_dis += min(dif_col*2,20);
	if (dif_size > 30) all_dis += min(dif_size,50);
	*/

}

void MSA::Prim(const uchar img3[]) {
	memset(p_used, 0, sizeof(bool)*topR);
	p_used[0] = 1;
	getSize(img3);
	for (int i = 0; i < topR; ++i) {
		++rec[size[root[i]]];
		p_pre[i] = -1;
	}
	/*
	printf("===test===\n");
	for (int i = 0; i < N; ++i) {
	if (rec[i] > 0) {
	printf("%d --- %d\n",i,rec[i]);
	}
	}
	printf("===test===\n");
	*/
	for (int i = 1; i < topR; ++i) {
		p_dis[i] = dist(root[0], root[i], img3);
		p_pre[i] = 0;
	}
	while (1) {
		int k = -1;
		for (int i = 0; i < topR; ++i) {
			if (!p_used[i] && (k == -1 || p_dis[i] < p_dis[k])) {
				k = i;
			}
		}
		if (k == -1) break;
		p_used[k] = 1;
		for (int i = 0; i < topR; ++i) {
			if (!p_used[i]) {
				double tmp = dist(root[i], root[k], img3);
				if (p_dis[i] > tmp) {
					p_dis[i] = tmp;
					p_pre[i] = k;
				}
			}
		}
	}
	/*test
	int MM = 0;
	for (int i = 1; i < topR; ++i) {
	MM = max(MM,Dis[i]);
	}
	test*/
	//���graͼӦ���ڽ����и�������֮ǰ

	paint();


	for (int i = 1; i < topR; ++i) {
		int dif_col = 0;
		int u = root[i], v = root[p_pre[i]];
		for (int t = 0; t < 3; ++t) {
			dif_col += //max(dif_col, abs(img3[root[i]*3+t]-img3[root[p_pre[i]]*3+t]));
				max(dif_col, abs(sum_col[u * 3 + t] / size[u] - sum_col[v * 3 + t] / size[v]));
			//max(dif_col, abs((sum_col[u*3+t]+size[u]-1)/size[u]-(sum_col[v*3+t]+size[v]-1)/size[v]));


		}
		//if (max(size[u],size[v]) > 30) add(root[p_pre[i]],root[i],dif_col);
		//else add(root[p_pre[i]],root[i],0.9*dif_col);
		//if (dif_col <= 10) add(root[p_pre[i]],root[i],0.95*dif_col);

		add(u, v, dif_col);//add(u,v,255);
		add(v, u, dif_col);//add(v,u,255);

		//add(u,v,10*dif_col);
		//add(v,u,10*dif_col);
		//printf("%.3lf\n",Exp[255]);
		//add(root[p_pre[i]],root[i],dif_col);
	}
	/*test*/
	for (int i = 0; i < topR; ++i) {
		int k = i;
		int tt = 0;
		while (k != 0) {
			k = p_pre[k];
			if (++tt > topR) {
				printf("really not connect to root[0]!\n");
				break;
			}
		}
	}
	assert(totB / 2 == n*m - 1);
	check();
	/*test*/

}

void MSA::baseSort(const uchar img3[]) {
	totE = 0;
	memset(sum, 0, sizeof(sum));
	/*use 3 channel*/
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m - 1; ++j) {
			int u = root[anc[T(i, j)]], v = root[anc[T(i, j + 1)]];
			if (u == v) continue;
			if (size[u] == 0 || size[v] == 0) {
				printf("/0 wrong!\n");
				return;
			}
			int dif_col = 0;
			for (int k = 0; k < 3; ++k) {
				dif_col = max(dif_col,
					//abs(sum_col[u*3+k]/size[u] - sum_col[v*3+k]/size[v]));
					abs((sum_col[u * 3 + k] + size[u] - 1) / size[u] - (sum_col[v * 3 + k] + size[v] - 1) / size[v]));
				//dif_col = max(dif_col, abs(img3[u*3+k]-img3[v*3+k]));
			}
			int u1 = u / m, u2 = u%m;
			int v1 = v / m, v2 = v%m;
			//double w = sqrt((u1-v1)*(u1-v1) + (u2-v2)*(u2-v2) + dif_col*dif_col);
			int dif_size = abs(size[u] - size[v]);
			double wei = 0.05;
			//double w = sqrt((u1-v1)*(u1-v1) + (u2-v2)*(u2-v2)) + 10*dif_col;
			double all_dis = sqrt((u1 - v1)*(u1 - v1) + (u2 - v2)*(u2 - v2));
			if (dif_col > 2) all_dis += min(dif_col * 2, 20);
			if (dif_size > 30) all_dis += min(dif_size, 50);
			//double all_dis = sqrt((u1-v1)*(u1-v1)+(u2-v2)*(u2-v2));
			//if (dif_col > 2) all_dis += min(dif_col*2,20);
			//if (dif_size > 30) all_dis += min(dif_size,50);
			oriE[totE++] = Edge2(u, v, dif_col, all_dis);
			//++sum[oriE[totE].c];
			//++totE;
		}
	}
	for (int j = 0; j < m; ++j) {
		for (int i = 0; i < n - 1; ++i) {
			int u = root[anc[T(i, j)]], v = root[anc[T(i + 1, j)]];
			if (u == v) continue;
			int dif_col = 0;
			for (int k = 0; k < 3; ++k) {
				dif_col = max(dif_col,
					//abs(sum_col[u*3+k]/size[u] - sum_col[v*3+k]/size[v]));
					abs((sum_col[u * 3 + k] + size[u] - 1) / size[u] - (sum_col[v * 3 + k] + size[v] - 1) / size[v]));
				//dif_col = max(dif_col, abs(img3[u*3+k]-img3[v*3+k]));
			}
			int u1 = u / m, u2 = u%m;
			int v1 = v / m, v2 = v%m;
			int dif_size = abs(size[u] - size[v]);
			double wei = 0.05;
			//double w = sqrt((u1-v1)*(u1-v1) + (u2-v2)*(u2-v2)) + 10*dif_col;
			//double w = abs(u1-v1)+abs(u2-v2)+2*dif_col;
			double all_dis = sqrt((u1 - v1)*(u1 - v1) + (u2 - v2)*(u2 - v2));
			if (dif_col > 2) all_dis += min(dif_col * 2, 20);
			if (dif_size > 30) all_dis += min(dif_size, 50);
			//double all_dis = sqrt((u1-v1)*(u1-v1)+(u2-v2)*(u2-v2));
			//if (dif_col > 2) all_dis += min(dif_col*2,20);
			//if (dif_size > 30) all_dis += min(dif_size,50);
			oriE[totE++] = Edge2(u, v, dif_col, all_dis);
			//++sum[oriE[totE].c];
			//++totE;
		}
	}
	printf("the amount of edges of region is %d\n", totE);
	sort(oriE, oriE + totE);
	memcpy(sortE, oriE, sizeof(Edge2)*totE);
	/*
	int prev = sum[0];
	sum[0] = 0;
	for(int i = 1; i <= 255; ++i) {
	int cur = sum[i];
	sum[i] = prev + sum[i-1];
	prev = cur;
	}
	for (int i = 0; i < totE; ++i) {
	sortE[sum[oriE[i].c]++] = oriE[i];
	//sortE[--sum[oriE[i].c]] = oriE[i];
	}
	*/
}

void MSA::Kruskal(const uchar img3[]) {
	paint();
	getSize(img3);
	baseSort(img3);
	S.init(n*m);
	for (int i = 0; i < totE && totB < (n*m - 1) * 2; ++i) {
		int u = sortE[i].u, v = sortE[i].v, c = sortE[i].c;
		int fu = S.Find(u), fv = S.Find(v);
		int dif_size = abs(size[u] - size[v]);
		if (fu != fv && c <= 0 && dif_size <= n*m / 2000) {
			add(u, v, c);
			add(v, u, c);
			S.f[fu] = fv;
		}
	}

	for (int i = 0; i < totE && totB < (n*m - 1) * 2; ++i) {
		int u = sortE[i].u, v = sortE[i].v, c = sortE[i].c;
		int fu = S.Find(u), fv = S.Find(v);
		int dif_size = abs(size[u] - size[v]);
		if (fu != fv) {
			add(u, v, 255);
			add(v, u, 255);
			S.f[fu] = fv;
		}
	}
	/*the least cost of mst is right*/
	if (totB / 2 != n*m - 1) {
		printf("the total of edges of mst is wrong!\n");
		return;
	}
	/*tested*/
}

void MSA::baseSort1(const uchar img3[]) {
	totE = 0;
	memset(sum, 0, sizeof(sum));
	/*use 3 channel*/
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m - 1; ++j) {
			int u = T(i, j), v = T(i, j + 1);
			int fu = anc[u], fv = anc[v];
			assert(fu<topR && fv<topR);
			if (fu == fv) continue;
			int ru = root[fu], rv = root[fv];
			int dif_col = 0;
			for (int k = 0; k < 3; ++k) {
				//dif_col = max(dif_col, abs(sum_col[ru*3+k]/size[ru]-sum_col[rv*3+k]/size[rv]));
				dif_col = max(dif_col, abs(img3[u * 3 + k] - img3[v * 3 + k]));
			}

			oriE[totE++] = Edge2(u, v, dif_col, dif_col);

			dif_col = 0;
			for (int k = 0; k < 3; ++k) {
				dif_col = max(dif_col, abs(sum_col[ru * 3 + k] / size[ru] - sum_col[rv * 3 + k] / size[rv]));
				//dif_col = max(dif_col, abs(img3[u*3+k]-img3[v*3+k]));
			}

			int u1 = root[fu] / m, u2 = root[fu] % m;
			int v1 = root[fv] / m, v2 = root[fv] % m;
			int dif_size = abs(size[u] - size[v]);
			//double all_dis = sqrt((u1-v1)*(u1-v1)+(u2-v2)*(u2-v2));
			//if (dif_col > 2) all_dis += min(dif_col*2,20);
			//if (dif_size > 30) all_dis += min(dif_size,50);
			/*�������ֱ߲��ҵڶ���dif_col/5��Tsukuba��Teddy��Ч����ǰ�����㷨�ã�Cones�Բ�һ�㣬��Venus��ܶ�*/
			/*����getRoot��Cones��������֮�䣬��Venus��Ȼ��ܶ�*/
			/*���뵽�ĸĽ���Ҫô�����������ʱ���¹���Ҫô����һ���Լ������occ����θĽ�*/
			double all_dis = sqrt((u1 - v1)*(u1 - v1) + (u2 - v2)*(u2 - v2))*dif_size*dif_col;
			//if (dif_col > 2) all_dis += min(dif_col*2,20);
			//if (dif_size > 30) all_dis += min(dif_size,50);
			/*��0.2��Ч������/5��*/
			oriE[totE++] = Edge2(ru, rv, dif_col, dif_col*0.2);
			//oriE[totE++] = Edge2(ru, rv, dif_col, all_dis);
			//++sum[oriE[totE].c];
			//++totE;
		}
	}
	for (int j = 0; j < m; ++j) {
		for (int i = 0; i < n - 1; ++i) {
			int u = T(i, j), v = T(i, j + 1);
			int fu = anc[u], fv = anc[v];
			assert(fu<topR && fv<topR);
			if (fu == fv) continue;
			int ru = root[fu], rv = root[fv];
			int dif_col = 0;
			for (int k = 0; k < 3; ++k) {
				//dif_col = max(dif_col, abs(sum_col[ru*3+k]/size[ru]-sum_col[rv*3+k]/size[rv]));
				dif_col = max(dif_col, abs(img3[u * 3 + k] - img3[v * 3 + k]));
			}

			oriE[totE++] = Edge2(u, v, dif_col, dif_col);

			dif_col = 0;
			for (int k = 0; k < 3; ++k) {
				dif_col = max(dif_col, abs(sum_col[ru * 3 + k] / size[ru] - sum_col[rv * 3 + k] / size[rv]));
				//dif_col = max(dif_col, abs(img3[u*3+k]-img3[v*3+k]));
			}

			int u1 = root[fu] / m, u2 = root[fu] % m;
			int v1 = root[fv] / m, v2 = root[fv] % m;
			int dif_size = abs(size[u] - size[v]);
			double all_dis = sqrt((u1 - v1)*(u1 - v1) + (u2 - v2)*(u2 - v2))*dif_size*dif_col;
			//if (dif_col > 2) all_dis += min(dif_col*2,20);
			//if (dif_size > 30) all_dis += min(dif_size,50);
			oriE[totE++] = Edge2(ru, rv, dif_col, dif_col*0.2);
			//oriE[totE++] = Edge2(ru, rv, dif_col, all_dis);
			//++sum[oriE[totE].c];
			//++totE;
		}
	}
	printf("the amount of edges of region is %d\n", totE);
	/*stable_sort��Ч����sort���*/
	sort(oriE, oriE + totE);
	memcpy(sortE, oriE, sizeof(Edge2)*totE);
	/*
	for (int i = 0; i < totE; ++i) {
	double w = sortE[i].w;
	assert((w-((int)w))==0.0);
	}
	printf("all double is int!\n");
	*/
	/*
	int prev = sum[0];
	sum[0] = 0;
	for(int i = 1; i <= 255; ++i) {
	int cur = sum[i];
	sum[i] = prev + sum[i-1];
	prev = cur;
	}
	for (int i = 0; i < totE; ++i) {
	sortE[sum[oriE[i].c]++] = oriE[i];
	//sortE[--sum[oriE[i].c]] = oriE[i];
	}
	*/
}

void MSA::Kruskal1(const uchar img3[]) {
	getSize(img3);
	baseSort1(img3);
	S.init(topR);
	int new_add = 0;
	for (int i = 0; i < totE && totB < (n*m - 1) * 2; ++i) {
		int u = sortE[i].u, v = sortE[i].v, c = sortE[i].c;
		int fu = S.Find(anc[u]), fv = S.Find(anc[v]);
		assert(fu<topR && fv<topR);
		int s_u = size[root[anc[u]]], s_v = size[root[anc[v]]];
		int tu = m_edg[anc[u]] + sqrt(n*m) * 150 / 128 * s_u;
		int tv = m_edg[anc[v]] + sqrt(n*m) * 150 / 128 * s_v;
		int lim = 50;
		int area = 50;

		if (fu != fv &&c<min(tu, tv) && (abs(s_u - s_v) <= lim) && (s_u <= area || s_v <= area)) {
			int top = max(m_edg[anc[u]], m_edg[anc[v]]);
			add(u, v, c);
			add(v, u, c);
			S.f[fu] = fv;
			++new_add;
			size[root[anc[u]]] += size[root[anc[v]]];
			size[root[anc[v]]] = size[root[anc[u]]];
			m_edg[anc[u]] = max(m_edg[anc[u]], m_edg[anc[v]]);
			anc[v] = anc[u];
		}
	}
	printf("new_add = %d\n", new_add);
	printf("max_edge = %d\n", max_edg);

	for (int i = 0; i < totE && totB < (n*m - 1) * 2; ++i) {
		int u = sortE[i].u, v = sortE[i].v, c = sortE[i].c;
		int fu = S.Find(anc[u]), fv = S.Find(anc[v]);
		assert(fu<topR && fv<topR);
		if (fu != fv) {
			add(u, v, 255);
			add(v, u, 255);
			S.f[fu] = fv;
		}
	}

	paint();
	assert(totB / 2 == n*m - 1);

}

void MSA::paint() {
	seg.create(n, m, CV_8UC3);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			seg.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
	}
	srand((unsigned int)time(NULL));
	for (int i = 0; i < topR; ++i) {
		for (int j = 0; j < 3; ++j) {
			col[root[i]][j] = rand() % 256 + 1;
		}
	}

	for (int k = 0; k < n*m; ++k) {
		int u = seq[k];
		int rt = root[anc[u]];
		seg.at<Vec3b>(u / m, u%m) = Vec3b(col[rt][0], col[rt][1], col[rt][2]);
	}
	/*
	for (int i = 0; i < totB; ++i) {
	int u = branch[i].u, v = branch[i].v;
	line(seg,Point(u%m,u/m),Point(v%m,v/m),Scalar(255,0,0));
	}
	*/
	/*
	for (int i = 0; i < topR; ++i) {
	int k = root[i];
	int y = k/m, x = k%m;
	circle(seg,Point(x,y),1,Scalar(0,0,255),1);
	}
	*/
	//imshow("SegByGra", seg);
	//imwrite("seg.png", seg);
	/*
	seg.at<Vec3b>(rt/m,rt%m) = Vec3b(col[k][0], col[k][1], col[k][2]);
	for (int i = lk[rt]; i > -1; i = branch[i].next) {
	if (branch[i].v != fa) {
	paint(branch[i].v, rt, k);
	}
	}
	*/
}

void MSA::getSize(const uchar img3[]) {
	memset(size, 0, sizeof(int)*n*m);
	memset(rec, 0, sizeof(int)*n*m);
	memset(sum_col, 0, sizeof(int)*n*m * 3);
	for (int k = 0; k < n*m; ++k) {
		int u = seq[k];
		int rt = root[anc[u]];
		++size[rt];
		for (int i = 0; i < 3; ++i) {
			sum_col[rt * 3 + i] += img3[u * 3 + i];
		}
	}
	/*
	++size[rt];
	for (int k = 0; k < 3; ++k) {
	sum_col[rt*3+k] += img3[u*3+k];
	}
	for (int i = lk[u]; i > -1; i = branch[i].next) {
	if (branch[i].v != fa) {
	getSize(branch[i].v, u, rt, img3);
	}
	}
	*/
}

void MSA::check() {
	memset(vis, 0, sizeof(vis));
	bb(root[0], -1);
	for (int i = 0; i < n*m; ++i) {
		assert(vis[i] == true);
	}
}

void MSA::bb(int u, int fa) {
	assert(vis[u] == false);
	vis[u] = 1;
	//printf("%d\n",u);
	for (int i = lk[u]; i > -1; i = branch[i].next) {
		if (branch[i].v != fa) {
			bb(branch[i].v, u);
		}
	}
}

void MSA::getSeq() {
	/*
	head = tail = -1;
	for (int i = 0; i < topR; ++i) {
	queue[++head] = root[i];
	fa[root[i]] = -1;
	}
	*/
	queue[0] = root[0];
	fa[root[0]] = -1;
	head = 0;
	tail = -1;
	memset(fa, -1, sizeof(int)*n*m);
	int tot = 0;
	while (tail < head) {
		int u = queue[++tail];
		seq[tot++] = u;
		for (int i = lk[u]; i > -1; i = branch[i].next) {
			int v = branch[i].v;
			if (v == fa[u]) continue;
			queue[++head] = v;
			fa[v] = u;
		}
	}
	if (tot != n*m) {
		printf("the size of seq is wrong!\n");
	}
	/*tested*/
}

//void MSA::TreeDp(const double cost[]) {
void MSA::TreeDp(Mat cost) {
	//memcpy(costUp,cost,sizeof(double)*n*m*Disp);
	cost.copyTo(costUp);
	float *p = cost.ptr<float>(0);
	float *pUp = costUp.ptr<float>(0);
	float *pA = costA.ptr<float>(0);

	double res = 0;
	for (int i = 0; i < n*m*Disp; ++i) {
		res += p[i];//cost[i];
	}
	printf("the tot cost = %.5lf\n", res);
	for (int k = n*m - 1; k >= 0; --k) {
		int u = seq[k];
		for (int i = lk[u]; i > -1; i = branch[i].next) {
			int v = branch[i].v;
			double w = Exp[branch[i].c];
			if (v == fa[u]) continue;
			if (v == u) {
				printf("self cycle!\n");
			}
			for (int d = 0; d < Disp; ++d) {
				//costUp[u*Disp+d] += w * costUp[v*Disp+d];
				pUp[u*Disp + d] += w * pUp[v*Disp + d];
			}
		}
	}
	res = 0;
	for (int i = 0; i < n*m*Disp; ++i) {
		res += pUp[i];//costUp[i];
	}
	printf("the tot costUp = %.5lf\n", res);
	for (int d = 0; d < Disp; ++d) {
		//costA[root[0]*Disp+d] = costUp[root[0]*Disp+d];
		pA[root[0] * Disp + d] = pUp[root[0] * Disp + d];
	}
	/*
	for (int i = 0; i < topR; ++i) {
	int k = root[i];
	for (int d = 0; d < Disp; ++d) {
	costA[k*Disp+d] = costUp[k*Disp+d];
	}
	}
	*/
	for (int k = 0; k < n*m; ++k) {
		int u = seq[k];
		for (int i = lk[u]; i > -1; i = branch[i].next) {
			int v = branch[i].v;
			double w = Exp[branch[i].c];
			if (v == fa[u]) continue;
			for (int d = 0; d < Disp; ++d) {
				//costA[v*Disp+d] = w * costA[u*Disp+d] + (1 - w * w) * costUp[v*Disp+d];
				pA[v*Disp + d] = w * pA[u*Disp + d] + (1 - w*w) * pUp[v*Disp + d];
			}
		}
	}
	res = 0;
	for (int i = 0; i < n*m*Disp; ++i) {
		res += pA[i];//costA[i];
	}
	printf("the tot costA = %.5lf\n", res);
}

void MSA::WTA(uchar disparity[]) {
	float *pA = costA.ptr<float>(0);
	for (int i = 0; i < n*m; ++i) {
		int k = 0;
		for (int j = 1; j < Disp; ++j) {
			//if (costA[i*Disp+j] < costA[i*Disp+k]) {
			if (pA[i*Disp + j] < pA[i*Disp + k]) {
				k = j;
			}
		}
		disparity[i] = k;
	}
	uchar tmpD[N];
	memcpy(tmpD, disparity, sizeof(uchar)*n*m);
	ctmf(tmpD, disparity, m, n, m, m, 2, 1, n*m);
	/*test*/
	int res = 0;
	Mat test;
	test.create(n, m, CV_8UC1);
	uchar det = 256 / Disp;
	for (int i = 0, tot = 0; i < n; ++i) {
		uchar *p = test.ptr<uchar>(i);
		for (int j = 0; j < m; ++j) {
			res += disparity[tot];
			*p++ = disparity[tot++];
		}
	}
	//imshow("test", test);
//	imwrite("disp.png", test);
//	waitKey();
	printf("the tot disparity = %d\n", res);
	/*test*/
}

//void MSA::LRcheck(uchar d1[],const uchar d2[],double cost[]) {
void MSA::LRcheck(uchar d1[], const uchar d2[], Mat cost) {
	printf("LR-check");
	float *p = cost.ptr<float>(0);
	memset(mask, 0, sizeof(bool)*n*m);

	memset(nonocc, 0, sizeof(bool)*n*m);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			int d = d1[T(i, j)];
			if (j - d >= 0) {
				if (d > 0 && abs(d - d2[T(i, j - d)]) == 0) {
					mask[T(i, j)] = 1;
				}
				else if (abs(d - d2[T(i, j - d)])<1)
					nonocc[T(i, j)] = 1;
			}
		}
	}

	Mat tmp;
	uchar det = 255 / Disp;
	tmp.create(n, m, CV_8UC1);
	for (int i = 0, tot = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j, ++tot) {
			if (mask[tot]) tmp.at<uchar>(i, j) = d1[tot] * det;
			//if (mask[tot] || nonocc[tot]) tmp.at<Vec3b>(i, j) = { 255, 255, 255 };
			//else if (nonocc[tot]) tmp.at<Vec3b>(i, j) = { 255, 0, 0};
			else tmp.at<uchar>(i, j) = 0;
		}
	}
//	imshow("stable image", tmp);
//	imwrite("stable.png", tmp);
//	waitKey();

	//memset(cost,0,sizeof(double)*n*m*Disp);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			int k = T(i, j);
			//int costmin = 0.1*(Disp-1)+0.5;
			if (mask[k]) {
				for (int d = 0; d < Disp; ++d) {
					p[k*Disp + d] = abs(d - d1[k]);
				}
			}

			//else if(nonocc[k]){
			//	int dmin = 80;
			//	for (int m = -1; m < 2; m++)
			//	{
			//		//printf("%d   ", d1[T(i, j + m)]);
			//			if (mask[T(i, j+m)])
			//				if (dmin>d1[T(i, j + m)])
			//					dmin = d1[T(i, j+m)];
			//	}
			//	if (dmin == 80)
			//	{
			//		for (int d = 0; d < Disp; ++d) {
			//			p[k*Disp + d] = 0;
			//		}
			//		cuowu = 1;
			//	}

			//	//printf("~~~~~~~~%d~~~%d~~~~", dmin, mask[T(i, j + m)]);
			//	if (!cuowu)
			//	{
			//		for (int d = 0; d < Disp; ++d) {
			//			p[k*Disp + d] = abs(d - dmin);
			//		}
			//	}
			//}
			else
			for (int d = 0; d < Disp; ++d) {
				p[k*Disp + d] = 0;
			}

		}
	}
}

cv::Mat MSA::output(uchar d0[], int scale, bool Save) {
	Mat ans;
	ans.create(n, m, CV_8UC1);
	int cur = 0;
	//uchar det = 256/Disp;
	for (int i = 0; i < n; ++i) {
		uchar *p = ans.ptr<uchar>(i);
		for (int j = 0; j < m; ++j) {
			*p++ = d0[cur++] * scale;
		}
	}
	imshow("ans",ans);
	if (Save) {
		printf("n = %d, m = %d\n", ans.rows, ans.cols);
		imwrite("test.png", ans);
	}
	return ans;
}

void MSA::setExp(double o) {
	for (int i = 0; i <= 255; ++i) {
		Exp[i] = exp(-i*1.0 / o / 255);
	}
}

cv::Mat MSA::solve(Mat l, Mat r, int d, int scale, bool Save) {

	Disp = d + 1;
	double o = 0.1;
	/*init*/
	init(l, r);
	printf("n == %d, m == %d\n", n, m);
	setExp(o);

	/*right image as base image*/
	build(n*m, r_graR, c_graR, m_img3R);/// row  col
	Tarjan(n*m);
	getSeq0();
	//getRoot();
	Kruskal1(m_img3R);
	getSeq();
	TreeDp(costR);
	WTA(disparity1);



	/*left image as base image*/
	build(n*m, r_graL, c_graL, m_img3L);
	Tarjan(n*m);
	getSeq0();
	//getRoot();
	Kruskal1(m_img3L);
	getSeq();
	TreeDp(costL);
	WTA(disparity0);
	/*refine*/
	LRcheck(disparity0, disparity1, costL);
	setExp(o / 2);
	TreeDp(costL);
	WTA(disparity0);
	cv::Mat disp_img=output(disparity0, scale, Save);
	return disp_img;
}


void DFU::init(int n) {
	for (int i = 0; i <= n; ++i) f[i] = i;
}

int DFU::Find(int u) {
	top0 = 0;
	while (u != f[u]) {
		st0[top0++] = u;
		u = f[u];
	}
	while (top0) {
		f[st0[--top0]] = u;
	}
	return u;
}

void DFU::Union(int u, int v) {
	u = Find(u);
	v = Find(v);
	f[v] = u;
}


Edge::Edge() {}
Edge::Edge(int _x, int _y, int _c) : x(_x), y(_y), c(_c) {}

Edge2::Edge2() {}
Edge2::Edge2(int _u, int _v, int _c, double _w) : u(_u), v(_v), c(_c), w(_w) {}
bool Edge2::operator <(const Edge2 &o)const {
	return w < o.w;
}

Branch::Branch() {}
Branch::Branch(int _u, int _v, int _c, int _next) : u(_u), v(_v), c(_c), next(_next) {}

int LTREE::Merge(int a, int b) {
	if (!a || !b) return a + b;
	tt = 0;
	st[tt][0] = a;
	st[tt][1] = b;
	while (1) {
		int u = st[tt][0], v = st[tt][1];
		if (!u || !v) {
			st[tt][0] = u + v;
			break;
		}
		if (val[u] > val[v]) {
			swap(st[tt][0], st[tt][1]);
			swap(u, v);
		}
		++tt;
		st[tt][0] = r[u];
		st[tt][1] = v;

		if (tt >= M) {
			printf("bao zhan!\n");
			break;
		}
	}
	for (int i = tt - 1; i >= 0; --i) {
		int u = st[i][0], v = st[i][1];
		r[u] = st[i + 1][0];
		if (dis[l[u]] < dis[r[u]]) swap(l[u], r[u]);
		dis[u] = (!r[u]) ? 0 : (dis[r[u]] + 1);
	}
	return st[0][0];
	/*
	if (!a || !b) return a + b;
	if (val[a] > val[b]) swap(a, b);
	r[a] = Merge(r[a], b);
	if (dis[l[a]] < dis[r[a]]) swap(l[a], r[a]);
	dis[a] = (!r[a]) ? 0 : (dis[r[a]] + 1);
	return a;
	*/
}

void LTREE::top(int a, int &w, int &_id) {
	w = val[a];
	_id = id[a];
}

void LTREE::pop(int &a) {
	a = Merge(l[a], r[a]);
}

int LTREE::newNode(int k) {
	id[++tot] = k;
	val[tot] = e[k].c;
	l[tot] = r[tot] = dis[tot] = 0;
	return tot;
}

void LTREE::update(int a, int w) {
	if (!a) return;
	head = tail = -1;
	q[++tail] = a;
	while (head < tail) {
		int u = q[++head];
		val[u] -= w;
		if (l[u]) q[++tail] = l[u];
		if (r[u]) q[++tail] = r[u];
		if (tail >= E) {
			printf("bao queue!\n");
			for (int i = 0; i < head; ++i) {
				if (q[i] == u) {
					printf("overlap q[]!\n");
					printf("%d --- %d\n", i, head);
					printf("%d --- %d\n", q[i], u);
					break;
				}
			}
		}
	}
	/*
	if (!a) return;
	val[a] -= w;
	update(l[a], w);
	update(r[a], w);
	*/
}

void DFU2::init(int n) {
	for (int i = 0; i <= n; ++i) {
		f[i] = i;
		g[i] = i;
	}
}
int DFU2::Find(int u) {
	top = 0;
	int k = u;
	while (u != f[u]) {
		st[top++] = u;
		u = f[u];
	}
	if (f[k] == u || f[k] == k) return k;
	u = st[--top];
	while (top) {
		f[st[--top]] = u;
	}
	return u;
}

