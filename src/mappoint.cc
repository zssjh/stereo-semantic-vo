//
// Created by zss on 18-12-17.
//

#include <mappoint.h>
#include <pnpmatch.h>
#include <vector>

using namespace std;
mappoint::mappoint(cv::Mat &pos,frame *pFrame, int id):worldpos(pos),bad(false)
{
    pFrame->f_descriptor.row(id).copyTo(m_descriptor);//// copyto包含了create一步。所以不需要初始化大
    observation_num=0;
    create_id=-1;
}

void mappoint::AddObservation(frame *FM, size_t idx)
{
    if(observations.count(FM))
        return;
    observations[FM]=idx;
    observation_num++;
}


cv::Mat mappoint::ComputeDistinctiveDescriptors() {

    cv::Mat best_descriptor;
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    vDescriptors.reserve(observations.size());
    cout << "观测次数:" << observations.size() << endl;

    for (map<frame *, int>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
        frame *fm = mit->first;
        vDescriptors.push_back(fm->f_descriptor.row(mit->second));
    }

    if (vDescriptors.empty())
        return cv::Mat();

    // Compute distances between them
    const size_t N = vDescriptors.size();

    float Distances[N][N];
    for (size_t i = 0; i < N; i++) {
        Distances[i][i] = 0;
        for (size_t j = i + 1; j < N; j++) {
            int distij = pnpmatch::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i][j] = distij;
            Distances[j][i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for (size_t i = 0; i < N; i++) {
        vector<int> vDists(Distances[i], Distances[i] + N);
        sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];

        if (median < BestMedian) {
            BestMedian = median;
            BestIdx = i;
        }
    }
    best_descriptor = vDescriptors[BestIdx].clone();
    cout << best_descriptor << endl;

    return best_descriptor;
}

vector<cv::Mat> mappoint::GetDescriptors()
{
    cv::Mat best_descriptor;
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    vDescriptors.reserve(observations.size());

    for (map<frame *, int>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++) {
        frame *fm = mit->first;
        vDescriptors.push_back(fm->f_descriptor.row(mit->second));
    }
    return vDescriptors;

}

cv::Mat mappoint::GetWorldPos()
{
    return worldpos.clone();
}

