// Opencv lib
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//NCNN lib
#include "net.h"

// Standard lib
#include <iostream>


ncnn::Mat randn_mat(int weight, int height, int channels, int seed)
{
    cv::Mat cv_x(cv::Size(weight, height), CV_32FC(channels));
    cv::RNG rng(seed);
    rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);
    ncnn::Mat x_mat(weight, height, channels, (void*)cv_x.data);
    return x_mat.clone();
}

void visualize(const char* title, const ncnn::Mat& m)
{
    std::vector<cv::Mat> normed_feats(m.c);

    for (int i = 0; i < m.c; i++)
    {
        cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));

        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

        // check NaN
        for (int y = 0; y < m.h; y++)
        {
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x = 0; x < m.w; x++)
            {
                float v = tp[x];
                if (v != v)
                {
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }

                sp += 3;
            }
        }
    }

    int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
    int th = (m.c - 1) / tw + 1;

    cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
    show_map = cv::Scalar(127);

    // tile
    for (int i = 0; i < m.c; i++)
    {
        int ty = i / tw;
        int tx = i % tw;

        normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
    }

    cv::resize(show_map, show_map, cv::Size(0, 0), 2, 2, cv::INTER_NEAREST);
    cv::imshow(title, show_map);
}


void pretty_print(const ncnn::Mat& m)
{
    for (int q = 0; q < m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int z = 0; z < m.d; z++)
        {
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");

    }
    printf("Matric shape: [%d, %d, %d]\n", m.c, m.h, m.w);
}

void shape(cv::Mat& image)
{
    int rows = image.rows;
    int cols = image.cols;
    printf("Image's shape: [%d, %d, %d]\n", image.channels(), rows, cols);
    // std::cout << "Rows: " << rows << " columns: " << cols << " Channels: " << image.channels() << std::endl;
}

void shape(ncnn::Mat& m)
{
    int h = m.h;
    int w = m.w;
    int c = m.c;

    printf("Matric shape: [%d, %d, %d]\n", c, h, w);
}

void post_process_img(ncnn::Mat & img)
{
    const float _mean_[3] = { -1.0f, -1.0f, -1.0f };
    const float _norm_[3] = { 127.5f, 127.5f, 127.5f };
    img.substract_mean_normalize(_mean_, _norm_);
}
