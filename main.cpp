// Opencv lib
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

//NCNN lib
#include "net.h"

// Standard lib
#include <iostream>
#include <time.h>
//common
#include "utils.h"


int main() 
{
    srand(time(NULL));
	// Load network
	ncnn::Net progan;
	progan.load_param("model/script_pro.ncnn.param");
	progan.load_model("model/script_pro.ncnn.bin");
	printf("Loaded");

	// Generate noise
    ncnn::Mat noise = randn_mat(32, 16, 1, rand());
    noise = noise.reshape(1, 1, 512);
    shape(noise);
  
    // Forward
    ncnn::Extractor ex = progan.create_extractor();
    ex.input("in0", noise);
    ncnn::Mat outputs;
    ex.extract("out0", outputs);
    printf("Output's shape");
    shape(outputs);
    post_process_img(outputs);

    // Display the generated images
    cv::Mat cv_img = cv::Mat::zeros(outputs.w, outputs.h, CV_8UC3);
    outputs.to_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB);
    shape(cv_img);
    cv::imshow("Testing", cv_img);
    cv::waitKey(0);
    return 0;
}