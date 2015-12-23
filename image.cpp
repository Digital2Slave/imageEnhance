//
//  image.cpp
//  imageEnhance
//
//  Created by tianzhaixing on 15/12/2.
//  Copyright © 2015年 beautifulreading. All rights reserved.
//

#include "image.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void calhistOfRGB(Mat& src)
{
    
    Mat dst;
    
    //分割图像为3个通道即：B, G and R
    vector<Mat> bgr_planes;
    split( src, bgr_planes );
        
    //创建箱子的数目
    int histSize = 256;
        
    //设置范围 ( for B,G,R) )
    float range[] = { 0, 256 } ;//不包含上界256
    const float* histRange = { range };
        
    //归一化，起始位置直方图清除内容
    bool uniform = true; bool accumulate = false;
        
    Mat b_hist, g_hist, r_hist;
        
    //计算每个平面的直方图
    //&bgr_planes[]原数组，1原数组个数，0只处理一个通道，
    //Mat()用于处理原来数组的掩膜，b_hist将要用来存储直方图的Mat对象
    //1直方图的空间尺寸，histsize每一维的箱子数目，histrange每一维的变化范围
    //uniform和accumulate箱子的大小一样，直方图开始的位置清除内容
        
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
        
    //画直方图（ B, G and R）
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
        
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
        
    //归一化结果为 [ 0, histImage.rows ]
    //b_hist输入数组，b_hist输出数组，
    //0和histImage.rows归一化的两端限制值，
    //NORM_MINMAX归一化类型 -1输出和输入类型一样，Mat()可选掩膜
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
        
    //为每个通道画图
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                 Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                 Scalar( 255, 0, 0), 2, 8, 0
             );
        
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                 Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                Scalar( 0, 255, 0), 2, 8, 0
             );
        
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                 Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                 Scalar( 0, 0, 255), 2, 8, 0
             );
    }
        
    //显示输出结果
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histImage );
        
    waitKey(10);
    
}

float GetGamma(Mat& src)
{
    CV_Assert(src.data);
    CV_Assert(src.depth() != sizeof(uchar));
    
    int height = src.rows;
    int width  = src.cols;
    long size  = height * width;
        
    //!< histogram
    float histogram[256] = {0};
    uchar pvalue = 0;
    MatIterator_<uchar> it, end;
    for( it = src.begin<uchar>(), end = src.end<uchar>(); it != end; it++ )
    {
        pvalue = (*it);
        histogram[pvalue]++;
        
    }
    
    int threshold = 0;       //otsu阈值
    long sum0 = 0, sum1 = 0; //前景的灰度总和和背景灰度总和
    long cnt0 = 0, cnt1 = 0; //前景的总个数和背景的总个数
    
    double w0 = 0, w1 = 0;   //前景和背景所占整幅图像的比例
    double u0 = 0, u1 = 0;   //前景和背景的平均灰度
    double u = 0;            //图像总平均灰度
    double variance = 0;     //前景和背景的类间方差
    double maxVariance = 0;  //前景和背景的最大类间方差
    
    int i, j;
    for(i = 1; i < 256; i++) //一次遍历每个像素
    {
        sum0 = 0;
        sum1 = 0;
        cnt0 = 0;
        cnt1 = 0;
        w0   = 0;
        w1   = 0;
        for(j = 0; j < i; j++)
        {
            cnt0 += histogram[j];
            sum0 += j * histogram[j];
        }
        
        u0 = (double)sum0 /  cnt0;
        w0 = (double)cnt0 / size;
        
        for(j = i ; j <= 255; j++)
        {
            cnt1 += histogram[j];
            sum1 += j * histogram[j];
        }
        
        u1 = (double)sum1 / cnt1;
        w1 = 1 - w0;                 // (double)cnt1 / size;
        
        u = u0 * w0 + u1 * w1;
        
        //variance =  w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
        variance =  w0 * w1 *  (u0 - u1) * (u0 - u1);
        
        if(variance > maxVariance) 
        {  
            maxVariance = variance;  
            threshold = i;  
        } 
    }
    
    // convert threshold to gamma.
    float gamma = 0.0;
    gamma = threshold/255.0;

    // return
    return gamma;
}



void GammaCorrection(Mat& src, Mat& dst, float fGamma)
{
    CV_Assert(src.data);
    
    // accept only char type matrices
    CV_Assert(src.depth() != sizeof(uchar));
    
    // build look up table
    unsigned char lut[256];
    for( int i = 0; i < 256; i++ )
    {
        lut[i] = saturate_cast<uchar>(pow((float)(i/255.0), fGamma) * 255.0f);
    }
    
    // case 1 and 3 for different channels
    dst = src.clone();
    const int channels = dst.channels();
    switch(channels)
    {
        case 1:
        {
            
            MatIterator_<uchar> it, end;
            for( it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++ )
                *it = lut[(*it)];
            
            break;
        }
        case 3:
        {
            
            MatIterator_<Vec3b> it, end;
            for( it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++ )
            {
                (*it)[0] = lut[((*it)[0])]; // B
                (*it)[1] = lut[((*it)[1])]; // G
                (*it)[2] = lut[((*it)[2])]; // R
            }
            break;
            
        }
    } // end for switch
}



int main(int argc, char* argv[])
{
    
    const string s = "./test.jpg";
    
    Mat img = imread(s);
    
    if (!img.data)
    {
        return -1;
    }
    
    int height = img.rows;
    int width  = img.cols;
    Mat simg;
    
    resize(img, simg, Size(height/4, width/4));
    
    // R,G,B 通道分析
    calhistOfRGB(simg);
    
    // Gray
    Mat Grayimg;
    cvtColor(simg, Grayimg, COLOR_BGR2GRAY);
    
    float gamma = GetGamma(Grayimg);
    cout << "gamma is " << gamma << endl;
    
    Mat dst;
    GammaCorrection(simg, dst, gamma);
    
    // Show
    imshow("src",   simg);
    imshow("dst",   dst);
    
    waitKey(0);
    
    return 0;
}