#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
//#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
//#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <stereo_msgs/DisparityImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tf/transform_listener.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  string dataset_path = "/home/samuel/Dropbox/Dissertacao/repo/samples/geometry_dataset/";
  string save_path = "/home/samuel/verification/"
  
  string line;
  stringstream conv;
  ifstream file ((path+"objects.txt").c_str());
  if (file.is_open()) {
  	while(getline (file,line)) {
      for(int i = 0; i < 10; i++){
		conv.str(std::string());
		conv << i;
		Mat img = imread(path+"bag/"+line+"-"+conv.str()+"-rgb.png", CV_LOAD_IMAGE_GRAYSCALE);
		
		vector<KeyPoint> keypoints;
			
		int threshold = 25;
		bool nonmaxSuppression = true;
		cv::FastFeatureDetector fast(threshold, nonmaxSuppression);
		fast.detect(img, keypoints);

		cout << keypoints.size() << endl;
		cv::Mat keypoints_image;
		cv::drawKeypoints(img,keypoints, keypoints_image);
		cv::imwrite(savepath+"bag/"+line+"-"+conv.str()+"-keypoints.png", keypoints_image);
      }
    }
    file.close();
  }

  return 0;
}