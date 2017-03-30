#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

int i = 0;

string savepath = "/home/samuel/verification";

//Save sample image for verification pourposes
void saveImageSample(const sensor_msgs::Image::ConstPtr& rgb,
	      const sensor_msgs::Image::ConstPtr& depth) {
    cv_bridge::CvImagePtr cv_ptr;
    try  {
      cv_ptr = cv_bridge::toCvCopy(depth, depth->encoding);
    }
    catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // Update GUI Window
    cv::imshow("sample", cv_ptr->image);
    cv::Mat sample;
    cv_ptr->image.convertTo(ok,CV_8UC1);
    if (i==0) {
      cv::imwrite(savepath + "sample.png", sample);
      i++;
    }
}

int main(int argc, char **argv) {
  
  ros::init(argc, argv, "image_verification_node");

  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);

  image_transport::SubscriberFilter rgb_subscriber(it, "/camera/rgb/image_rect_color", 10);
  image_transport::SubscriberFilter depth_subscriber(it, "/camera/depth/image_rect", 10);    
  
  //Subscribers synchronization set up
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncPolicy;
  message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), rgb_subscriber, depth_subscriber);
  sync.registerCallback(boost::bind(&saveImageSample, _1, _2));
  
  ROS_INFO("Image verification node ready!");
  ros::spin();

  return 0;
}
