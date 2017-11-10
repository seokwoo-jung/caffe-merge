#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/dense_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
template <typename Dtype>
DenseImageDataLayer<Dtype>::~DenseImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void DenseImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.dense_image_data_param().new_height();
  const int new_width  = this->layer_param_.dense_image_data_param().new_width();
  const int crop_height = this->layer_param_.dense_image_data_param().crop_height();
  const int crop_width  = this->layer_param_.dense_image_data_param().crop_width();
  const bool is_color  = this->layer_param_.dense_image_data_param().is_color();
  string root_folder = this->layer_param_.dense_image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  CHECK((crop_height == 0 && crop_width == 0) ||
      (crop_height > 0 && crop_width > 0)) << "Current implementation requires "
      "crop_height and crop_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.dense_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  string label_filename;
  while (infile >> filename >> label_filename) {
    lines_.push_back(std::make_pair(filename, label_filename));
  }


  // Read an image, and use it to initialize the top blobs.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  // sanity check label image
  cv::Mat cv_lab = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                                    new_height, new_width, false);
  CHECK(cv_lab.channels() == 1) << "Can only handle grayscale label images";
  CHECK(cv_lab.rows == height && cv_lab.cols == width) << "Input and label "
      << "image heights and widths must match";



  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.dense_image_data_param().batch_size();



  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    for(int i = 0; i < this->PREFETCH_COUNT; ++i)
    {
        this->prefetch_[i].data_.Reshape(batch_size, channels, crop_size, crop_size);
        this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
    }
    // similarly reshape label data blobs
    top[1]->Reshape(batch_size, 1, crop_size, crop_size);
    for(int i = 0; i < this->PREFETCH_COUNT; ++i)
    {
        this->prefetch_[i].label_.Reshape(batch_size, 1, crop_size, crop_size);
        this->transformed_label_.Reshape(1, 1, crop_size, crop_size);
    }
  } else if (crop_height > 0 && crop_width > 0) {
    top[0]->Reshape(batch_size, channels, crop_height, crop_width);
    for(int i = 0; i < this->PREFETCH_COUNT; ++i)
    {
        this->prefetch_[i].data_.Reshape(batch_size, channels, crop_height, crop_width);
        this->transformed_data_.Reshape(1, channels, crop_height, crop_width);
    }
    // similarly reshape label data blobs
    top[1]->Reshape(batch_size, 1, crop_height, crop_width);
    for(int i = 0; i < this->PREFETCH_COUNT; ++i)
    {
        this->prefetch_[i].label_.Reshape(batch_size, 1, crop_height, crop_width);
        this->transformed_label_.Reshape(1, 1, crop_height, crop_width);
    }
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    for(int i = 0; i < this->PREFETCH_COUNT; ++i)
    {
        this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
        this->transformed_data_.Reshape(1, channels, height, width);
    }
    // similarly reshape label data blobs
    top[1]->Reshape(batch_size, 1, height, width);
    for(int i = 0; i < this->PREFETCH_COUNT; ++i)
    {
        this->prefetch_[i].label_.Reshape(batch_size, 1, height, width);
        this->transformed_label_.Reshape(1, 1, height, width);
    }
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}


template <typename Dtype>
void DenseImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void DenseImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
{
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  DenseImageDataParameter dense_image_data_param = this->layer_param_.dense_image_data_param();
  const int batch_size = dense_image_data_param.batch_size();
  const int new_height = dense_image_data_param.new_height();
  const int new_width = dense_image_data_param.new_width();
  const int crop_height = dense_image_data_param.crop_height();
  const int crop_width = dense_image_data_param.crop_width();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = dense_image_data_param.is_color();
  string root_folder = dense_image_data_param.root_folder();

  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);

  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();


  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
      // get a blob
      timer.Start();
      CHECK_GT(lines_size, lines_id_);
      cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
          new_height, new_width, is_color);
      CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
      cv::Mat cv_lab = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
          new_height, new_width, false);
      CHECK(cv_lab.data) << "Could not load " << lines_[lines_id_].second;
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply random horizontal mirror of images
      if (this->layer_param_.dense_image_data_param().mirror()) {
        const bool do_mirror = caffe_rng_rand() % 2;
        if (do_mirror) {
          cv::flip(cv_img,cv_img,1);
          cv::flip(cv_lab,cv_lab,1);
        }
      }

      // Apply crop
      int height = cv_img.rows;
      int width = cv_img.cols;

      int h_off = 0;
      int w_off = 0;
      if (crop_height>0 && crop_width>0) {
        h_off = caffe_rng_rand() % (height - crop_height + 1);
        w_off = caffe_rng_rand() % (width - crop_width + 1);
        cv::Rect myROI(w_off, h_off, crop_width, crop_height);
        cv_img = cv_img(myROI);
        cv_lab = cv_lab(myROI);
      }
      // Apply transformations (mirror, crop...) to the image
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      // transform label the same way

      int label_offset = batch->label_.offset(item_id);
      this->transformed_label_.set_cpu_data(prefetch_label + label_offset);
      this->data_transformer_->Transform(cv_lab, &this->transformed_label_);
      CHECK(!this->layer_param_.transform_param().mirror() &&
          this->layer_param_.transform_param().crop_size() == 0)
          << "FIXME: Any stochastic transformation will break layer due to "
          << "the need to transform input and label images in the same way";
      trans_time += timer.MicroSeconds();

      // go to the next iter
      lines_id_++;
      if (lines_id_ >= lines_size) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        lines_id_ = 0;
        if (this->layer_param_.dense_image_data_param().shuffle()) {
          ShuffleImages();
        }
      }
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }
    INSTANTIATE_CLASS(DenseImageDataLayer);
    REGISTER_LAYER_CLASS(DenseImageData);
}

#endif


