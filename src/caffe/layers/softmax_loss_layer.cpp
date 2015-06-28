#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	softmax_bottom_vec_.clear();
	softmax_bottom_vec_.push_back(bottom[0]);
	softmax_top_vec_.clear();
	softmax_top_vec_.push_back(&prob_);
	softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	LossLayer<Dtype>::Reshape(bottom, top);
	softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
	if (top->size() >= 2) {
		// softmax output
		(*top)[1]->ReshapeLike(*bottom[0]);
	}
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu_regular(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	// The forward pass computes the softmax prob values.
	softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
	const Dtype* prob_data = prob_.cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	int num = prob_.num();
	int dim = prob_.count() / num;
	int spatial_dim = prob_.height() * prob_.width();
	Dtype loss = 0;
	for (int i = 0; i < num; ++i) {
		for (int j = 0; j < spatial_dim; j++) {
			loss -= log(std::max(prob_data[i * dim +
			                               static_cast<int>(label[i * spatial_dim + j]) * spatial_dim + j],
					Dtype(FLT_MIN)));
		}
	}
	(*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
	if (top->size() == 2) {
		(*top)[1]->ShareData(prob_);
	}
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu_regular(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	if (propagate_down[1]) {
		LOG(FATAL) << this->type_name()
            		   << " Layer cannot backpropagate to label inputs.";
	}
	if (propagate_down[0]) {
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const Dtype* prob_data = prob_.cpu_data();
		caffe_copy(prob_.count(), prob_data, bottom_diff);
		const Dtype* label = (*bottom)[1]->cpu_data();
		int num = prob_.num();
		int dim = prob_.count() / num;
		int spatial_dim = prob_.height() * prob_.width();
		for (int i = 0; i < num; ++i) {
			for (int j = 0; j < spatial_dim; ++j) {
				bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
				            * spatial_dim + j] -= 1;
			}
		}
		// Scale gradient
		const Dtype loss_weight = top[0]->cpu_diff()[0];
		caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
	}
}


// The constructor is made such that the softmax loss layer may have simple functionality of loss = maximal input
// or loss = correct label's input. These options allow finding a desired derivative of the loss w.r.t. input pixels
template <typename Dtype>
SoftmaxWithLossLayer<Dtype>::SoftmaxWithLossLayer(const LayerParameter& param)
: LossLayer<Dtype>(param)
  , softmax_layer_(new SoftmaxLayer<Dtype>(param))
  {
	if (param.has_softmax_loss_behavior())
	{
		m_behavior = param.softmax_loss_behavior();
	} else {
		m_behavior = LayerParameter_SoftmaxLossBehavior_REGULAR;
	}
  }

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	switch (m_behavior)
	{
	case ::caffe::LayerParameter_SoftmaxLossBehavior_REGULAR:
		SoftmaxWithLossLayer<Dtype>::Forward_cpu_regular(bottom, top);
		break;
	case ::caffe::LayerParameter_SoftmaxLossBehavior_ONLY_MAXIMAL:
		SoftmaxWithLossLayer<Dtype>::Forward_cpu_variants(bottom, top, MAX_ONLY);
		break;
	case ::caffe::LayerParameter_SoftmaxLossBehavior_ONLY_CORRECT:
		SoftmaxWithLossLayer<Dtype>::Forward_cpu_variants(bottom, top, CORRECT_ONLY);
		break;
	default:
		LOG(FATAL)  << this->type_name() << " Unknown softmax behavior " ; //<< ::caffe::LayerParameter_SoftmaxBehavior_Name(m_behavior);
	} // switch
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu_variants(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top, const VariantType variantType) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int dim = bottom[0]->count() / bottom[0]->num();
	int spatial_dim = bottom[0]->height() * bottom[0]->width();

	// currently I assume only the simplest settings.
	bool isValidLayer = true;
	if (top->size() != 1) {
		LOG(WARNING) << "SoftMax layer was called with " << top->size() << " outputs and non regular behavior. Setting loss to INF";
		isValidLayer = false;
	} else if (spatial_dim > 1) {
		LOG(WARNING) << "SoftMax layer was called with spatial dimension of  " << spatial_dim << " and non regular behavior. Setting loss to INF";
		isValidLayer = false;
	}
	if (isValidLayer == false){
		(*top)[0]->mutable_cpu_data()[0] = FLT_MAX;
		return;
	}

	const Dtype* label = bottom[1]->cpu_data();
	Dtype loss = 0;
	for (int i = 0; i < num; ++i) {
		switch(variantType)
		{
		case MAX_ONLY:
			// loss = -1 * maximal cahnnel's score
			loss -= *std::max_element(bottom_data + i * dim,  bottom_data + i * dim + channels);
			break;
		case CORRECT_ONLY:
			// loss = -1 * correct label's score
			loss -= bottom_data[i * dim + static_cast<int>(label[i])];
			break;
		default:
			LOG(FATAL) << "unknown softmax variation :" << (int)variantType;
		}
	}
	(*top)[0]->mutable_cpu_data()[0] = loss / num;

	std::cout << "loss: " << (*top)[0]->mutable_cpu_data()[0] << std::endl;
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom) {

	switch (m_behavior)
	{
	case ::caffe::LayerParameter_SoftmaxLossBehavior_REGULAR:
		SoftmaxWithLossLayer<Dtype>::Backward_cpu_regular(top, propagate_down, bottom);
		break;
	case ::caffe::LayerParameter_SoftmaxLossBehavior_ONLY_MAXIMAL:
		SoftmaxWithLossLayer<Dtype>::Backward_cpu_variant(top, propagate_down, bottom, MAX_ONLY);
		break;
	case ::caffe::LayerParameter_SoftmaxLossBehavior_ONLY_CORRECT:
		SoftmaxWithLossLayer<Dtype>::Backward_cpu_variant(top, propagate_down, bottom, CORRECT_ONLY);
		break;
	default:
		LOG(FATAL)  << this->type_name() << " Unknown softmax behavior " ; //<< ::caffe::LayerParameter_SoftmaxBehavior_Name(m_behavior);
	} // switch
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu_variant(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom, const VariantType variantType) {
	if (propagate_down[1]) {
		LOG(FATAL) << this->type_name() << " Layer cannot backpropagate to label inputs.";
	}

	if (propagate_down[0]) {
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const int num = (*bottom)[0]->num();
		const int dim = (*bottom)[0]->count() / num;
		const int channels = (*bottom)[0]->channels();
		const int spatial_dim = (*bottom)[0]->height() * (*bottom)[0]->width();

		caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);

		const Dtype* label = (*bottom)[1]->cpu_data();

		// currently I assume only the simplest settings.
		if (spatial_dim > 1) {
			LOG(WARNING) << "SoftMax layer was called with spatial dimension of  " << spatial_dim << " and non regular behavior. Setting loss to INF";
			return;
		}

		const Dtype dLossdInput = -1*top[0]->cpu_diff()[0]/num;

		for (int i = 0; i < num; ++i) {
			switch(variantType)
			{
			case MAX_ONLY:
				// loss = -1 * maximal cahnnel's score
				bottom_diff[std::max_element(bottom_data + i * dim,  bottom_data + i * dim + channels) - bottom_data] = dLossdInput;
				break;
			case CORRECT_ONLY:
				// loss = -1 * correct label's score
				bottom_diff[i * dim + static_cast<int>(label[i])] = dLossdInput;
				break;
			default:
				LOG(FATAL) << "unknown softmax variation :" << (int)variantType;
			}
		}

	}
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
