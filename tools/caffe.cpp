#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, -1,
    "The number of iterations to run.");
DEFINE_string(output, "output.txt",
    "The output file name.");
DEFINE_string(filelist, "file_list.txt",
    "The list of images to process.");

// A simple registry for caffe commands.
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin();
         it != g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;  // not reachable, just to suppress old compiler warnings.
  }
}

// caffe commands to call by
//     caffe <command> <args>
//
// To add a command, define a function "int command()" and register it with
// RegisterBrewFunction(action);

// Device Query: show diagnostic information for a GPU device.
int device_query() {
  CHECK_GT(FLAGS_gpu, -1) << "Need a device ID to query.";
  LOG(INFO) << "Querying device ID = " << FLAGS_gpu;
  caffe::Caffe::SetDevice(FLAGS_gpu);
  caffe::Caffe::DeviceQuery();
  return 0;
}
RegisterBrewFunction(device_query);


// Train / Finetune a model.
int train() {
  CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(FLAGS_solver, &solver_param);

  // If the gpu flag is not provided, allow the mode and device to be set
  // in the solver prototxt.
  if (FLAGS_gpu < 0
      && solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
    FLAGS_gpu = solver_param.device_id();
  }

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  LOG(INFO) << "Starting Optimization";
  shared_ptr<caffe::Solver<float> >
    solver(caffe::GetSolver<float>(solver_param));

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Solve(FLAGS_snapshot);
  } else if (FLAGS_weights.size()) {
    LOG(INFO) << "Finetuning from " << FLAGS_weights;
    solver->net()->CopyTrainedLayersFrom(FLAGS_weights);
    solver->Solve();
  } else {
    solver->Solve();
  }
  LOG(INFO) << "Optimization Done.";
  return 0;
}
RegisterBrewFunction(train);


// Test: score a model.
int test() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

  std::ifstream instream(FLAGS_filelist.c_str());
  std::ofstream outstream(FLAGS_output.c_str());
  outstream.close();
  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  vector<std::string> file_list;
  std::string name;
  int lab;

  std::string line;
  vector<std::string> imageID;

  std::ofstream ostream("Catalog_list.txt");

  while (instream >> line)
  {
	  int p = line.find(',',0);
	  int p2 = line.find(',',p+1);

	  //imageID.push_back(line.substr(p+1,p2-p-1));

	  imageID.push_back(line.substr(0,p));
	  name = line.substr(p2+1,*line.end()-p2);
	  file_list.push_back(name);

	  ostream << name << " 0 \n";
	  //std::cout << name << std::endl;

  }


//  while (instream >> name >> lab)
//	  file_list.push_back(name);
  instream.close();
  ostream.close();
  std::cout << "number of files: " << file_list.size() << std::endl;


  // Instantiate the caffe net.
  Caffe::set_phase(Caffe::TEST);
  Net<float> caffe_net(FLAGS_model);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  if (FLAGS_iterations < 0)
  {
	  int b_size = caffe_net.blob_by_name("labels")->num();
	  FLAGS_iterations = file_list.size()/b_size;
  }
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;

  Timer time, total_time;

  total_time.Start();

  for (int i = 0; i < FLAGS_iterations; ++i) {
	time.Start();
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(bottom_vec, &iter_loss);
    loss += iter_loss;
    int idx = 0;
    std::cout << "iteration: " << i << std::endl;
    //std::cout << "iteration result size: " << result.size() << std::endl;
    //std::cout << "result[0] size: " << result[0]->count() << std::endl;
    //std::cout << "result[1] size: " << result[1]->num() << " " << result[1]->channels() << " " << result[1]->width() << " "<< " " << result[1]->height() << std::endl;

    //shared_ptr<Blob<float> > probBlob = caffe_net.blob_by_name("prob");
    shared_ptr<Blob<float> > labelsBlob = caffe_net.blob_by_name("labels");

    //std::cout << "prob Blob size: " << probBlob->num() << " " << probBlob->channels() << " " << probBlob->width() << " "<< " " << probBlob->height() << std::endl;
    std::cout << "labels Blob size: " << labelsBlob->num() << " " << labelsBlob->channels() << " " << labelsBlob->width() << " "<< " " << labelsBlob->height() << std::endl;

    outstream.open(FLAGS_output.c_str(),std::ios::app);

    for (int j=0; j < labelsBlob->num(); ++j)
    {

    	std::string file_name = file_list[j+labelsBlob->num()*i];

    	std::string ID = imageID[j+labelsBlob->num()*i];

    	outstream << file_name << ", " << ID << ", ";

    	for (int k=0; k < labelsBlob->height(); ++k)
    		outstream << labelsBlob->data_at(j,0,k,0) << ", " << labelsBlob->data_at(j,1,k,0) << ", ";
    	outstream << std::endl;
    }
    outstream.close();

    std::cout << time.Seconds()  << " Seconds \n";
    time.Stop();
//
//    const float* result_vec = result[1]->cpu_data();
//    for (int k = 0; k < result[1]->count(); ++k, ++idx) {
//
//    	const float score = result_vec[k];
//    	//LOG(INFO) << score;
//    	if (i == 0) {
//    		test_score.push_back(score);
//    		test_score_output_id.push_back(1);
//    	} else {
//    		test_score[idx] += score;
//    	}
//    	const std::string& output_name = caffe_net.blob_names()[
//																caffe_net.output_blob_indices()[1]];
//    	//LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
//    }
   /* for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
    */
  }

  LOG(INFO) << "Total time: " << total_time.Seconds() << " Seconds";

  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight =
        caffe_net.blob_loss_weights()[caffe_net.output_blob_indices()[i]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  return 0;
}
RegisterBrewFunction(test);


// Time: benchmark the execution time of a model.
int time() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to time.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Caffe::set_phase(Caffe::TRAIN);
  Net<float> caffe_net(FLAGS_model);

  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.
  float initial_loss;
  caffe_net.Forward(vector<Blob<float>*>(), &initial_loss);
  LOG(INFO) << "Initial loss: " << initial_loss;
  LOG(INFO) << "Performing Backward";
  caffe_net.Backward();

  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  vector<vector<Blob<float>*> >& bottom_vecs = caffe_net.bottom_vecs();
  vector<vector<Blob<float>*> >& top_vecs = caffe_net.top_vecs();
  const vector<vector<bool> >& bottom_need_backward =
      caffe_net.bottom_need_backward();
  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  Timer total_timer;
  total_timer.Start();
  Timer forward_timer;
  forward_timer.Start();
  Timer timer;
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    timer.Start();
    for (int j = 0; j < FLAGS_iterations; ++j) {
      // Although Reshape should be essentially free, we include it here
      // so that we will notice Reshape performance bugs.
      layers[i]->Reshape(bottom_vecs[i], &top_vecs[i]);
      layers[i]->Forward(bottom_vecs[i], &top_vecs[i]);
    }
    LOG(INFO) << layername << "\tforward: " << timer.MilliSeconds() <<
        " milliseconds.";
  }
  LOG(INFO) << "Forward pass: " << forward_timer.MilliSeconds() <<
      " milliseconds.";
  Timer backward_timer;
  backward_timer.Start();
  for (int i = layers.size() - 1; i >= 0; --i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    timer.Start();
    for (int j = 0; j < FLAGS_iterations; ++j) {
      layers[i]->Backward(top_vecs[i], bottom_need_backward[i],
                          &bottom_vecs[i]);
    }
    LOG(INFO) << layername << "\tbackward: "
        << timer.MilliSeconds() << " milliseconds.";
  }
  LOG(INFO) << "Backward pass: " << backward_timer.MilliSeconds() <<
      " milliseconds.";
  LOG(INFO) << "Total Time: " << total_timer.MilliSeconds() <<
      " milliseconds.";
  LOG(INFO) << "*** Benchmark ends ***";
  return 0;
}
RegisterBrewFunction(time);

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: caffe <command> <args>\n\n"
      "commands:\n"
      "  train           train or finetune a model\n"
      "  test            score a model\n"
      "  device_query    show GPU diagnostic information\n"
      "  time            benchmark model execution time");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/caffe");
  }
}
