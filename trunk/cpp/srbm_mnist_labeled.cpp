#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "../tensor/apex_tensor.h"
#include "../utils/data_set/apex_mnist_iterator.h"
#include "../utils/data_set/apex_mnist_label_iterator.h"
#include "../utils/task/apex_tensor_update_task.h"
#include "../srbm/apex_srbm_trainer.h"

using namespace apex_utils;

int main(int argc, char *argv[]) {
	apex_tensor::init_tensor_engine_cpu(10);
	MNISTIterator<apex_tensor::CTensor2D> data_iter;
	MNISTLabelIterator<apex_tensor::CTensor2D> label_iter;
	apex_rbm::SRBMTrainer trainer;
	TensorLabeledUpdateTask<apex_tensor::CTensor2D> tsk(&trainer, &label_iter, &data_iter);
	return run_task(argc, argv, &tsk);
}
