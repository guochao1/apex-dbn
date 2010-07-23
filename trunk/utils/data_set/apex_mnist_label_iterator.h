#ifndef _APEX_MNIST_LABEL_ITERATOR_H_
#define _APEX_MNIST_LABEL_ITERATOR_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../apex_utils.h"
#include "../apex_tensor_iterator.h"

// gavinhu
namespace apex_utils {
	template<typename T>
	inline void __MNIST_set_param(T &m, int y_max, int x_max, size_t pitch);

	template<>
	inline void __MNIST_set_param<apex_tensor::CTensor2D>(apex_tensor::CTensor2D &m, int y_max, int x_max, size_t pitch) {
		m.y_max = y_max; m.x_max = x_max; m.pitch = pitch;
	}
	template<>
	inline void __MNIST_set_param<apex_tensor::CTensor3D>(apex_tensor::CTensor3D &m, int y_max, int x_max, size_t pitch) {
		m.z_max = y_max; m.y_max = 1; m.x_max = x_max; m.pitch = pitch;
	}
	template<>
	inline void __MNIST_set_param<apex_tensor::CTensor4D>(apex_tensor::CTensor4D &m, int y_max, int x_max, size_t pitch) {
		m.h_max = y_max; m.z_max = 1; m.y_max = 1; m.x_max = x_max; m.pitch = pitch;
	}

	// iterator that iterates over the MNIST label set.
	template<typename T>
	class MNISTLabelIterator: public ITensorIterator<T> {
	private:
		int idx, max_idx;
		int pitch;
		int num_labels;
		int trunk_size;

		apex_tensor::CTensor2D labels;
	private:
		char name_label_set[256];

		const T get_trunk(int start_idx, int max_idx) const {
			int y_max = max_idx - start_idx;
			if (y_max > trunk_size) y_max = trunk_size;
			T m;
			m.elem = labels[start_idx].elem;
			__MNIST_set_param<T>(m, y_max, labels.x_max, labels.pitch);
			return m;
		}

	public:
		MNISTLabelIterator() {
			labels.elem = NULL;
			max_idx = 1 << 30;
		}

		virtual ~MNISTLabelIterator() {
			if (labels.elem != NULL)
				delete[] labels.elem;
		}

		virtual void set_param(const char *name, const char *val) {
			if (!strcmp(name, "label_set")) strcpy(name_label_set, val);
			if (!strcmp(name, "image_amount")) max_idx = atoi(val);
			if (!strcmp(name, "trunk_size")) trunk_size = atoi(val);
		}

		// initialize the model
		virtual void init(void) {
			FILE *fi = apex_utils::fopen_check(name_label_set, "rb");
			unsigned char zz[4];
			unsigned char *t_data;

			if (fread(zz, 4, 1, fi) == 0) {
				apex_utils::error("load MNIST label");
			}

			if (fread(zz, 4, 1, fi) == 0) {
				apex_utils::error("load MNIST label");
			}

            num_labels = (int)(zz[3]) 
                | (((int)(zz[2])) << 8)
                | (((int)(zz[1])) << 16)
                | (((int)(zz[0])) << 24);

			t_data = new unsigned char[num_labels];
			if (fread(t_data, num_labels, 1, fi) == 0) {
				apex_utils::error("load MNIST label");
			}

			fclose(fi);

			labels.set_param(num_labels, 10);
			labels.pitch = 10 * sizeof(apex_tensor::TENSOR_FLOAT);
			labels.elem = new apex_tensor::TENSOR_FLOAT[num_labels*10];

			for (int i = 0; i < num_labels; i++) {
				labels[i] = 0.0f;
				labels[i][t_data[i]] = 1.0f;
			}

			delete[] t_data;
			if (max_idx > labels.y_max)
				max_idx = labels.y_max;
		}

		// move to next mat
		virtual bool next_trunk() {
			idx += trunk_size;
			if (idx >= max_idx) return false;
			return true;
		}

		// get current matrix
		virtual const T trunk() const {
			return get_trunk((int)idx, (int)max_idx);
		}

		// set before first of the item
		virtual void before_first() {
			idx = -trunk_size;
		}

		// trunk used for validation
		virtual const T validation_trunk() const {
			return get_trunk((int)max_idx, num_labels);
		}
	};
};

#endif
