/*
 * optimizers.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include "utils.hpp"

#define ROUND_SMALL_TO_ZERO(x, eps) if ((x) > -eps && (x) < eps) x = 0.0f;
namespace
{
	using namespace avocado::backend;

	template<typename T>
	T round_small_to_zero(T x) noexcept
	{
		return (abs(x) < eps<T>()) ? zero<T>() : x;
	}

	template<typename T>
	void kernel_learn_sgd(T *weight, const T *update, T *momentum, avSize_t elements, T learning_rate, T beta, bool use_momentum, bool use_nesterov)
	{
		for (avSize_t i = 0; i < elements; i++)
		{
			T tmp;
			if (use_momentum)
			{
				momentum[i] = beta * momentum[i] - learning_rate * update[i];
				if (use_nesterov)
					tmp = beta * momentum[i] - learning_rate * update[i];
				else
					tmp = momentum[i];
			}
			else
				tmp = -learning_rate * update[i];

			weight[i] = round_small_to_zero(weight[i] + tmp);
		}
	}
	template<typename T>
	void kernel_learn_adam(T *weight, const T *update, T *momentum, T *variance, avSize_t elements, T learning_rate, T beta1, T beta2)
	{
		for (avSize_t i = 0; i < elements; i++)
		{
			momentum[i] = momentum[i] * beta1 + update[i] * (one<T>() - beta1);
			variance[i] = variance[i] * beta2 + update[i] * update[i] * (one<T>() - beta2);
			T tmp = -momentum[i] * learning_rate / sqrt(variance[i] + eps<T>());

			weight[i] = round_small_to_zero(weight[i] + tmp);
		}
	}

//	avStatus_t sgd_helper(const avOptimizer_t optimizer, avTensor_t weight, const avTensor_t update, avTensor_t workspace1)
//	{
//		avSize_t elements = volume(weight);
//		bool use_momentum = optimizer->flags[0];
//		bool use_nesterov = optimizer->flags[1];
//
//		switch (weight->dtype)
//		{
//			case AVOCADO_DTYPE_FLOAT32:
//			{
//				float beta = optimizer->coef[0];
//				float learning_rate = optimizer->learning_rate;
//				float *momentum = (workspace1 == nullptr) ? nullptr : data<float>(workspace1);
//				kernel_learn_sgd(data<float>(weight), data<float>(update), momentum, elements, learning_rate, beta, use_momentum, use_nesterov);
//				break;
//			}
//			case AVOCADO_DTYPE_FLOAT64:
//			{
//				double beta = optimizer->coef[0];
//				double learning_rate = optimizer->learning_rate;
//				double *momentum = (workspace1 == nullptr) ? nullptr : data<double>(workspace1);
//				kernel_learn_sgd(data<double>(weight), data<double>(update), momentum, elements, learning_rate, beta, use_momentum, use_nesterov);
//				break;
//			}
//			default:
//				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//		}
//		return AVOCADO_STATUS_SUCCESS;
//	}
//	avStatus_t adam_helper(const avOptimizer_t optimizer, avTensor_t weight, const avTensor_t update, avTensor_t workspace1, avTensor_t workspace2)
//	{
//		avSize_t elements = volume(weight);
//		switch (weight->dtype)
//		{
//			case AVOCADO_DTYPE_FLOAT32:
//			{
//				float beta1 = optimizer->coef[0];
//				float beta2 = optimizer->coef[1];
//				float learning_rate = optimizer->learning_rate;
//				kernel_learn_adam(data<float>(weight), data<float>(update), data<float>(workspace1), data<float>(workspace2), elements, learning_rate,
//						beta1, beta2);
//				break;
//			}
//			case AVOCADO_DTYPE_FLOAT64:
//			{
//				double beta1 = optimizer->coef[0];
//				double beta2 = optimizer->coef[1];
//				double learning_rate = optimizer->learning_rate;
//				kernel_learn_adam(data<double>(weight), data<double>(update), data<double>(workspace1), data<double>(workspace2), elements,
//						learning_rate, beta1, beta2);
//				break;
//			}
//			default:
//				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//		}
//		return AVOCADO_STATUS_SUCCESS;
//	}
}

namespace avocado
{
	namespace backend
	{
//		avStatus_t refOptimizerLearn(avContext_t context, const avOptimizer_t optimizer, avTensor_t weight, const avTensor_t update,
//				avTensor_t workspace1, avTensor_t workspace2)
//		{
//			assert(context != nullptr);
//			assert(optimizer != nullptr);
//			assert(weight != nullptr);
//			assert(update != nullptr);
//
//			switch (optimizer->type)
//			{
//				case AVOCADO_OPTIMIZER_SGD:
//					return sgd_helper(optimizer, weight, update, workspace1);
//				case AVOCADO_OPTIMIZER_ADAM:
//					return adam_helper(optimizer, weight, update, workspace1, workspace2);
//				default:
//					return AVOCADO_STATUS_BAD_PARAM;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
	} /* namespace backend */
} /* namespace avocado */

