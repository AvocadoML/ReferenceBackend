/*
 * optimizers.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

#include "utils.hpp"

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
			T tmp = -momentum[i] * learning_rate / std::sqrt(variance[i] + eps<T>());

			weight[i] = round_small_to_zero(weight[i] + tmp);
		}
	}

	avStatus_t sgd_helper(const OptimizerDescriptor &optimizer, const TensorDescriptor &weightDesc, MemoryDescriptor &weight,
			const MemoryDescriptor &update, MemoryDescriptor &workspace)
	{
		const avSize_t elements = weightDesc.volume();
		bool use_momentum = optimizer.flags[0];
		bool use_nesterov = optimizer.flags[1];

		switch (weightDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				float beta = optimizer.coef[0];
				float learning_rate = optimizer.learning_rate;
				float *momentum = use_momentum ? nullptr : workspace.data<float>();
				kernel_learn_sgd(weight.data<float>(), update.data<float>(), momentum, elements, learning_rate, beta, use_momentum, use_nesterov);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				double beta = optimizer.coef[0];
				double learning_rate = optimizer.learning_rate;
				double *momentum = use_momentum ? nullptr : workspace.data<double>();
				kernel_learn_sgd(weight.data<double>(), update.data<double>(), momentum, elements, learning_rate, beta, use_momentum, use_nesterov);
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
	avStatus_t adam_helper(const OptimizerDescriptor &optimizer, const TensorDescriptor &wDesc, MemoryDescriptor &weight,
			const MemoryDescriptor &update, MemoryDescriptor &workspace)
	{
		const avSize_t elements = wDesc.volume();

		if (workspace.size() < 2 * elements * dataTypeSize(wDesc.dtype()))
			return AVOCADO_STATUS_INTERNAL_ERROR;
		switch (wDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				float beta1 = optimizer.coef[0];
				float beta2 = optimizer.coef[1];
				float learning_rate = optimizer.learning_rate;
				kernel_learn_adam(weight.data<float>(), update.data<float>(), workspace.data<float>(), workspace.data<float>() + elements, elements,
						learning_rate, beta1, beta2);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				double beta1 = optimizer.coef[0];
				double beta2 = optimizer.coef[1];
				double learning_rate = optimizer.learning_rate;
				kernel_learn_adam(weight.data<double>(), update.data<double>(), workspace.data<double>(), workspace.data<double>() + elements,
						elements, learning_rate, beta1, beta2);
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const avTensorDescriptor_t wDesc,
				avMemoryDescriptor_t wMem, const avTensorDescriptor_t dwDesc, const avMemoryDescriptor_t dwMem, avMemoryDescriptor_t workspace)
		{
			switch (getOptimizer(config).type)
			{
				case AVOCADO_OPTIMIZER_SGD:
					return sgd_helper(getOptimizer(config), getTensor(wDesc), getMemory(wMem), getMemory(dwMem), getMemory(workspace));
				case AVOCADO_OPTIMIZER_ADAM:
					return adam_helper(getOptimizer(config), getTensor(wDesc), getMemory(wMem), getMemory(dwMem), getMemory(workspace));
				default:
					return AVOCADO_STATUS_BAD_PARAM;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

