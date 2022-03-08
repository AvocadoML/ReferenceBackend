/*
 * optimizers.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

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
	void kernel_learn_sgd(T *weight, const T *update, T *momentum, av_int64 elements, T learning_rate, T beta1, bool use_momentum, bool use_nesterov,
			T alpha, T beta)
	{
		if (beta == zero<T>())
			clear(weight, elements);
		for (av_int64 i = 0; i < elements; i++)
		{
			T tmp;
			if (use_momentum)
			{
				momentum[i] = beta1 * momentum[i] - learning_rate * update[i];
				if (use_nesterov)
					tmp = beta1 * momentum[i] - learning_rate * update[i];
				else
					tmp = momentum[i];
			}
			else
				tmp = -learning_rate * update[i];
			weight[i] = round_small_to_zero(alpha * tmp + beta * weight[i]);
		}
	}
	template<typename T>
	void kernel_learn_adam(T *weight, const T *update, T *momentum, T *variance, av_int64 elements, T learning_rate, T beta1, T beta2, T alpha,
			T beta)
	{
		if (beta == zero<T>())
			clear(weight, elements);
		for (av_int64 i = 0; i < elements; i++)
		{
			momentum[i] = momentum[i] * beta1 + update[i] * (one<T>() - beta1);
			variance[i] = variance[i] * beta2 + update[i] * update[i] * (one<T>() - beta2);
			T tmp = -momentum[i] * learning_rate / std::sqrt(variance[i] + eps<T>());
			weight[i] = round_small_to_zero(alpha * tmp + beta * weight[i]);
		}
	}

	avStatus_t sgd_helper(const reference::OptimizerDescriptor &optimizer, const void *alpha, const reference::MemoryDescriptor &dwMem,
			const void *beta, const reference::TensorDescriptor &wDesc, reference::MemoryDescriptor &wMem, reference::MemoryDescriptor &workspace)
	{
		const av_int64 elements = wDesc.volume();
		bool use_momentum = optimizer.flags[0];
		bool use_nesterov = optimizer.flags[1];

		switch (wDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				float _alpha = reference::getAlphaValue(alpha);
				float _beta = reference::getBetaValue(beta);
				float beta1 = optimizer.coef[0];
				float learning_rate = optimizer.learning_rate;
				float *momentum = use_momentum ? workspace.data<float>() : nullptr;
				kernel_learn_sgd(wMem.data<float>(), dwMem.data<float>(), momentum, elements, learning_rate, beta1, use_momentum, use_nesterov,
						_alpha, _beta);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				double _alpha = reference::getAlphaValue<double>(alpha);
				double _beta = reference::getBetaValue<double>(beta);
				double beta1 = optimizer.coef[0];
				double learning_rate = optimizer.learning_rate;
				double *momentum = use_momentum ? workspace.data<double>() : nullptr;
				kernel_learn_sgd(wMem.data<double>(), dwMem.data<double>(), momentum, elements, learning_rate, beta1, use_momentum, use_nesterov,
						_alpha, _beta);
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
	avStatus_t adam_helper(const reference::OptimizerDescriptor &optimizer, const void *alpha, const reference::MemoryDescriptor &dwMem,
			const void *beta, const reference::TensorDescriptor &wDesc, reference::MemoryDescriptor &wMem, reference::MemoryDescriptor &workspace)
	{
		const av_int64 elements = wDesc.volume();

		if (workspace.size() < 2 * elements * reference::dataTypeSize(wDesc.dtype()))
			return AVOCADO_STATUS_INTERNAL_ERROR;
		switch (wDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				float _alpha = reference::getAlphaValue(alpha);
				float _beta = reference::getBetaValue(beta);
				float beta1 = optimizer.coef[0];
				float beta2 = optimizer.coef[1];
				float learning_rate = optimizer.learning_rate;
				kernel_learn_adam(wMem.data<float>(), dwMem.data<float>(), workspace.data<float>(), workspace.data<float>() + elements, elements,
						learning_rate, beta1, beta2, _alpha, _beta);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				double _alpha = reference::getAlphaValue<double>(alpha);
				double _beta = reference::getBetaValue<double>(beta);
				double beta1 = optimizer.coef[0];
				double beta2 = optimizer.coef[1];
				double learning_rate = optimizer.learning_rate;
				kernel_learn_adam(wMem.data<double>(), dwMem.data<double>(), workspace.data<double>(), workspace.data<double>() + elements, elements,
						learning_rate, beta1, beta2, _alpha, _beta);
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
		avStatus_t refOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t dwDesc, const avMemoryDescriptor_t dwMem, const void *beta, const avTensorDescriptor_t wDesc,
				avMemoryDescriptor_t wMem, avMemoryDescriptor_t workspace)
		{
			switch (reference::getOptimizer(config).type)
			{
				case AVOCADO_OPTIMIZER_SGD:
					return sgd_helper(reference::getOptimizer(config), alpha, reference::getMemory(dwMem), beta, reference::getTensor(wDesc),
							reference::getMemory(wMem), reference::getMemory(workspace));
				case AVOCADO_OPTIMIZER_ADAM:
					return adam_helper(reference::getOptimizer(config), alpha, reference::getMemory(dwMem), beta, reference::getTensor(wDesc),
							reference::getMemory(wMem), reference::getMemory(workspace));
				default:
					return AVOCADO_STATUS_BAD_PARAM;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

