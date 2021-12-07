/*
 * losses.cpp
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
	T kernel_MSE_loss(const T *output, const T *target, avSize_t elements) noexcept
	{
		T result = zero<T>();
		for (avSize_t i = 0; i < elements; i++)
			result += square(output[i] - target[i]);
		return static_cast<T>(0.5) * result;
	}
	template<typename T>
	void kernel_MSE_gradient(T *gradient, const T *output, const T *target, avSize_t elements, T inv_batch_size) noexcept
	{
		for (avSize_t i = 0; i < elements; i++)
			gradient[i] = inv_batch_size * (output[i] - target[i]);
	}

	template<typename T>
	T kernel_CE_loss(const T *output, const T *target, avSize_t elements) noexcept
	{
		T result = zero<T>();
		for (avSize_t i = 0; i < elements; i++)
			result += target[i] * safe_log(output[i]) + (one<T>() - target[i]) * safe_log(one<T>() - output[i]);
		return -result;
	}
	template<typename T>
	void kernel_CE_gradient(T *gradient, const T *output, const T *target, avSize_t elements, T inv_batch_size, bool fused) noexcept
	{
		if (fused)
		{
			for (avSize_t i = 0; i < elements; i++)
				gradient[i] = inv_batch_size * (output[i] - target[i]);
		}
		else
		{
			for (avSize_t i = 0; i < elements; i++)
				gradient[i] = inv_batch_size * (output[i] - target[i]) / (eps<T>() + output[i] * (one<T>() - output[i]));
		}
	}

	template<typename T>
	T kernel_KL_loss(const T *output, const T *target, avSize_t elements) noexcept
	{
		T result = zero<T>();
		for (avSize_t i = 0; i < elements; i++)
			result += target[i] * safe_log(output[i]) + (one<T>() - target[i]) * safe_log(one<T>() - output[i]) - target[i] * safe_log(target[i])
					- (one<T>() - target[i]) * safe_log(one<T>() - target[i]);
		return -result;
	}
	template<typename T>
	void kernel_KL_gradient(T *gradient, const T *output, const T *target, avSize_t elements, T inv_batch_size, bool fused) noexcept
	{
		if (fused)
		{
			for (avSize_t i = 0; i < elements; i++)
				gradient[i] = inv_batch_size * (output[i] - target[i]);
		}
		else
		{
			for (avSize_t i = 0; i < elements; i++)
				gradient[i] = inv_batch_size * (output[i] - target[i]) / (eps<T>() + output[i] * (one<T>() - output[i]));
		}
	}

	template<typename T>
	T loss_helper(avLossType_t lossType, const T *output, const T *target, avSize_t elements) noexcept
	{
		switch (lossType)
		{
			case AVOCADO_MEAN_SQUARE_LOSS:
				return kernel_MSE_loss(output, target, elements);
			case AVOCADO_CROSS_ENTROPY_LOSS:
				return kernel_CE_loss(output, target, elements);
			case AVOCADO_KL_DIVERGENCE_LOSS:
				return kernel_KL_loss(output, target, elements);
			default:
				return zero<T>();
		}
	}
	template<typename T>
	void gradient_helper(avLossType_t lossType, T *gradient, const T *output, const T *target, avSize_t elements, T inv_batch_size, bool fused)
	noexcept
	{
		switch (lossType)
		{
			case AVOCADO_MEAN_SQUARE_LOSS:
				kernel_MSE_gradient(gradient, output, target, elements, inv_batch_size);
				break;
			case AVOCADO_CROSS_ENTROPY_LOSS:
				kernel_CE_gradient(gradient, output, target, elements, inv_batch_size, fused);
				break;
			case AVOCADO_KL_DIVERGENCE_LOSS:
				kernel_KL_gradient(gradient, output, target, elements, inv_batch_size, fused);
				break;
		}
	}
}

namespace avocado
{
	namespace backend
	{
//		avStatus_t refLossFunction(avContext_t context, avLossType_t lossType, avScalar_t result, const avTensor_t output, const avTensor_t target)
//		{
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//				{
//					float loss = loss_helper(lossType, data<float>(output), data<float>(target), volume(output));
//					setScalarValue<float>(result, loss / firstDim(output));
//					break;
//				}
//				case AVOCADO_DTYPE_FLOAT64:
//				{
//					double loss = loss_helper(lossType, data<double>(output), data<double>(target), volume(output));
//					setScalarValue<double>(result, loss / firstDim(output));
//					break;
//				}
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refLossGradient(avContext_t context, avLossType_t lossType, const avScalar_t alpha, const avScalar_t beta, avTensor_t gradient,
//				const avTensor_t output, const avTensor_t target, bool isFused)
//		{
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//				{
//					gradient_helper(lossType, data<float>(gradient), data<float>(output), data<float>(target), volume(output),
//							one<float>() / firstDim(output), isFused);
//					break;
//				}
//				case AVOCADO_DTYPE_FLOAT64:
//				{
//					gradient_helper(lossType, data<double>(gradient), data<double>(output), data<double>(target), volume(output),
//							one<double>() / firstDim(output), isFused);
//					break;
//				}
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
	} /* namespace backend */
} /* namespace avocado */

