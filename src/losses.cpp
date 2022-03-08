/*
 * losses.cpp
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
	T kernel_MSE_loss(const T *output, const T *target, av_int64 elements) noexcept
	{
		T result = zero<T>();
		for (av_int64 i = 0; i < elements; i++)
			result += square(output[i] - target[i]);
		return static_cast<T>(0.5) * result;
	}
	template<typename T>
	void kernel_MSE_gradient(T *gradient, const T *output, const T *target, av_int64 elements, T alpha, T beta) noexcept
	{
		if (beta == zero<T>())
			clear(gradient, elements);
		for (av_int64 i = 0; i < elements; i++)
			gradient[i] = alpha * (output[i] - target[i]) + beta * gradient[i];
	}

	template<typename T>
	T kernel_CE_loss(const T *output, const T *target, av_int64 elements) noexcept
	{
		T result = zero<T>();
		for (av_int64 i = 0; i < elements; i++)
			result += target[i] * safe_log(output[i]) + (one<T>() - target[i]) * safe_log(one<T>() - output[i]);
		return -result;
	}
	template<typename T>
	void kernel_CE_gradient(T *gradient, const T *output, const T *target, av_int64 elements, T alpha, T beta, bool fused) noexcept
	{
		if (beta == zero<T>())
			clear(gradient, elements);
		if (fused)
		{
			for (av_int64 i = 0; i < elements; i++)
				gradient[i] = alpha * (output[i] - target[i]) + beta * gradient[i];
		}
		else
		{
			for (av_int64 i = 0; i < elements; i++)
				gradient[i] = alpha * (output[i] - target[i]) / (eps<T>() + output[i] * (one<T>() - output[i])) + beta * gradient[i];
		}
	}

	template<typename T>
	T kernel_KL_loss(const T *output, const T *target, av_int64 elements) noexcept
	{
		T result = zero<T>();
		for (av_int64 i = 0; i < elements; i++)
			result += target[i] * safe_log(output[i]) + (one<T>() - target[i]) * safe_log(one<T>() - output[i]) - target[i] * safe_log(target[i])
					- (one<T>() - target[i]) * safe_log(one<T>() - target[i]);
		return -result;
	}
	template<typename T>
	void kernel_KL_gradient(T *gradient, const T *output, const T *target, av_int64 elements, T alpha, T beta, bool fused) noexcept
	{
		if (beta == zero<T>())
			clear(gradient, elements);
		if (fused)
		{
			for (av_int64 i = 0; i < elements; i++)
				gradient[i] = alpha * (output[i] - target[i]) + beta * gradient[i];
		}
		else
		{
			for (av_int64 i = 0; i < elements; i++)
				gradient[i] = alpha * (output[i] - target[i]) / (eps<T>() + output[i] * (one<T>() - output[i])) + beta * gradient[i];
		}
	}

	template<typename T>
	T loss_helper(avLossType_t lossType, const T *output, const T *target, av_int64 elements) noexcept
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
	void gradient_helper(avLossType_t lossType, T *gradient, const T *output, const T *target, av_int64 elements, T alpha, T beta, bool fused)
	noexcept
	{
		switch (lossType)
		{
			case AVOCADO_MEAN_SQUARE_LOSS:
				kernel_MSE_gradient(gradient, output, target, elements, alpha, beta);
				break;
			case AVOCADO_CROSS_ENTROPY_LOSS:
				kernel_CE_gradient(gradient, output, target, elements, alpha, beta, fused);
				break;
			case AVOCADO_KL_DIVERGENCE_LOSS:
				kernel_KL_gradient(gradient, output, target, elements, alpha, beta, fused);
				break;
		}
	}
}

namespace avocado
{
	namespace backend
	{

		avStatus_t refLossFunction(avContextDescriptor_t context, avLossType_t lossType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
			const av_int64 elements = reference::getTensor(outputDesc).volume();
			switch (reference::getTensor(outputDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					float loss = loss_helper(lossType, reference::getPointer<float>(outputMem), reference::getPointer<float>(targetMem), elements);
					std::memcpy(result, &loss, sizeof(float));
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					double loss = loss_helper(lossType, reference::getPointer<double>(outputMem), reference::getPointer<double>(targetMem), elements);
					std::memcpy(result, &loss, sizeof(double));
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refLossGradient(avContextDescriptor_t context, avLossType_t lossType, const void *alpha, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, const void *beta,
				const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem, bool isFused)
		{
			const av_int64 elements = reference::getTensor(outputDesc).volume();
			switch (reference::getTensor(outputDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					gradient_helper(lossType, reference::getPointer<float>(gradientMem), reference::getPointer<float>(outputMem),
							reference::getPointer<float>(targetMem), elements, reference::getAlphaValue(alpha), reference::getBetaValue(beta),
							isFused);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					gradient_helper(lossType, reference::getPointer<double>(gradientMem), reference::getPointer<double>(outputMem),
							reference::getPointer<double>(targetMem), elements, reference::getAlphaValue<double>(alpha),
							reference::getBetaValue<double>(beta), isFused);
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

