/*
 * batchnorm.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

#include "activations.hpp"
#include "utils.hpp"

#include <memory>
#include <cassert>

namespace
{
	using namespace avocado::backend;

	template<typename T, typename U>
	void kernel_affine_forward(U alpha, U beta, const T *input, T *output, const T *weight, const T *bias, reference::BroadcastedDimensions dims,
			avActivationType_t type)
	{
		if (beta == zero<U>())
			clear(output, volume(dims));
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
				output[i * dims.last + j] = alpha * activation_forward(type, weight[j] * input[i * dims.last + j] + bias[j])
						+ beta * output[i * dims.last + j];
	}
	template<typename T>
	void kernel_batchnorm_inference(T alpha, T beta, const T *input, T *output, const T *scale, const T *bias, const T *estimated_mean,
			const T *estimated_variance, T epsilon, reference::BroadcastedDimensions dims, avActivationType_t type)
	{
		if (beta == zero<T>())
			clear(output, volume(dims));
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
			{
				T tmp = (input[i * dims.last + j] - estimated_mean[j]) / std::sqrt(epsilon + estimated_variance[j]);
				tmp = activation_forward(type, scale[j] * tmp + bias[j]);
				output[i * dims.last + j] = alpha * tmp + beta * output[i * dims.last + j];
			}
	}
	template<typename T>
	void kernel_batchnorm_forward(T alpha, T beta, const T *input, T *output, const T *scale, const T *bias, T *saved_mean, T *saved_variance,
			T epsilon, reference::BroadcastedDimensions dims, avActivationType_t type)
	{
		clear(saved_mean, dims.last);
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
				saved_mean[j] += input[i * dims.last + j];
		for (av_int64 j = 0; j < dims.last; j++)
			saved_mean[j] /= static_cast<T>(dims.first);

		clear(saved_variance, dims.last);
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
			{
				T tmp = input[i * dims.last + j] - saved_mean[j];
				saved_variance[j] += square(tmp);
			}
		for (av_int64 j = 0; j < dims.last; j++)
			saved_variance[j] /= static_cast<T>(dims.first);

		kernel_batchnorm_inference(alpha, beta, input, output, scale, bias, saved_mean, saved_variance, epsilon, dims, type);
	}
	template<typename T>
	void kernel_batchnorm_backward(T alpha, T beta, const T *input, const T *output, T *gradient_prev, T *gradient_next, const T *scale,
			const T *savedMean, const T *savedVariance, T epsilon, reference::BroadcastedDimensions dims, avActivationType_t type)
	{
		std::unique_ptr<T[]> d_sigma = std::make_unique<T[]>(dims.last);
		std::unique_ptr<T[]> d_mu = std::make_unique<T[]>(dims.last);
		clear(d_sigma.get(), dims.last);
		clear(d_mu.get(), dims.last);

		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
			{
				gradient_next[i * dims.last + j] = activation_backward(type, gradient_next[i * dims.last + j], output[i * dims.last + j]);

				T stddev = std::sqrt(epsilon + savedVariance[j]);
				T in = (input[i * dims.last + j] - savedMean[j]) / stddev;
				T tmp = -scale[j] * gradient_next[i * dims.last + j] / stddev;
				d_sigma[j] += tmp * in;
				d_mu[j] += tmp;
			}

		if (beta == zero<T>())
			clear(gradient_prev, volume(dims));
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
			{
				T stddev = std::sqrt(epsilon + savedVariance[j]);
				T in = (input[i * dims.last + j] - savedMean[j]) / stddev;
				T m = static_cast<T>(dims.first);
				T tmp1 = scale[j] * gradient_next[i * dims.last + j] / stddev;
				T tmp2 = d_sigma[j] * in / m;
				T tmp3 = d_mu[j] / m;
				gradient_prev[i * dims.last + j] = alpha * (tmp1 + tmp2 + tmp3) + beta * gradient_prev[i * dims.last + j];
			}
	}
	template<typename T>
	void kernel_batchnorm_update(T alpha, T beta, const T *input, const T *gradient_next, T *scale_update, T *bias_update, const T *savedMean,
			const T *savedVariance, T epsilon, reference::BroadcastedDimensions dims)
	{
		std::unique_ptr<T[]> d_gamma = std::make_unique<T[]>(dims.last);
		std::unique_ptr<T[]> d_beta = std::make_unique<T[]>(dims.last);
		clear(d_gamma.get(), dims.last);
		clear(d_beta.get(), dims.last);

		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
			{
				T in = (input[i * dims.last + j] - savedMean[j]) / std::sqrt(epsilon + savedVariance[j]);
				d_gamma[j] += gradient_next[i * dims.last + j] * in;
				d_beta[j] += gradient_next[i * dims.last + j];
			}

		if (beta == zero<T>())
		{
			clear(scale_update, dims.last);
			clear(bias_update, dims.last);
		}
		for (av_int64 j = 0; j < dims.last; j++)
		{
			scale_update[j] = alpha * d_gamma[j] + beta * scale_update[j];
			bias_update[j] = alpha * d_beta[j] + beta * bias_update[j];
		}
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refAffineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			reference::BroadcastedDimensions dimensions = getBroadcastDimensions(reference::getTensor(yDesc), reference::getTensor(xDesc));
			switch (reference::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_affine_forward<float16>(reference::getAlphaValue(alpha), reference::getBetaValue(beta), reference::getPointer<float16>(xMem), reference::getPointer<float16>(yMem),
							reference::getPointer<float16>(wMem), reference::getPointer<float16>(bMem), dimensions, activation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_affine_forward(reference::getAlphaValue(alpha), reference::getBetaValue(beta), reference::getPointer<bfloat16>(xMem), reference::getPointer<bfloat16>(yMem),
							reference::getPointer<bfloat16>(wMem), reference::getPointer<bfloat16>(bMem), dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_affine_forward(reference::getAlphaValue(alpha), reference::getBetaValue(beta), reference::getPointer<float>(xMem), reference::getPointer<float>(yMem),
							reference::getPointer<float>(wMem), reference::getPointer<float>(bMem), dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_affine_forward(reference::getAlphaValue<double>(alpha), reference::getBetaValue<double>(beta), reference::getPointer<double>(xMem),
							reference::getPointer<double>(yMem), reference::getPointer<double>(wMem), reference::getPointer<double>(bMem), dimensions, activation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refBatchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon)
		{
			reference::BroadcastedDimensions dimensions = getBroadcastDimensions(reference::getTensor(xDesc), reference::getTensor(scaleBiasMeanVarDesc));
			switch (reference::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_batchnorm_inference<float>(reference::getAlphaValue(alpha), reference::getBetaValue(beta), reference::getPointer<float>(xMem), reference::getPointer<float>(yMem),
							reference::getPointer<float>(scaleMem), reference::getPointer<float>(biasMem), reference::getPointer<float>(meanMem), reference::getPointer<float>(varianceMem),
							epsilon, dimensions, activation);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_batchnorm_inference<double>(reference::getAlphaValue<double>(alpha), reference::getBetaValue<double>(beta), reference::getPointer<double>(xMem),
							reference::getPointer<double>(yMem), reference::getPointer<double>(scaleMem), reference::getPointer<double>(biasMem), reference::getPointer<double>(meanMem),
							reference::getPointer<double>(varianceMem), epsilon, dimensions, activation);
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refBatchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon)
		{
			reference::BroadcastedDimensions dimensions = getBroadcastDimensions(reference::getTensor(xDesc), reference::getTensor(scaleBiasMeanVarDesc));
			switch (reference::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_batchnorm_forward<float>(reference::getAlphaValue(alpha), reference::getBetaValue(beta), reference::getPointer<float>(xMem), reference::getPointer<float>(yMem),
							reference::getPointer<float>(scaleMem), reference::getPointer<float>(biasMem), reference::getPointer<float>(meanMem), reference::getPointer<float>(varianceMem),
							epsilon, dimensions, activation);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_batchnorm_forward<double>(reference::getAlphaValue<double>(alpha), reference::getBetaValue<double>(beta), reference::getPointer<double>(xMem),
							reference::getPointer<double>(yMem), reference::getPointer<double>(scaleMem), reference::getPointer<double>(biasMem), reference::getPointer<double>(meanMem),
							reference::getPointer<double>(varianceMem), epsilon, dimensions, activation);
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refBatchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem,
				const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc,
				avMemoryDescriptor_t dyMem, const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, const void *alpha2, const void *beta2,
				avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem, double epsilon)
		{
			reference::BroadcastedDimensions dimensions = getBroadcastDimensions(reference::getTensor(xDesc), reference::getTensor(scaleMeanVarDesc));
			switch (reference::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_batchnorm_backward<float>(reference::getAlphaValue(alpha), reference::getBetaValue(beta), reference::getPointer<float>(xMem), reference::getPointer<float>(yMem),
							reference::getPointer<float>(dxMem), reference::getPointer<float>(dyMem), reference::getPointer<float>(scaleMem), reference::getPointer<float>(meanMem),
							reference::getPointer<float>(varianceMem), epsilon, dimensions, activation);
					kernel_batchnorm_update<float>(reference::getAlphaValue(alpha2), reference::getBetaValue(beta2), reference::getPointer<float>(xMem), reference::getPointer<float>(dyMem),
							reference::getPointer<float>(scaleUpdateMem), reference::getPointer<float>(biasUpdateMem), reference::getPointer<float>(meanMem),
							reference::getPointer<float>(varianceMem), epsilon, dimensions);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_batchnorm_backward<double>(reference::getAlphaValue<double>(alpha), reference::getBetaValue<double>(beta), reference::getPointer<double>(xMem),
							reference::getPointer<double>(yMem), reference::getPointer<double>(dxMem), reference::getPointer<double>(dyMem), reference::getPointer<double>(scaleMem),
							reference::getPointer<double>(meanMem), reference::getPointer<double>(varianceMem), epsilon, dimensions, activation);
					kernel_batchnorm_update<double>(reference::getAlphaValue<double>(alpha2), reference::getBetaValue<double>(beta2), reference::getPointer<double>(xMem), reference::getPointer<double>(dyMem),
							reference::getPointer<double>(scaleUpdateMem), reference::getPointer<double>(biasUpdateMem), reference::getPointer<double>(meanMem),
							reference::getPointer<double>(varianceMem), epsilon, dimensions);
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

