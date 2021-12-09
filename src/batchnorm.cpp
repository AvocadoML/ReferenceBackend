/*
 * batchnorm.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

#include "activations.hpp"
#include "utils.hpp"
#include "descriptors.hpp"

#include <memory>
#include <cassert>

namespace
{
	using namespace avocado::backend;

	template<typename T, typename U>
	void kernel_affine_forward(U alpha, U beta, const T *input, T *output, const T *weight, const T *bias, BroadcastedDimensions dims,
			avActivationType_t type)
	{
		if (beta == zero<U>())
			clear(output, volume(dims));
		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
				output[i * dims.last + j] = alpha * activation_forward(type, weight[j] * input[i * dims.last + j] + bias[j])
						+ beta * output[i * dims.last + j];
	}
	template<typename T>
	void kernel_batchnorm_inference(T alpha, T beta, const T *input, T *output, const T *scale, const T *bias, const T *estimated_mean,
			const T *estimated_variance, T epsilon, BroadcastedDimensions dims, avActivationType_t type)
	{
		if (beta == zero<T>())
			clear(output, volume(dims));
		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
			{
				T tmp = scale[j] * (input[i * dims.last + j] - estimated_mean[j]) / std::sqrt(epsilon + estimated_variance[j]) + bias[j];
				tmp = activation_forward(type, scale[j] * tmp + bias[j]);
				output[i * dims.last + j] = alpha * tmp + beta * output[i * dims.last + j];
			}
	}
	template<typename T>
	void kernel_batchnorm_forward(T alpha, T beta, const T *input, T *output, const T *scale, const T *bias, T *saved_mean, T *saved_variance,
			T epsilon, BroadcastedDimensions dims, avActivationType_t type)
	{
		clear(saved_mean, dims.last);
		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
				saved_mean[j] += input[i * dims.last + j];
		for (avSize_t j = 0; j < dims.last; j++)
			saved_mean[j] /= static_cast<T>(dims.first);

		clear(saved_variance, dims.last);
		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
			{
				T tmp = input[i * dims.last + j] - saved_mean[j];
				saved_variance[j] += square(tmp);
			}
		for (avSize_t j = 0; j < dims.last; j++)
			saved_variance[j] /= static_cast<T>(dims.first);

		kernel_batchnorm_inference(alpha, beta, input, output, scale, bias, saved_mean, saved_variance, epsilon, dims, type);
	}
	template<typename T>
	void kernel_batchnorm_backward(T alpha, T beta, const T *input, const T *output, T *gradient_prev, T *gradient_next, const T *scale,
			const T *savedMean, const T *savedVariance, T epsilon, BroadcastedDimensions dims, avActivationType_t type)
	{
		std::unique_ptr<T[]> d_sigma = std::make_unique<T[]>(dims.last);
		std::unique_ptr<T[]> d_mu = std::make_unique<T[]>(dims.last);
		clear(d_sigma.get(), dims.last);
		clear(d_mu.get(), dims.last);

		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
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
		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
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
			const T *savedVariance, T epsilon, BroadcastedDimensions dims)
	{
		std::unique_ptr<T[]> d_gamma = std::make_unique<T[]>(dims.last);
		std::unique_ptr<T[]> d_beta = std::make_unique<T[]>(dims.last);
		clear(d_gamma.get(), dims.last);
		clear(d_beta.get(), dims.last);

		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
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
		for (avSize_t j = 0; j < dims.last; j++)
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
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(yDesc), getTensor(xDesc));
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_affine_forward<float16>(getAlphaValue(alpha), getBetaValue(beta), getPointer<float16>(xMem), getPointer<float16>(yMem),
							getPointer<float16>(wMem), getPointer<float16>(bMem), dimensions, activation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_affine_forward(getAlphaValue(alpha), getBetaValue(beta), getPointer<bfloat16>(xMem), getPointer<bfloat16>(yMem),
							getPointer<bfloat16>(wMem), getPointer<bfloat16>(bMem), dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_affine_forward(getAlphaValue(alpha), getBetaValue(beta), getPointer<float>(xMem), getPointer<float>(yMem),
							getPointer<float>(wMem), getPointer<float>(bMem), dimensions, activation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_affine_forward(getAlphaValue<double>(alpha), getBetaValue<double>(beta), getPointer<double>(xMem),
							getPointer<double>(yMem), getPointer<double>(wMem), getPointer<double>(bMem), dimensions, activation);
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
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(xDesc), getTensor(scaleBiasMeanVarDesc));
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_batchnorm_inference<float>(getAlphaValue(alpha), getBetaValue(beta), getPointer<float>(xMem), getPointer<float>(yMem),
							getPointer<float>(scaleMem), getPointer<float>(biasMem), getPointer<float>(meanMem), getPointer<float>(varianceMem),
							epsilon, dimensions, activation);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_batchnorm_inference<double>(getAlphaValue<double>(alpha), getBetaValue<double>(beta), getPointer<double>(xMem),
							getPointer<double>(yMem), getPointer<double>(scaleMem), getPointer<double>(biasMem), getPointer<double>(meanMem),
							getPointer<double>(varianceMem), epsilon, dimensions, activation);
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
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(xDesc), getTensor(scaleBiasMeanVarDesc));
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_batchnorm_forward<float>(getAlphaValue(alpha), getBetaValue(beta), getPointer<float>(xMem), getPointer<float>(yMem),
							getPointer<float>(scaleMem), getPointer<float>(biasMem), getPointer<float>(meanMem), getPointer<float>(varianceMem),
							epsilon, dimensions, activation);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_batchnorm_forward<double>(getAlphaValue<double>(alpha), getBetaValue<double>(beta), getPointer<double>(xMem),
							getPointer<double>(yMem), getPointer<double>(scaleMem), getPointer<double>(biasMem), getPointer<double>(meanMem),
							getPointer<double>(varianceMem), epsilon, dimensions, activation);
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
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(xDesc), getTensor(scaleMeanVarDesc));
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_batchnorm_backward<float>(getAlphaValue(alpha), getBetaValue(beta), getPointer<float>(xMem), getPointer<float>(yMem),
							getPointer<float>(dxMem), getPointer<float>(dyMem), getPointer<float>(scaleMem), getPointer<float>(meanMem),
							getPointer<float>(varianceMem), epsilon, dimensions, activation);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_batchnorm_backward<double>(getAlphaValue<double>(alpha), getBetaValue<double>(beta), getPointer<double>(xMem),
							getPointer<double>(yMem), getPointer<double>(dxMem), getPointer<double>(dyMem), getPointer<double>(scaleMem),
							getPointer<double>(meanMem), getPointer<double>(varianceMem), epsilon, dimensions, activation);
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refBatchNormUpdate(avContextDescriptor_t context, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t scaleBiasDesc, avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(xDesc), getTensor(scaleUpdateMem));
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_batchnorm_update<float>(getAlphaValue(alpha), getBetaValue(beta), getPointer<float>(xMem), getPointer<float>(dyMem),
							getPointer<float>(scaleUpdateMem), getPointer<float>(biasUpdateMem), getPointer<float>(meanMem),
							getPointer<float>(varianceMem), epsilon, dimensions);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_batchnorm_update<double>(getAlphaValue(alpha), getBetaValue(beta), getPointer<double>(xMem), getPointer<double>(dyMem),
							getPointer<double>(scaleUpdateMem), getPointer<double>(biasUpdateMem), getPointer<double>(meanMem),
							getPointer<double>(varianceMem), epsilon, dimensions);
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

