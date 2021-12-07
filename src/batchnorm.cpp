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

#include <memory>
#include <cassert>

namespace
{
	using namespace avocado::backend;

	template<typename T>
	void kernel_affine_forward(T alpha, T beta, const T *input, T *output, const T *weight, const T *bias, BroadcastedDimensions dims,
			avActivationType_t type)
	{
		if (beta == zero<T>())
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
				T tmp = scale[j] * (input[i * dims.last + j] - estimated_mean[j]) / sqrt(epsilon + estimated_variance[j]) + bias[j];
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

				T stddev = sqrt(epsilon + savedVariance[j]);
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
				T stddev = sqrt(epsilon + savedVariance[j]);
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
				T in = (input[i * dims.last + j] - savedMean[j]) / sqrt(epsilon + savedVariance[j]);
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
//		avStatus_t refAffineForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input, avTensor_t output,
//				const avTensor_t weight, const avTensor_t bias, const avActivationType_t activation)
//		{
//			assert(same_type(input, output, weight, bias));
//			assert(same_shape(input, output));
//			assert(same_shape(weight, bias));
//
//			BroadcastedDimensions dimensions = getBroadcastDimensions(output, bias);
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT16:
//				{
//					kernel_affine_forward<float16>(getDoubleValue(alpha), getDoubleValue(beta), data<float16>(input), data<float16>(output),
//							data<float16>(weight), data<float16>(bias), dimensions, activation);
//					break;
//				}
//				case AVOCADO_DTYPE_BFLOAT16:
//				{
//					kernel_affine_forward<bfloat16>(getDoubleValue(alpha), getDoubleValue(beta), data<bfloat16>(input), data<bfloat16>(output),
//							data<bfloat16>(weight), data<bfloat16>(bias), dimensions, activation);
//					break;
//				}
//				case AVOCADO_DTYPE_FLOAT32:
//				{
//					kernel_affine_forward<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output),
//							data<float>(weight), data<float>(bias), dimensions, activation);
//					break;
//				}
//				case AVOCADO_DTYPE_FLOAT64:
//				{
//					kernel_affine_forward<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(output),
//							data<double>(weight), data<double>(bias), dimensions, activation);
//					break;
//				}
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refBatchNormInference(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
//				avTensor_t output, const avTensor_t scale, const avTensor_t bias, const avTensor_t estimatedMean, const avTensor_t estimatedVariance,
//				double epsilon, const avActivationType_t activation)
//		{
//			assert(same_type(input, output, scale, bias, estimatedMean, estimatedVariance));
//			assert(same_shape(input, output));
//			assert(same_shape(scale, bias, estimatedMean, estimatedVariance));
//
//			BroadcastedDimensions dimensions = getBroadcastDimensions(output, bias);
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//				{
//					kernel_batchnorm_inference<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output),
//							data<float>(scale), data<float>(bias), data<float>(estimatedMean), data<float>(estimatedVariance), epsilon, dimensions,
//							activation);
//					break;
//				}
//				case AVOCADO_DTYPE_FLOAT64:
//				{
//					kernel_batchnorm_inference<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(output),
//							data<double>(scale), data<double>(bias), data<double>(estimatedMean), data<double>(estimatedVariance), epsilon,
//							dimensions, activation);
//					break;
//				}
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refBatchNormForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input, avTensor_t output,
//				const avTensor_t scale, const avTensor_t bias, avTensor_t savedMean, avTensor_t savedVariance, double epsilon,
//				const avActivationType_t activation)
//		{
//			assert(same_type(input, output, scale, bias, savedMean, savedVariance));
//			assert(same_shape(input, output));
//			assert(same_shape(scale, bias, savedMean, savedVariance));
//
//			BroadcastedDimensions dimensions = getBroadcastDimensions(output, bias);
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//				{
//					kernel_batchnorm_forward<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output),
//							data<float>(scale), data<float>(bias), data<float>(savedMean), data<float>(savedVariance), epsilon, dimensions,
//							activation);
//					break;
//				}
//				case AVOCADO_DTYPE_FLOAT64:
//				{
//					kernel_batchnorm_forward<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(output),
//							data<double>(scale), data<double>(bias), data<double>(savedMean), data<double>(savedVariance), epsilon, dimensions,
//							activation);
//					break;
//				}
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refBatchNormBackward(avContext_t context, const avActivationType_t activation, const avScalar_t alpha, const avScalar_t beta,
//				const avTensor_t input, const avTensor_t output, avTensor_t gradientPrev, avTensor_t gradientNext, const avTensor_t scale,
//				const avTensor_t savedMean, const avTensor_t savedVariance, double epsilon)
//		{
//			assert(same_type(input, output, gradientPrev, gradientNext, scale, savedMean, savedVariance));
//			assert(same_shape(input, output, gradientPrev, gradientNext));
//			assert(same_shape(scale, savedMean, savedVariance));
//
//			BroadcastedDimensions dimensions = getBroadcastDimensions(output, scale);
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//				{
//					kernel_batchnorm_backward<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output),
//							data<float>(gradientPrev), data<float>(gradientNext), data<float>(scale), data<float>(savedMean),
//							data<float>(savedVariance), epsilon, dimensions, activation);
//					break;
//				}
//				case AVOCADO_DTYPE_FLOAT64:
//				{
//					kernel_batchnorm_backward<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(output),
//							data<double>(gradientPrev), data<double>(gradientNext), data<double>(scale), data<double>(savedMean),
//							data<double>(savedVariance), epsilon, dimensions, activation);
//					break;
//				}
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refBatchNormUpdate(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
//				const avTensor_t gradientNext, avTensor_t scaleUpdate, avTensor_t biasUpdate, const avTensor_t savedMean,
//				const avTensor_t savedVariance, double epsilon)
//		{
//			assert(same_type(input, gradientNext, scaleUpdate, biasUpdate, savedMean, savedVariance));
//			assert(same_shape(input, gradientNext));
//			assert(same_shape(scaleUpdate, biasUpdate, savedMean, savedVariance));
//
//			BroadcastedDimensions dimensions = getBroadcastDimensions(input, biasUpdate);
//			switch (input->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//				{
//					kernel_batchnorm_update<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(gradientNext),
//							data<float>(scaleUpdate), data<float>(biasUpdate), data<float>(savedMean), data<float>(savedVariance), epsilon,
//							dimensions);
//					break;
//				}
//				case AVOCADO_DTYPE_FLOAT64:
//				{
//					kernel_batchnorm_update<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(gradientNext),
//							data<double>(scaleUpdate), data<double>(biasUpdate), data<double>(savedMean), data<double>(savedVariance), epsilon,
//							dimensions);
//					break;
//				}
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}

	} /* namespace backend */
} /* namespace avocado */

