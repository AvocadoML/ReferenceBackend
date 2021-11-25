/*
 * batchnorm.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/tensor_helpers.hpp>

#include "activations.hpp"
#include "utils.hpp"

#include <memory>
#include <cassert>

namespace
{
	using namespace avocado::backend;

	template<typename T>
	void kernel_affine_forward(T alpha, T beta, const T *input, T *output, const T *weight, const T *bias, size_t first_dim, size_t last_dim,
			avActivationType_t type)
	{
		if (beta == zero<T>())
			clear(output, first_dim * last_dim);
		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < last_dim; j++)
				output[i * last_dim + j] = alpha * activation_forward(type, weight[j] * input[i * last_dim + j] + bias[j])
						+ beta * output[i * last_dim + j];
	}
	template<typename T>
	void kernel_batchnorm_inference(T alpha, T beta, const T *input, T *output, const T *scale, const T *bias, const T *estimated_mean,
			const T *estimated_variance, T epsilon, size_t first_dim, size_t last_dim, avActivationType_t type)
	{
		if (beta == zero<T>())
			clear(output, first_dim * last_dim);
		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < last_dim; j++)
			{
				T tmp = scale[j] * (input[i * last_dim + j] - estimated_mean[j]) / sqrt(epsilon + estimated_variance[j]) + bias[j];
				tmp = activation_forward(type, scale[j] * tmp + bias[j]);
				output[i * last_dim + j] = alpha * tmp + beta * output[i * last_dim + j];
			}
	}
	template<typename T>
	void kernel_batchnorm_forward(T alpha, T beta, const T *input, T *output, const T *scale, const T *bias, T *saved_mean, T *saved_variance,
			T epsilon, size_t first_dim, size_t last_dim, avActivationType_t type)
	{
		clear(saved_mean, last_dim);
		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < last_dim; j++)
				saved_mean[j] += input[i * last_dim + j];
		for (size_t j = 0; j < last_dim; j++)
			saved_mean[j] /= static_cast<T>(first_dim);

		clear(saved_variance, last_dim);
		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < last_dim; j++)
			{
				T tmp = input[i * last_dim + j] - saved_mean[j];
				saved_variance[j] += square(tmp);
			}
		for (size_t j = 0; j < last_dim; j++)
			saved_variance[j] /= static_cast<T>(first_dim);

		kernel_batchnorm_inference(alpha, beta, input, output, scale, bias, saved_mean, saved_variance, epsilon, first_dim, last_dim, type);
	}
	template<typename T>
	void kernel_batchnorm_backward(T alpha, T beta, const T *input, const T *output, T *gradient_prev, T *gradient_next, const T *scale,
			const T *savedMean, const T *savedVariance, T epsilon, size_t first_dim, size_t last_dim, avActivationType_t type)
	{
		std::unique_ptr<T[]> d_sigma = std::make_unique<T[]>(last_dim);
		std::unique_ptr<T[]> d_mu = std::make_unique<T[]>(last_dim);
		clear(d_sigma.get(), last_dim);
		clear(d_mu.get(), last_dim);

		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < last_dim; j++)
			{
				gradient_next[i * last_dim + j] = activation_backward(type, gradient_next[i * last_dim + j], output[i * last_dim + j]);

				T stddev = sqrt(epsilon + savedVariance[j]);
				T in = (input[i * last_dim + j] - savedMean[j]) / stddev;
				T tmp = -scale[j] * gradient_next[i * last_dim + j] / stddev;
				d_sigma[j] += tmp * in;
				d_mu[j] += tmp;
			}

		if (beta == zero<T>())
			clear(gradient_prev, first_dim * last_dim);
		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < last_dim; j++)
			{
				T stddev = sqrt(epsilon + savedVariance[j]);
				T in = (input[i * last_dim + j] - savedMean[j]) / stddev;
				T m = static_cast<T>(first_dim);
				T tmp1 = scale[j] * gradient_next[i * last_dim + j] / stddev;
				T tmp2 = d_sigma[j] * in / m;
				T tmp3 = d_mu[j] / m;
				gradient_prev[i * last_dim + j] = alpha * (tmp1 + tmp2 + tmp3) + beta * gradient_prev[i * last_dim + j];
			}
	}
	template<typename T>
	void kernel_batchnorm_update(T alpha, T beta, const T *input, const T *gradient_next, T *scale_update, T *bias_update, const T *savedMean,
			const T *savedVariance, T epsilon, size_t first_dim, size_t last_dim)
	{
		std::unique_ptr<T[]> d_gamma = std::make_unique<T[]>(last_dim);
		std::unique_ptr<T[]> d_beta = std::make_unique<T[]>(last_dim);
		clear(d_gamma.get(), last_dim);
		clear(d_beta.get(), last_dim);

		for (size_t i = 0; i < first_dim; i++)
			for (size_t j = 0; j < last_dim; j++)
			{
				T in = (input[i * last_dim + j] - savedMean[j]) / sqrt(epsilon + savedVariance[j]);
				d_gamma[j] += gradient_next[i * last_dim + j] * in;
				d_beta[j] += gradient_next[i * last_dim + j];
			}

		if (beta == zero<T>())
		{
			clear(scale_update, last_dim);
			clear(bias_update, last_dim);
		}
		for (size_t j = 0; j < last_dim; j++)
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
		avStatus_t refAffineForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input, avTensor_t output,
				const avTensor_t weight, const avTensor_t bias, const avActivation_t activation)
		{
			assert(same_type(input, output, weight, bias));
			assert(same_shape(input, output));
			assert(same_shape(weight, bias));

			std::pair<size_t, size_t> dimensions = getBroadcastDimensions(output, bias);
			switch (output->dtype)
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_affine_forward<float16>(getDoubleValue(alpha), getDoubleValue(beta), data<float16>(input), data<float16>(output),
							data<float16>(weight), data<float16>(bias), dimensions.first, dimensions.second, activation->type);
					return AVOCADO_STATUS_SUCCESS;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_affine_forward<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output),
							data<float>(weight), data<float>(bias), dimensions.first, dimensions.second, activation->type);
					return AVOCADO_STATUS_SUCCESS;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_affine_forward<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(output),
							data<double>(weight), data<double>(bias), dimensions.first, dimensions.second, activation->type);
					return AVOCADO_STATUS_SUCCESS;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
		avStatus_t refBatchNormInference(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, const avTensor_t scale, const avTensor_t bias, const avTensor_t estimatedMean, const avTensor_t estimatedVariance,
				const avScalar_t epsilon, const avActivation_t activation)
		{
			assert(same_type(input, output, scale, bias, estimatedMean, estimatedVariance));
			assert(same_shape(input, output));
			assert(same_shape(scale, bias, estimatedMean, estimatedVariance));

			std::pair<size_t, size_t> dimensions = getBroadcastDimensions(output, bias);
			switch (output->dtype)
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_batchnorm_inference<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output),
							data<float>(scale), data<float>(bias), data<float>(estimatedMean), data<float>(estimatedVariance), dimensions.first,
							dimensions.second, getDoubleValue(epsilon), activation->type);
					return AVOCADO_STATUS_SUCCESS;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_batchnorm_inference<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(output),
							data<double>(scale), data<double>(bias), data<double>(estimatedMean), data<double>(estimatedVariance), dimensions.first,
							dimensions.second, getDoubleValue(epsilon), activation->type);
					return AVOCADO_STATUS_SUCCESS;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
		avStatus_t refBatchNormForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input, avTensor_t output,
				const avTensor_t scale, const avTensor_t bias, avTensor_t savedMean, avTensor_t savedVariance, const avScalar_t epsilon,
				const avActivation_t activation)
		{
			assert(same_type(input, output, scale, bias, savedMean, savedVariance));
			assert(same_shape(input, output));
			assert(same_shape(scale, bias, savedMean, savedVariance));

			std::pair<size_t, size_t> dimensions = getBroadcastDimensions(output, bias);
			switch (output->dtype)
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_batchnorm_forward<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output),
							data<float>(scale), data<float>(bias), data<float>(savedMean), data<float>(savedVariance), dimensions.first,
							dimensions.second, getDoubleValue(epsilon), activation->type);
					return AVOCADO_STATUS_SUCCESS;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_batchnorm_forward<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(output),
							data<double>(scale), data<double>(bias), data<double>(savedMean), data<double>(savedVariance), dimensions.first,
							dimensions.second, getDoubleValue(epsilon), activation->type);
					return AVOCADO_STATUS_SUCCESS;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
		avStatus_t refBatchNormBackward(avContext_t context, const avActivation_t activation, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, const avTensor_t output, avTensor_t gradientPrev, avTensor_t gradientNext, const avTensor_t scale,
				const avTensor_t savedMean, const avTensor_t savedVariance, const avScalar_t epsilon)
		{
			assert(same_type(input, output, gradientPrev, gradientNext, scale, savedMean, savedVariance));
			assert(same_shape(input, output, gradientPrev, gradientNext));
			assert(same_shape(scale, savedMean, savedVariance));

			std::pair<size_t, size_t> dimensions = getBroadcastDimensions(output, scale);
			switch (output->dtype)
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_batchnorm_backward<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output),
							data<float>(gradientPrev), data<float>(gradientNext), data<float>(scale), data<float>(savedMean),
							data<float>(savedVariance), dimensions.first, dimensions.second, getDoubleValue(epsilon), activation->type);
					return AVOCADO_STATUS_SUCCESS;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_batchnorm_backward<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(output),
							data<double>(gradientPrev), data<double>(gradientNext), data<double>(scale), data<double>(savedMean),
							data<double>(savedVariance), dimensions.first, dimensions.second, getDoubleValue(epsilon), activation->type);
					return AVOCADO_STATUS_SUCCESS;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
		avStatus_t refBatchNormUpdate(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				const avTensor_t gradientNext, avTensor_t scaleUpdate, avTensor_t biasUpdate, const avTensor_t savedMean,
				const avTensor_t savedVariance, const avScalar_t epsilon)
		{
			assert(same_type(input, gradientNext, scaleUpdate, biasUpdate, savedMean, savedVariance));
			assert(same_shape(input, gradientNext));
			assert(same_shape(scaleUpdate, biasUpdate, savedMean, savedVariance));

			std::pair<size_t, size_t> dimensions = getBroadcastDimensions(input, biasUpdate);
			switch (input->dtype)
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_batchnorm_update<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(gradientNext),
							data<float>(scaleUpdate), data<float>(biasUpdate), data<float>(savedMean), data<float>(savedVariance), dimensions.first,
							dimensions.second, getDoubleValue(epsilon));
					return AVOCADO_STATUS_SUCCESS;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_batchnorm_update<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(gradientNext),
							data<double>(scaleUpdate), data<double>(biasUpdate), data<double>(savedMean), data<double>(savedVariance),
							dimensions.first, dimensions.second, getDoubleValue(epsilon));
					return AVOCADO_STATUS_SUCCESS;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
	} /* namespace backend */
} /* namespace avocado */

