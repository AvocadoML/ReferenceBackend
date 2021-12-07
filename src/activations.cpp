/*
 * activations.cpp
 *
 *  Created on: Nov 11, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/backend/backend_descriptors.hpp>

#include "activations.hpp"

#include <avocado/reference_backend.h>

#include "descriptors.hpp"

#include <memory>

namespace
{
	using namespace avocado::backend;

	template<typename T, typename U = T>
	void kernel_activation_forward(avActivationType_t type, U alpha, const T *input, U beta, T *output, avSize_t elements) noexcept
	{
		if (beta == zero<U>())
			clear(output, elements);
		for (avSize_t i = 0; i < elements; i++)
			output[i] = alpha * activation_forward(type, input[i]) + beta * output[i];
	}
	template<typename T>
	void kernel_softmax_forward(T alpha, T beta, const T *input, T *output, avSize_t first_dim, avSize_t last_dim)
	{
		std::unique_ptr<T[]> workspace = std::make_unique<T[]>(last_dim);
		clear(workspace.get(), last_dim);

		if (beta == zero<T>())
			clear(output, first_dim * last_dim);

		for (avSize_t i = 0; i < first_dim; i++)
		{
			T max = input[i * last_dim];
			for (avSize_t j = 0; j < last_dim; j++)
				max = std::max(max, input[i * last_dim + j]);

			T tmp = zero<T>();
			for (avSize_t j = 0; j < last_dim; j++)
			{
				workspace[i * last_dim + j] = avocado::backend::exp(input[i * last_dim + j] - max);
				tmp += workspace[i * last_dim + j];
			}

			if (tmp == zero<T>())
				fill(workspace.get(), last_dim, one<T>() / static_cast<T>(last_dim));
			else
			{
				tmp = one<T>() / tmp;
				for (avSize_t j = 0; j < last_dim; j++)
					workspace[j] *= tmp;
			}

			for (avSize_t j = 0; j < last_dim; j++)
				output[i * last_dim + j] = alpha * workspace[j] + beta * output[i * last_dim + j];
		}
	}

	template<typename T>
	void kernel_activation_backward(avActivationType_t type, T alpha, T beta, T *gradient_prev, const T *gradient_next, const T *output,
			avSize_t elements) noexcept
	{
		if (beta == zero<T>())
			clear(gradient_prev, elements);
		for (avSize_t i = 0; i < elements; i++)
			gradient_prev[i] = alpha * activation_backward(type, gradient_next[i], output[i]) + beta * gradient_prev[i];
	}

}

namespace avocado
{
	namespace backend
	{
		avStatus_t refActivationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
//			assert(same_type(input, output));
//			assert(same_shape(input, output));

			const avSize_t elements = getTensor(xDesc).volume();

			switch (getTensor(xDesc).dtype())
			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_activation_forward<float16, float>(activation, getAlphaValue(alpha), getPointer<float16>(xMem), getBetaValue(beta),
//							getPointer<float16>(yMem), elements);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_activation_forward<bfloat16, float>(activation, getAlphaValue(alpha), getPointer<bfloat16>(xMem), getBetaValue(beta),
//							getPointer<bfloat16>(yMem), elements);
//					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_activation_forward<float>(activation, getAlphaValue(alpha), getPointer<float>(xMem), getBetaValue(beta),
							getPointer<float>(yMem), elements);
					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_activation_forward<double>( activation, getAlphaValue<double>(alpha), getPointer<double>(xMem), getBetaValue<double>(beta),
//							getPointer<double>(yMem), elements);
//					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refActivationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
		}

		avStatus_t refSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
		}
		avStatus_t refSoftmaxBackward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t yDesc,
				const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
		}

//		avStatus_t refActivationForward(avContext_t context, const avActivationType_t activation, const avScalar_t alpha, const avScalar_t beta,
//				const avTensor_t input, avTensor_t output)
//		{
//			assert(same_type(input, output));
//			assert(same_shape(input, output));
//
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_activation_forward<float16>(activation, getDoubleValue(alpha), getDoubleValue(beta), data < float16 > (input),
//							data < float16 > (output), volume(input));
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_activation_forward<bfloat16>(activation, getDoubleValue(alpha), getDoubleValue(beta), data < bfloat16 > (input),
//							data < bfloat16 > (output), volume(input));
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_activation_forward<float>(activation, getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output),
//							volume(input));
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_activation_forward<double>(activation, getDoubleValue(alpha), getDoubleValue(beta), data<double>(input),
//							data<double>(output), volume(input));
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refActivationBackward(avContext_t context, const avActivationType_t activation, const avScalar_t alpha, const avScalar_t beta,
//				avTensor_t gradientPrev, const avTensor_t gradientNext, const avTensor_t output)
//		{
//			assert(same_type(gradientPrev, gradientNext, output));
//			assert(same_shape(gradientPrev, gradientNext, output));
//
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_activation_backward<float>(activation, getDoubleValue(alpha), getDoubleValue(beta), data<float>(gradientPrev),
//							data<float>(gradientNext), data<float>(output), volume(output));
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_activation_backward<double>(activation, getDoubleValue(alpha), getDoubleValue(beta), data<double>(gradientPrev),
//							data<double>(gradientNext), data<double>(output), volume(output));
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refSoftmaxForward(avContext_t context, avSoftmaxMode_t mode, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
//				avTensor_t output)
//		{
//			assert(same_type(input, output));
//			assert(same_shape(input, output));
//
//			avSize_t first_dim, last_dim;
//			if (mode == AVOCADO_SOFTMAX_MODE_CHANNEL)
//			{
//				first_dim = volumeWithoutLastDim(input);
//				last_dim = lastDim(input);
//			}
//			else
//			{
//				first_dim = firstDim(input);
//				last_dim = volumeWithoutFirstDim(input);
//			}
//
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_softmax_forward<float16>(getDoubleValue(alpha), getDoubleValue(beta), data < float16 > (input), data < float16 > (output),
//							first_dim, last_dim);
//					break;
//				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_softmax_forward<bfloat16>(getDoubleValue(alpha), getDoubleValue(beta), data < bfloat16 > (input),
//							data < bfloat16 > (output), first_dim, last_dim);
//					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_softmax_forward<float>(getDoubleValue(alpha), getDoubleValue(beta), data<float>(input), data<float>(output), first_dim,
//							last_dim);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_softmax_forward<double>(getDoubleValue(alpha), getDoubleValue(beta), data<double>(input), data<double>(output), first_dim,
//							last_dim);
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
//		avStatus_t refSoftmaxBackward(avContext_t context, avSoftmaxMode_t mode, const avScalar_t alpha, const avScalar_t beta,
//				avTensor_t gradientPrev, const avTensor_t gradientNext, const avTensor_t output)
//		{
//			assert(same_type(gradientPrev, gradientNext, output));
//			assert(same_shape(gradientPrev, gradientNext, output));
//
//			switch (output->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_activation_backward<float>(AVOCADO_ACTIVATION_SOFTMAX, getDoubleValue(alpha), getDoubleValue(beta),
//							data<float>(gradientPrev), data<float>(gradientNext), data<float>(output), volume(output));
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
//					kernel_activation_backward<double>(AVOCADO_ACTIVATION_SOFTMAX, getDoubleValue(alpha), getDoubleValue(beta),
//							data<double>(gradientPrev), data<double>(gradientNext), data<double>(output), volume(output));
//					break;
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}

	} /* namespace backend */
} /* namespace avocado */

