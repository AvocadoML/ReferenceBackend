/*
 * activations.cpp
 *
 *  Created on: Nov 11, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/backend_descriptors.hpp>
#include "activations.hpp"

#include <Avocado/reference_backend.h>
#include <memory>

namespace
{
	using namespace avocado::backend;

	template<typename T, typename U = T>
	void kernel_activation_forward(avActivationType_t type, U alpha, const T *input, U beta, T *output, av_int64 elements) noexcept
	{
		if (beta == zero<U>())
			clear(output, elements);
		for (av_int64 i = 0; i < elements; i++)
			output[i] = alpha * activation_forward(type, input[i]) + beta * output[i];
	}
	template<typename T, typename U = T>
	void kernel_softmax_forward(U alpha, const T *input, U beta, T *output, av_int64 first_dim, av_int64 last_dim)
	{
		std::unique_ptr<U[]> workspace = std::make_unique<U[]>(last_dim);
		clear(workspace.get(), last_dim);

		if (beta == zero<U>())
			clear(output, first_dim * last_dim);

		for (av_int64 i = 0; i < first_dim; i++)
		{
			U max_value = input[i * last_dim];
			for (av_int64 j = 0; j < last_dim; j++)
				max_value = std::max(max_value, static_cast<U>(input[i * last_dim + j]));

			U tmp = zero<U>();
			for (av_int64 j = 0; j < last_dim; j++)
			{
				workspace[j] = avocado::backend::exp(input[i * last_dim + j] - max_value);
				tmp += workspace[j];
			}

			if (tmp == zero<U>())
				fill(workspace.get(), last_dim, one<U>() / static_cast<U>(last_dim));
			else
			{
				tmp = one<U>() / tmp;
				for (av_int64 j = 0; j < last_dim; j++)
					workspace[j] *= tmp;
			}

			for (av_int64 j = 0; j < last_dim; j++)
				output[i * last_dim + j] = alpha * workspace[j] + beta * output[i * last_dim + j];
		}
	}

	template<typename T, typename U = T>
	void kernel_activation_backward(avActivationType_t type, U alpha, T *gradient_prev, U beta, const T *gradient_next, const T *output,
			av_int64 elements) noexcept
	{
		if (beta == zero<U>())
			clear(gradient_prev, elements);
		for (av_int64 i = 0; i < elements; i++)
			gradient_prev[i] = alpha * activation_backward(type, gradient_next[i], output[i]) + beta * gradient_prev[i];
	}
}

namespace avocado
{
	namespace backend
	{
		using namespace BACKEND_NAMESPACE;

		avStatus_t refActivationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			const av_int64 elements = getTensor(xDesc).volume();
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_activation_forward(activation, getAlphaValue(alpha), getPointer<float16>(xMem), getBetaValue(beta),
							getPointer<float16>(yMem), elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_activation_forward(activation, getAlphaValue(alpha), getPointer<bfloat16>(xMem), getBetaValue(beta),
							getPointer<bfloat16>(yMem), elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_activation_forward(activation, getAlphaValue(alpha), getPointer<float>(xMem), getBetaValue(beta), getPointer<float>(yMem),
							elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_activation_forward(activation, getAlphaValue<double>(alpha), getPointer<double>(xMem), getBetaValue<double>(beta),
							getPointer<double>(yMem), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refActivationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			const av_int64 elements = getTensor(yDesc).volume();
			switch (getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_activation_backward(activation, getAlphaValue(alpha), getPointer<float>(dxMem), getBetaValue(beta),
							getPointer<float>(dyMem), getPointer<float>(yMem), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_activation_backward(activation, getAlphaValue<double>(alpha), getPointer<double>(dxMem), getBetaValue<double>(beta),
							getPointer<double>(dyMem), getPointer<double>(yMem), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
			av_int64 first_dim, last_dim;
			if (mode == AVOCADO_SOFTMAX_MODE_CHANNEL)
			{
				first_dim = getTensor(xDesc).volumeWithoutLastDim();
				last_dim = getTensor(xDesc).lastDim();
			}
			else
			{
				first_dim = getTensor(xDesc).firstDim();
				last_dim = getTensor(xDesc).volumeWithoutFirstDim();
			}

			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_softmax_forward(getAlphaValue(alpha), getPointer<float16>(xMem), getBetaValue(beta), getPointer<float16>(yMem), first_dim,
							last_dim);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_softmax_forward(getAlphaValue(alpha), getPointer<bfloat16>(xMem), getBetaValue(beta), getPointer<bfloat16>(yMem),
							first_dim, last_dim);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_softmax_forward(getAlphaValue(alpha), getPointer<float>(xMem), getBetaValue(beta), getPointer<float>(yMem), first_dim,
							last_dim);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_softmax_forward(getAlphaValue<double>(alpha), getPointer<double>(xMem), getBetaValue<double>(beta),
							getPointer<double>(yMem), first_dim, last_dim);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refSoftmaxBackward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t yDesc,
				const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			const av_int64 elements = getTensor(yDesc).volume();
			switch (getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_activation_backward(AVOCADO_ACTIVATION_SOFTMAX, getAlphaValue(alpha), getPointer<float>(dxMem), getBetaValue(beta),
							getPointer<float>(dyMem), getPointer<float>(yMem), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_activation_backward(AVOCADO_ACTIVATION_SOFTMAX, getAlphaValue<double>(alpha), getPointer<double>(dxMem),
							getBetaValue<double>(beta), getPointer<double>(dyMem), getPointer<double>(yMem), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

