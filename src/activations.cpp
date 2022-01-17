/*
 * activations.cpp
 *
 *  Created on: Nov 11, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include "activations.hpp"

#include <ReferenceBackend/reference_backend.h>
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
	template<typename T, typename U = T>
	void kernel_softmax_forward(U alpha, const T *input, U beta, T *output, avSize_t first_dim, avSize_t last_dim)
	{
		std::unique_ptr<T[]> workspace = std::make_unique<T[]>(last_dim);
		clear(workspace.get(), last_dim);

		if (beta == zero<U>())
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

	template<typename T, typename U = T>
	void kernel_activation_backward(avActivationType_t type, U alpha, T *gradient_prev, U beta, const T *gradient_next, const T *output,
			avSize_t elements) noexcept
	{
		if (beta == zero<U>())
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

			const avSize_t elements = reference::getTensor(xDesc).volume();
			switch (reference::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_activation_forward<float16, float>(activation, reference::getAlphaValue(alpha), reference::getPointer<float16>(xMem),
							reference::getBetaValue(beta), reference::getPointer<float16>(yMem), elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_activation_forward<bfloat16, float>(activation, reference::getAlphaValue(alpha), reference::getPointer<bfloat16>(xMem),
							reference::getBetaValue(beta), reference::getPointer<bfloat16>(yMem), elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_activation_forward<float>(activation, reference::getAlphaValue(alpha), reference::getPointer<float>(xMem),
							reference::getBetaValue(beta), reference::getPointer<float>(yMem), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_activation_forward<double>(activation, reference::getAlphaValue<double>(alpha), reference::getPointer<double>(xMem),
							reference::getBetaValue<double>(beta), reference::getPointer<double>(yMem), elements);
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
			const avSize_t elements = reference::getTensor(yDesc).volume();
			switch (reference::getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_activation_backward<float>(activation, reference::getAlphaValue(alpha), reference::getPointer<float>(dxMem),
							reference::getBetaValue(beta), reference::getPointer<float>(dyMem), reference::getPointer<float>(yMem), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_activation_backward<double>(activation, reference::getAlphaValue(alpha), reference::getPointer<double>(dxMem),
							reference::getBetaValue(beta), reference::getPointer<double>(dyMem), reference::getPointer<double>(yMem), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
			avSize_t first_dim, last_dim;
			if (mode == AVOCADO_SOFTMAX_MODE_CHANNEL)
			{
				first_dim = reference::getTensor(xDesc).volumeWithoutLastDim();
				last_dim = reference::getTensor(xDesc).lastDim();
			}
			else
			{
				first_dim = reference::getTensor(xDesc).firstDim();
				last_dim = reference::getTensor(xDesc).volumeWithoutFirstDim();
			}

			switch (reference::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_softmax_forward<float16, float>(reference::getAlphaValue(alpha), reference::getPointer<float16>(xDesc),
							reference::getBetaValue(beta), reference::getPointer<float16>(yDesc), first_dim, last_dim);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_softmax_forward<bfloat16, float>(reference::getAlphaValue(alpha), reference::getPointer<bfloat16>(xDesc),
							reference::getBetaValue(beta), reference::getPointer<bfloat16>(yDesc), first_dim, last_dim);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_softmax_forward<float>(reference::getAlphaValue(alpha), reference::getPointer<float>(xDesc), reference::getBetaValue(beta),
							reference::getPointer<float>(yDesc), first_dim, last_dim);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_softmax_forward<double>(reference::getAlphaValue<double>(alpha), reference::getPointer<double>(xDesc),
							reference::getBetaValue<double>(beta), reference::getPointer<double>(yDesc), first_dim, last_dim);
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
			const avSize_t elements = reference::getTensor(yDesc).volume();
			switch (reference::getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_activation_backward<float>(AVOCADO_ACTIVATION_SOFTMAX, reference::getAlphaValue(alpha),
							reference::getPointer<float>(dxMem), reference::getBetaValue(beta), reference::getPointer<float>(dyMem),
							reference::getPointer<float>(yMem), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_activation_backward<double>(AVOCADO_ACTIVATION_SOFTMAX, reference::getAlphaValue(alpha),
							reference::getPointer<double>(dxMem), reference::getBetaValue(beta), reference::getPointer<double>(dyMem),
							reference::getPointer<double>(yMem), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

