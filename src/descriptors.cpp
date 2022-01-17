/*
 * descriptors.cpp
 *
 *  Created on: Dec 7, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

namespace avocado
{
	namespace backend
	{
		avStatus_t refCreateMemoryDescriptor(avMemoryDescriptor_t *result, avSize_t sizeInBytes)
		{
			return reference::create<reference::MemoryDescriptor>(result, sizeInBytes);
		}
		avStatus_t refCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, avSize_t sizeInBytes, avSize_t offsetInBytes)
		{
			return reference::create<reference::MemoryDescriptor>(result, reference::getMemory(desc), sizeInBytes, offsetInBytes);
		}
		avStatus_t refDestroyMemoryDescriptor(avMemoryDescriptor_t desc)
		{
			return reference::destroy<reference::MemoryDescriptor>(desc);
		}
		avStatus_t refSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstOffset, avSize_t dstSize, const void *pattern,
				avSize_t patternSize)
		{
			if (reference::getPointer(dst) == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			if (pattern == nullptr)
			{
				std::memset(reference::getPointer<int8_t>(dst) + dstOffset, 0, dstSize);
				return AVOCADO_STATUS_SUCCESS;
			}

			if (dstSize % patternSize != 0 or dstOffset % patternSize != 0)
				return AVOCADO_STATUS_BAD_PARAM;
			for (avSize_t i = 0; i < dstSize; i += patternSize)
				std::memcpy(reference::getPointer<int8_t>(dst) + dstOffset + i, pattern, patternSize);
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstOffset, const avMemoryDescriptor_t src,
				avSize_t srcOffset, avSize_t count)
		{
			std::memcpy(reference::getPointer<int8_t>(dst) + dstOffset, reference::getPointer<int8_t>(src) + srcOffset, count);
			return AVOCADO_STATUS_SUCCESS;
		}
		void* refGetMemoryPointer(avMemoryDescriptor_t mem)
		{
			return reference::getPointer(mem);
		}

		avStatus_t refCreateContextDescriptor(avContextDescriptor_t *result)
		{
			return reference::create<reference::ContextDescriptor>(result);
		}
		avStatus_t refDestroyContextDescriptor(avContextDescriptor_t desc)
		{
			return reference::destroy<reference::ContextDescriptor>(desc);
		}
		avStatus_t refSynchronizeWithContext(avContextDescriptor_t context)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refIsContextReady(avContextDescriptor_t context, bool *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			result[0] = true;
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refCreateTensorDescriptor(avTensorDescriptor_t *result)
		{
			return reference::create<reference::TensorDescriptor>(result);
		}
		avStatus_t refDestroyTensorDescriptor(avTensorDescriptor_t desc)
		{
			return reference::destroy<reference::TensorDescriptor>(desc);
		}
		avStatus_t refSetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t dtype, int nbDims, const int dimensions[])
		{
			if (nbDims < 0 or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS or dimensions == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				reference::getTensor(desc).set(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t *dtype, int *nbDims, int dimensions[])
		{
			try
			{
				reference::getTensor(desc).get(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refCreateConvolutionDescriptor(avConvolutionDescriptor_t *result)
		{
			return reference::create<reference::ConvolutionDescriptor>(result);
		}
		avStatus_t refDestroyConvolutionDescriptor(avConvolutionDescriptor_t desc)
		{
			return reference::destroy<reference::ConvolutionDescriptor>(desc);
		}
		avStatus_t refSetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t mode, int nbDims, const int padding[],
				const int strides[], const int dilation[], int groups, const void *paddingValue)
		{
			try
			{
				reference::getConvolution(desc).set(mode, nbDims, strides, padding, dilation, groups, paddingValue);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t *mode, int *nbDims, int padding[], int strides[],
				int dilation[], int *groups, void *paddingValue)
		{
			try
			{
				reference::getConvolution(desc).get(mode, nbDims, strides, padding, dilation, groups, paddingValue);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refCreateOptimizerDescriptor(avOptimizerDescriptor_t *result)
		{
			return reference::create<reference::OptimizerDescriptor>(result);
		}
		avStatus_t refDestroyOptimizerDescriptor(avOptimizerDescriptor_t desc)
		{
			return reference::destroy<reference::OptimizerDescriptor>(desc);
		}
		avStatus_t refSetOptimizerSGD(avOptimizerDescriptor_t desc, double learningRate, bool useMomentum, bool useNesterov, double beta1)
		{
			try
			{
				reference::getOptimizer(desc).set_sgd(learningRate, useMomentum, useNesterov, beta1);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGetOptimizerSGD(avOptimizerDescriptor_t desc, double *learningRate, bool *useMomentum, bool *useNesterov, double *beta1)
		{
			try
			{
				reference::getOptimizer(desc).get_sgd(learningRate, useMomentum, useNesterov, beta1);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refSetOptimizerADAM(avOptimizerDescriptor_t desc, double learningRate, double beta1, double beta2)
		{
			try
			{
				reference::getOptimizer(desc).set_adam(learningRate, beta1, beta2);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGetOptimizerADAM(avOptimizerDescriptor_t desc, double *learningRate, double *beta1, double *beta2)
		{
			try
			{
				reference::getOptimizer(desc).get_adam(learningRate, beta1, beta2);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGetOptimizerType(avOptimizerDescriptor_t desc, avOptimizerType_t *type)
		{
			if (type == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				reference::getOptimizer(desc).get_type(type);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, int *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				reference::getOptimizer(desc).get_workspace_size(result, reference::getTensor(wDesc));
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

