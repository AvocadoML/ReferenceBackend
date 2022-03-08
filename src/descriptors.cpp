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
		avStatus_t refCreateMemoryDescriptor(avMemoryDescriptor_t *result, av_int64 sizeInBytes)
		{
			return reference::create<reference::MemoryDescriptor>(result, sizeInBytes);
		}
		avStatus_t refCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, av_int64 sizeInBytes, av_int64 offsetInBytes)
		{
			return reference::create<reference::MemoryDescriptor>(result, reference::getMemory(desc), sizeInBytes, offsetInBytes);
		}
		avStatus_t refDestroyMemoryDescriptor(avMemoryDescriptor_t desc)
		{
			return reference::destroy<reference::MemoryDescriptor>(desc);
		}
		avStatus_t refSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, av_int64 dstOffset, av_int64 dstSize, const void *pattern,
				av_int64 patternSize)
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
			for (av_int64 i = 0; i < dstSize; i += patternSize)
				std::memcpy(reference::getPointer<int8_t>(dst) + dstOffset + i, pattern, patternSize);
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, av_int64 dstOffset, const avMemoryDescriptor_t src,
				av_int64 srcOffset, av_int64 count)
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
			if (nbDims < 0 or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS or (dimensions == nullptr and nbDims != 0))
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
				reference::getConvolution(desc).set(mode, nbDims, padding, strides, dilation, groups, paddingValue);
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
				reference::getConvolution(desc).get(mode, nbDims, padding, strides, dilation, groups, paddingValue);
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
		avStatus_t refSetOptimizerDescriptor(avOptimizerDescriptor_t desc, avOptimizerType_t type, double learningRate, const double coefficients[],
				const bool flags[])
		{
			try
			{
				reference::getOptimizer(desc).set(type, learningRate, coefficients, flags);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGetOptimizerDescriptor(avOptimizerDescriptor_t desc, avOptimizerType_t *type, double *learningRate, double coefficients[],
				bool flags[])
		{
			try
			{
				reference::getOptimizer(desc).get(type, learningRate, coefficients, flags);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, av_int64 *result)
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

