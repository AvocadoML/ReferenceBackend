/*
 * descriptors.cpp
 *
 *  Created on: Dec 7, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include "descriptors.hpp"

namespace
{
	using namespace avocado::backend;

	template<class T>
	DescriptorPool<T>& getPool()
	{
		thread_local DescriptorPool<T> result;
		return result;
	}
	template<>
	DescriptorPool<ContextDescriptor>& getPool()
	{
		thread_local DescriptorPool<ContextDescriptor> result = []()
		{
			DescriptorPool<ContextDescriptor> tmp;
			tmp.create(); // reserve descriptor 0 for default context
			return tmp;
		}();
		return result;
	}
}

namespace avocado
{
	namespace backend
	{
		MemoryDescriptor& getMemory(avMemoryDescriptor_t desc)
		{
			return getPool<MemoryDescriptor>().get(desc);
		}
		ContextDescriptor& getContext(avContextDescriptor_t desc)
		{
			return getPool<ContextDescriptor>().get(desc);
		}
		TensorDescriptor& getTensor(avTensorDescriptor_t desc)
		{
			return getPool<TensorDescriptor>().get(desc);
		}
		ConvolutionDescriptor& getConvolution(avConvolutionDescriptor_t desc)
		{
			return getPool<ConvolutionDescriptor>().get(desc);
		}
		PoolingDescriptor& getPooling(avPoolingDescriptor_t desc)
		{
			return getPool<PoolingDescriptor>().get(desc);
		}
		OptimizerDescriptor& getOptimizer(avOptimizerDescriptor_t desc)
		{
			return getPool<OptimizerDescriptor>().get(desc);
		}
		DropoutDescriptor& getDropout(avDropoutDescriptor_t desc)
		{
			return getPool<DropoutDescriptor>().get(desc);
		}

		avStatus_t refCreateMemoryDescriptor(avMemoryDescriptor_t *result, avSize_t count)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				result[0] = getPool<MemoryDescriptor>().create(count);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, avSize_t offset)
		{
			try
			{
				result[0] = getPool<MemoryDescriptor>().create(getMemory(desc), offset);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refDestroyMemoryDescriptor(avMemoryDescriptor_t desc)
		{
			try
			{
				getPool<MemoryDescriptor>().destroy(desc);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_FREE_FAILED;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstSize, const void *pattern, avSize_t patternSize)
		{
			if (getPointer(dst) == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			if (pattern == nullptr)
			{
				std::memset(getPointer(dst), 0, dstSize);
				return AVOCADO_STATUS_SUCCESS;
			}

			if (dstSize % patternSize != 0)
				return AVOCADO_STATUS_BAD_PARAM;
			for (avSize_t i = 0; i < dstSize; i += patternSize)
				std::memcpy(getPointer<int8_t>(dst) + i, pattern, patternSize);
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, const avMemoryDescriptor_t src, avSize_t count)
		{
			std::memcpy(getPointer(dst), getPointer(src), count);
			return AVOCADO_STATUS_SUCCESS;
		}
		void* refGetMemoryPointer(avMemoryDescriptor_t mem)
		{
			return getPointer(mem);
		}

		avStatus_t refCreateContextDescriptor(avContextDescriptor_t *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				result[0] = getPool<ContextDescriptor>().create();
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refDestroyContextDescriptor(avContextDescriptor_t context)
		{
			if (context == 0)
				return AVOCADO_STATUS_BAD_PARAM; // cannot destroy default context
			try
			{
				getPool<ContextDescriptor>().destroy(context);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
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
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				result[0] = getPool<TensorDescriptor>().create();
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refDestroyTensorDescriptor(avTensorDescriptor_t desc)
		{
			try
			{
				getPool<TensorDescriptor>().destroy(desc);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refSetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t dtype, int nbDims, const int dimensions[])
		{
			if (nbDims < 0 or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS or dimensions == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				getTensor(desc).set(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t refGetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t *dtype, int *nbDims, int dimensions[])
		{
			if (dtype == nullptr or nbDims == nullptr or dimensions == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				getTensor(desc).get(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

