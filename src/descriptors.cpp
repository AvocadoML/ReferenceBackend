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
	template<typename T, typename ... Args>
	avStatus_t create(int *result, Args &&... args)
	{
		if (result == nullptr)
			return AVOCADO_STATUS_BAD_PARAM;
		try
		{
			result[0] = getPool<T>().create(std::forward<Args>(args)...);
		} catch (std::exception &e)
		{
			return AVOCADO_STATUS_INTERNAL_ERROR;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
	template<typename T>
	avStatus_t destroy(int desc)
	{
		try
		{
			getPool<T>().destroy(desc);
		} catch (std::exception &e)
		{
			return AVOCADO_STATUS_FREE_FAILED;
		}
		return AVOCADO_STATUS_SUCCESS;
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

		avStatus_t refCreateMemoryDescriptor(avMemoryDescriptor_t *result, avSize_t sizeInBytes)
		{
			return create<MemoryDescriptor>(result, sizeInBytes);
		}
		avStatus_t refCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, avSize_t sizeInBytes, avSize_t offsetInBytes)
		{
			return create<MemoryDescriptor>(result, getMemory(desc), sizeInBytes, offsetInBytes);
		}
		avStatus_t refDestroyMemoryDescriptor(avMemoryDescriptor_t desc)
		{
			return destroy<MemoryDescriptor>(desc);
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
			return create<ContextDescriptor>(result);
		}
		avStatus_t refDestroyContextDescriptor(avContextDescriptor_t desc)
		{
			return destroy<ContextDescriptor>(desc);
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
			return create<TensorDescriptor>(result);
		}
		avStatus_t refDestroyTensorDescriptor(avTensorDescriptor_t desc)
		{
			return destroy<TensorDescriptor>(desc);
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
			try
			{
				getTensor(desc).get(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refCreateConvolutionDescriptor(avConvolutionDescriptor_t *result)
		{
			return create<ConvolutionDescriptor>(result);
		}
		avStatus_t refDestroyConvolutionDescriptor(avConvolutionDescriptor_t desc)
		{
			return destroy<ConvolutionDescriptor>(desc);
		}
		avStatus_t refSetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t mode, int nbDims, const int padding[],
				const int strides[], const int dilation[], int groups, const void *paddingValue)
		{
			try
			{
				getConvolution(desc).set(mode, nbDims, strides, padding, dilation, groups, paddingValue);
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
				getConvolution(desc).get(mode, nbDims, strides, padding, dilation, groups, paddingValue);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t refCreateOptimizerDescriptor(avOptimizerDescriptor_t *result)
		{
			return create<OptimizerDescriptor>(result);
		}
		avStatus_t refDestroyOptimizerDescriptor(avOptimizerDescriptor_t desc)
		{
			return destroy<OptimizerDescriptor>(desc);
		}
		avStatus_t refSetOptimizerSGD(avOptimizerDescriptor_t desc, double learningRate, bool useMomentum, bool useNesterov, double beta1)
		{
			try
			{
				getOptimizer(desc).set_sgd(learningRate, useMomentum, useNesterov, beta1);
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
				getOptimizer(desc).get_sgd(learningRate, useMomentum, useNesterov, beta1);
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
				getOptimizer(desc).set_adam(learningRate, beta1, beta2);
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
				getOptimizer(desc).get_adam(learningRate, beta1, beta2);
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
				getOptimizer(desc).get_type(type);
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
				getOptimizer(desc).get_workspace_size(result, getTensor(wDesc));
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

