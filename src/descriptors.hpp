/*
 * descriptors.hpp
 *
 *  Created on: Dec 7, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef DESCRIPTORS_HPP_
#define DESCRIPTORS_HPP_

#include <avocado/backend/backend_defs.h>
#include <avocado/backend/backend_descriptors.hpp>

namespace avocado
{
	namespace backend
	{
		MemoryDescriptor& getMemory(avMemoryDescriptor_t desc);
		ContextDescriptor& getContext(avContextDescriptor_t desc);
		TensorDescriptor& getTensor(avTensorDescriptor_t desc);
		ConvolutionDescriptor& getConvolution(avConvolutionDescriptor_t desc);
		PoolingDescriptor& getPooling(avPoolingDescriptor_t desc);
		OptimizerDescriptor& getOptimizer(avOptimizerDescriptor_t desc);
		DropoutDescriptor& getDropout(avDropoutDescriptor_t desc);

		template<typename T = void>
		T* getPointer(avMemoryDescriptor_t desc)
		{
			try
			{
				return getMemory(desc).data<T>();
			} catch (std::exception &e)
			{
				return nullptr;
			}
		}
	}
/* namespace backend */
} /* namespace avocado */

#endif /* DESCRIPTORS_HPP_ */
