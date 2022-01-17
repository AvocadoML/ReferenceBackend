/*
 * tensor_reduction.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */
#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

#include "fp16.hpp"
#include "utils.hpp"
#include "activations.hpp"

#include <memory>

namespace
{
	using namespace avocado::backend;

	template<typename T, typename U>
	void kernel_reduce_tensor(T *dst, const T *src, U alpha, U beta, reference::BroadcastedDimensions dims, avReduceOp_t operation) noexcept
	{
		std::unique_ptr<T[]> workspace = std::make_unique<T[]>(dims.last);

		// first perform initialization of workspace tensor
		for (avSize_t j = 0; j < dims.last; j++)
		{
			switch (operation)
			{
				case AVOCADO_REDUCE_ADD:
				case AVOCADO_REDUCE_AVG:
				case AVOCADO_REDUCE_NORM1:
				case AVOCADO_REDUCE_NORM2:
				case AVOCADO_REDUCE_AMAX:
					workspace[j] = zero<T>();
					break;
				case AVOCADO_REDUCE_MUL:
				case AVOCADO_REDUCE_MUL_NO_ZEROS:
					workspace[j] = one<T>();
					break;
				case AVOCADO_REDUCE_MIN:
					workspace[j] = max_value<T>::get();
					break;
				case AVOCADO_REDUCE_MAX:
					workspace[j] = -max_value<T>::get();
					break;
			}
		}

		// now perform the main reduction
		for (avSize_t i = 0; i < dims.first; i++)
		{
			for (avSize_t j = 0; j < dims.last; j++)
			{
				T value = src[i * dims.last + j];
				switch (operation)
				{
					case AVOCADO_REDUCE_ADD:
						workspace[j] += value;
						break;
					case AVOCADO_REDUCE_MUL:
						workspace[j] *= value;
						break;
					case AVOCADO_REDUCE_MIN:
						workspace[j] = std::min(workspace[j], value);
						break;
					case AVOCADO_REDUCE_MAX:
						workspace[j] = std::max(workspace[j], value);
						break;
					case AVOCADO_REDUCE_AMAX:
						workspace[j] = std::max(workspace[j], avocado::backend::abs(value));
						break;
					case AVOCADO_REDUCE_AVG:
						workspace[j] += value;
						break;
					case AVOCADO_REDUCE_NORM1:
						workspace[j] += avocado::backend::abs(value);
						break;
					case AVOCADO_REDUCE_NORM2:
						workspace[j] += square(value);
						break;
					case AVOCADO_REDUCE_MUL_NO_ZEROS:
						if (value != zero<T>())
							workspace[j] *= value;
						break;
				}
			}
		}

		// now perform final action, if necessary
		switch (operation)
		{
			default:
				break;
			case AVOCADO_REDUCE_AVG:
				for (avSize_t j = 0; j < dims.last; j++)
					workspace[j] = static_cast<U>(workspace[j]) / dims.first;
				break;
			case AVOCADO_REDUCE_NORM2:
				for (avSize_t j = 0; j < dims.last; j++)
					workspace[j] = avocado::backend::sqrt(static_cast<U>(workspace[j]));
				break;
		}

		if (beta == zero<U>())
			clear(dst, dims.last);
		// now store result in dst
		for (avSize_t j = 0; j < dims.last; j++)
			dst[j] = alpha * static_cast<U>(workspace[j]) + beta * static_cast<U>(dst[j]);
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			reference::BroadcastedDimensions dimensions = getBroadcastDimensions(reference::getTensor(aDesc), reference::getTensor(cDesc));
			switch (reference::getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					kernel_reduce_tensor(reference::getPointer<float16>(cMem), reference::getPointer<float16>(aMem), reference::getAlphaValue(alpha),
							reference::getBetaValue(beta), dimensions, operation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_reduce_tensor(reference::getPointer<bfloat16>(cMem), reference::getPointer<bfloat16>(aMem),
							reference::getAlphaValue(alpha), reference::getBetaValue(beta), dimensions, operation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_reduce_tensor(reference::getPointer<float>(cMem), reference::getPointer<float>(aMem), reference::getAlphaValue(alpha),
							reference::getBetaValue(beta), dimensions, operation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_reduce_tensor(reference::getPointer<double>(cMem), reference::getPointer<double>(aMem),
							reference::getAlphaValue<double>(alpha), reference::getBetaValue<double>(beta), dimensions, operation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

