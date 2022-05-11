/*
 * tensor_reduction.cpp
 *
 *  Created on: Nov 12, 2021
 *      Author: Maciej Kozarzewski
 */
#include <Avocado/reference_backend.h>

#include <Avocado/backend_descriptors.hpp>
#include "fp16.hpp"
#include "utils.hpp"
#include "activations.hpp"

#include <memory>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	template<typename T, typename U>
	void kernel_reduce_tensor(T *dst, const T *src, U alpha, U beta, BroadcastedDimensions dims, avReduceOp_t operation) noexcept
	{
		std::unique_ptr<U[]> workspace = std::make_unique<U[]>(dims.last);

		// first perform initialization of workspace tensor
		switch (operation)
		{
			case AVOCADO_REDUCE_ADD:
			case AVOCADO_REDUCE_AVG:
			case AVOCADO_REDUCE_NORM1:
			case AVOCADO_REDUCE_NORM2:
			case AVOCADO_REDUCE_AMAX:
				fill(workspace.get(), dims.last, zero<U>());
				break;
			case AVOCADO_REDUCE_MUL:
			case AVOCADO_REDUCE_MUL_NO_ZEROS:
				fill(workspace.get(), dims.last, one<U>());
				break;
			case AVOCADO_REDUCE_MIN:
				fill(workspace.get(), dims.last, max_value<U>::get());
				break;
			case AVOCADO_REDUCE_MAX:
				fill(workspace.get(), dims.last, -max_value<U>::get());
				break;
			default:
				break;
		}

		// now perform the main reduction
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
			{
				U value = static_cast<U>(src[i * dims.last + j]);
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
						if (value != zero<U>())
							workspace[j] *= value;
						break;
					default:
						break;
				}
			}

		// now perform final action, if necessary
		switch (operation)
		{
			default:
				break;
			case AVOCADO_REDUCE_AVG:
				for (av_int64 j = 0; j < dims.last; j++)
					workspace[j] = workspace[j] / static_cast<U>(dims.first);
				break;
			case AVOCADO_REDUCE_NORM2:
				for (av_int64 j = 0; j < dims.last; j++)
					workspace[j] = avocado::backend::sqrt(workspace[j]);
				break;
		}

		if (beta == zero<U>())
			clear(dst, dims.last);
		// now store result in dst
		for (av_int64 j = 0; j < dims.last; j++)
			dst[j] = alpha * workspace[j] + beta * static_cast<U>(dst[j]);
	}

	template<typename T>
	void kernel_reduce_logical_tensor(T *dst, const T *src, T alpha, T beta, BroadcastedDimensions dims, avReduceOp_t operation) noexcept
	{
		std::unique_ptr<T[]> workspace = std::make_unique<T[]>(dims.last);

		// first perform initialization of workspace tensor
		switch (operation)
		{
			case AVOCADO_REDUCE_LOGICAL_OR:
				fill(workspace.get(), dims.last, static_cast<T>(0));
				break;
			case AVOCADO_REDUCE_LOGICAL_AND:
				fill(workspace.get(), dims.last, static_cast<T>(-1));
				break;
			default:
				break;
		}

		// now perform the main reduction
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
			{
				T value = src[i * dims.last + j];
				switch (operation)
				{
					case AVOCADO_REDUCE_LOGICAL_OR:
						workspace[j] |= value;
						break;
					case AVOCADO_REDUCE_LOGICAL_AND:
						workspace[j] &= value;
						break;
					default:
						break;
				}
			}

		// now store result in dst
		for (av_int64 j = 0; j < dims.last; j++)
		{
			if (alpha == 0)
				workspace[j] = 0;
			if (beta == 0)
				dst[j] = workspace[j];
			else
			{
				switch (operation)
				{
					case AVOCADO_REDUCE_LOGICAL_OR:
						dst[j] |= workspace[j];
						break;
					case AVOCADO_REDUCE_LOGICAL_AND:
						dst[j] &= workspace[j];
						break;
					default:
						break;
				}
			}
		}
	}
}

namespace avocado
{
	namespace backend
	{
		using namespace BACKEND_NAMESPACE;

		avStatus_t refReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(aDesc), getTensor(cDesc));
			if (is_logical(operation))
			{
				switch (dataTypeSize(getTensor(aDesc).dtype()))
				{
					case 1:
						kernel_reduce_logical_tensor(getPointer<uint8_t>(cMem), getPointer<uint8_t>(aMem), getAlphaValue<uint8_t>(alpha),
								getBetaValue<uint8_t>(beta), dimensions, operation);
						break;
					case 2:
						kernel_reduce_logical_tensor(getPointer<uint16_t>(cMem), getPointer<uint16_t>(aMem), getAlphaValue<uint16_t>(alpha),
								getBetaValue<uint16_t>(beta), dimensions, operation);
						break;
					default:
					case 4:
						kernel_reduce_logical_tensor(getPointer<uint32_t>(cMem), getPointer<uint32_t>(aMem), getAlphaValue<uint32_t>(alpha),
								getBetaValue<uint32_t>(beta), dimensions, operation);
						break;
				}
			}
			else
			{
				switch (getTensor(cDesc).dtype())
				{
					case AVOCADO_DTYPE_FLOAT16:
						kernel_reduce_tensor(getPointer<float16>(cMem), getPointer<float16>(aMem), getAlphaValue(alpha), getBetaValue(beta),
								dimensions, operation);
						break;
					case AVOCADO_DTYPE_BFLOAT16:
						kernel_reduce_tensor(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getAlphaValue(alpha), getBetaValue(beta),
								dimensions, operation);
						break;
					case AVOCADO_DTYPE_FLOAT32:
						kernel_reduce_tensor(getPointer<float>(cMem), getPointer<float>(aMem), getAlphaValue(alpha), getBetaValue(beta), dimensions,
								operation);
						break;
					case AVOCADO_DTYPE_FLOAT64:
						kernel_reduce_tensor(getPointer<double>(cMem), getPointer<double>(aMem), getAlphaValue<double>(alpha),
								getBetaValue<double>(beta), dimensions, operation);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

