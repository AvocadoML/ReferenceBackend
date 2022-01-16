/*
 * tensor_binary_op.cpp
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

	template<typename T>
	T binary_op(avBinaryOp_t operation, T lhs, T rhs) noexcept
	{
		switch (operation)
		{
			case AVOCADO_BINARY_OP_ADD:
				return lhs + rhs;
			case AVOCADO_BINARY_OP_ADD_SQUARE:
				return lhs + square(rhs);
			case AVOCADO_BINARY_OP_SUB:
				return lhs - rhs;
			case AVOCADO_BINARY_OP_MUL:
				return lhs * rhs;
			case AVOCADO_BINARY_OP_DIV:
				return lhs / rhs;
			case AVOCADO_BINARY_OP_MOD:
				return mod(lhs, rhs);
			case AVOCADO_BINARY_OP_POW:
				return pow(lhs, rhs);
			case AVOCADO_BINARY_OP_MIN:
				return std::min(lhs, rhs);
			case AVOCADO_BINARY_OP_MAX:
				return std::max(lhs, rhs);
			case AVOCADO_BINARY_OP_COMPARE_EQ:
				return LogicalOp<T>::value_of(lhs == rhs);
			case AVOCADO_BINARY_OP_COMPARE_NEQ:
				return LogicalOp<T>::value_of(lhs != rhs);
			case AVOCADO_BINARY_OP_COMPARE_GT:
				return LogicalOp<T>::value_of(lhs > rhs);
			case AVOCADO_BINARY_OP_COMPARE_GE:
				return LogicalOp<T>::value_of(lhs >= rhs);
			case AVOCADO_BINARY_OP_COMPARE_LT:
				return LogicalOp<T>::value_of(lhs < rhs);
			case AVOCADO_BINARY_OP_COMPARE_LE:
				return LogicalOp<T>::value_of(lhs <= rhs);
			case AVOCADO_BINARY_OP_LOGICAL_AND:
				return LogicalAnd<T>::value(lhs, rhs);
			case AVOCADO_BINARY_OP_LOGICAL_OR:
				return LogicalOr<T>::value(lhs, rhs);
			case AVOCADO_BINARY_OP_LOGICAL_XOR:
				return LogicalXor<T>::value(lhs, rhs);
		}
		return zero<T>();
	}
	template<typename T, typename U>
	void kernel_binary_op(T *dst, const T *src1, const T *src2, U alpha1, U alpha2, U beta, BroadcastedDimensions dims, avBinaryOp_t operation)
	noexcept
	{
		if (beta == zero<U>())
			clear(dst, volume(dims));

		for (avSize_t i = 0; i < dims.first; i++)
		{
			for (avSize_t j = 0; j < dims.last; j++)
			{
				T value1 = static_cast<T>(alpha1 * static_cast<U>(src1[i * dims.last + j]));
				T value2 = static_cast<T>(alpha2 * static_cast<U>(src2[i * dims.last + j]));
				T result = binary_op(operation, value1, value2);
				dst[i * dims.last + j] = static_cast<U>(result) + beta * static_cast<U>(dst[i * dims.last + j]);
			}
		}
	}
	template<typename T>
	void kernel_binary_logical_op(T *dst, const T *src1, const T *src2, BroadcastedDimensions dims, avBinaryOp_t operation)
	noexcept
	{
		clear(dst, volume(dims));
		for (avSize_t i = 0; i < dims.first; i++)
			for (avSize_t j = 0; j < dims.last; j++)
			{
				T lhs = src1[i * dims.last + j];
				T rhs = src2[i * dims.last + j];
				switch (operation)
				{
					case AVOCADO_BINARY_OP_LOGICAL_AND:
						dst[i * dims.last + j] = lhs & rhs;
						break;
					case AVOCADO_BINARY_OP_LOGICAL_OR:
						dst[i * dims.last + j] = lhs | rhs;
						break;
					case AVOCADO_BINARY_OP_LOGICAL_XOR:
						dst[i * dims.last + j] = lhs ^ rhs;
						break;
					default:
						break;
				}
			}
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refBinaryOp(avContextDescriptor_t context, avBinaryOp_t operation, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(aDesc), getTensor(bDesc));
			if (is_logical(operation))
			{
				switch (dataTypeSize(getTensor(aDesc).dtype()))
				{
					case 1:
						kernel_binary_logical_op(getPointer<uint8_t>(cMem), getPointer<uint8_t>(aMem), getPointer<uint8_t>(bMem), dimensions,
								operation);
						break;
					case 2:
						kernel_binary_logical_op(getPointer<uint16_t>(cMem), getPointer<uint16_t>(aMem), getPointer<uint16_t>(bMem), dimensions,
								operation);
						break;
					default:
					case 4:
						kernel_binary_logical_op(getPointer<uint32_t>(cMem), getPointer<uint32_t>(aMem), getPointer<uint32_t>(bMem), dimensions,
								operation);
						break;
				}
			}
			else
			{
				switch (getTensor(cDesc).dtype())
				{
					case AVOCADO_DTYPE_FLOAT16:
						kernel_binary_op(getPointer<float16>(cMem), getPointer<float16>(aMem), getPointer<float16>(bMem), getAlphaValue(alpha1),
								getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
						break;
					case AVOCADO_DTYPE_BFLOAT16:
						kernel_binary_op(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getPointer<bfloat16>(bMem), getAlphaValue(alpha1),
								getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
						break;
					case AVOCADO_DTYPE_FLOAT32:
						kernel_binary_op(getPointer<float>(cMem), getPointer<float>(aMem), getPointer<float>(bMem), getAlphaValue(alpha1),
								getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
						break;
					case AVOCADO_DTYPE_FLOAT64:
						kernel_binary_op(getPointer<double>(cMem), getPointer<double>(aMem), getPointer<double>(bMem), getAlphaValue<double>(alpha1),
								getAlphaValue<double>(alpha2), getBetaValue<double>(beta), dimensions, operation);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

