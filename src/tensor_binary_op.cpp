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
	void kernel_binary_op(T *dst, const T *src1, const T *src2, U alpha1, U alpha2, U beta, reference::BroadcastedDimensions dims,
			avBinaryOp_t operation)
			noexcept
	{
		if (beta == zero<U>())
			clear(dst, volume(dims));

		for (av_int64 i = 0; i < dims.first; i++)
		{
			for (av_int64 j = 0; j < dims.last; j++)
			{
				U value1 = alpha1 * static_cast<U>(src1[i * dims.last + j]);
				U value2;
				if (dims.first == 1)
					value2 = alpha2 * static_cast<U>(src2[i * dims.last + j]);
				else
				{
					if (dims.last == 1)
						value2 = alpha2 * static_cast<U>(src2[0]);
					else
						value2 = alpha2 * static_cast<U>(src2[j]);
				}
				U result = binary_op(operation, value1, value2);
				dst[i * dims.last + j] = result + beta * static_cast<U>(dst[i * dims.last + j]);
			}
		}
	}
	template<typename T>
	void kernel_binary_logical_op(T *dst, const T *src1, const T *src2, reference::BroadcastedDimensions dims, avBinaryOp_t operation)
	noexcept
	{
		clear(dst, volume(dims));
		for (av_int64 i = 0; i < dims.first; i++)
			for (av_int64 j = 0; j < dims.last; j++)
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
			reference::BroadcastedDimensions dimensions = getBroadcastDimensions(reference::getTensor(aDesc), reference::getTensor(bDesc));
			if (reference::is_logical(operation))
			{
				switch (reference::dataTypeSize(reference::getTensor(aDesc).dtype()))
				{
					case 1:
						kernel_binary_logical_op(reference::getPointer<uint8_t>(cMem), reference::getPointer<uint8_t>(aMem),
								reference::getPointer<uint8_t>(bMem), dimensions, operation);
						break;
					case 2:
						kernel_binary_logical_op(reference::getPointer<uint16_t>(cMem), reference::getPointer<uint16_t>(aMem),
								reference::getPointer<uint16_t>(bMem), dimensions, operation);
						break;
					default:
					case 4:
						kernel_binary_logical_op(reference::getPointer<uint32_t>(cMem), reference::getPointer<uint32_t>(aMem),
								reference::getPointer<uint32_t>(bMem), dimensions, operation);
						break;
				}
			}
			else
			{
				switch (reference::getTensor(cDesc).dtype())
				{
					case AVOCADO_DTYPE_FLOAT16:
						kernel_binary_op(reference::getPointer<float16>(cMem), reference::getPointer<float16>(aMem),
								reference::getPointer<float16>(bMem), reference::getAlphaValue(alpha1), reference::getAlphaValue(alpha2),
								reference::getBetaValue(beta), dimensions, operation);
						break;
					case AVOCADO_DTYPE_BFLOAT16:
						kernel_binary_op(reference::getPointer<bfloat16>(cMem), reference::getPointer<bfloat16>(aMem),
								reference::getPointer<bfloat16>(bMem), reference::getAlphaValue(alpha1), reference::getAlphaValue(alpha2),
								reference::getBetaValue(beta), dimensions, operation);
						break;
					case AVOCADO_DTYPE_FLOAT32:
						kernel_binary_op(reference::getPointer<float>(cMem), reference::getPointer<float>(aMem), reference::getPointer<float>(bMem),
								reference::getAlphaValue(alpha1), reference::getAlphaValue(alpha2), reference::getBetaValue(beta), dimensions,
								operation);
						break;
					case AVOCADO_DTYPE_FLOAT64:
						kernel_binary_op(reference::getPointer<double>(cMem), reference::getPointer<double>(aMem),
								reference::getPointer<double>(bMem), reference::getAlphaValue<double>(alpha1),
								reference::getAlphaValue<double>(alpha2), reference::getBetaValue<double>(beta), dimensions, operation);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

