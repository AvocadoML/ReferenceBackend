/*
 * tensor_unary_op.cpp
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
	T unary_op(avUnaryOp_t operation, T x) noexcept
	{
		switch (operation)
		{
			case AVOCADO_UNARY_OP_ABS:
				return avocado::backend::abs(x);
			case AVOCADO_UNARY_OP_CEIL:
				return avocado::backend::ceil(x);
			case AVOCADO_UNARY_OP_COS:
				return avocado::backend::cos(x);
			case AVOCADO_UNARY_OP_EXP:
				return avocado::backend::exp(x);
			case AVOCADO_UNARY_OP_FLOOR:
				return avocado::backend::floor(x);
			case AVOCADO_UNARY_OP_LN:
				return avocado::backend::log(x);
			case AVOCADO_UNARY_OP_NEG:
				return -x;
			case AVOCADO_UNARY_OP_RCP:
				return one<T>() / x;
			case AVOCADO_UNARY_OP_RSQRT:
				return one<T>() / avocado::backend::sqrt(x);
			case AVOCADO_UNARY_OP_SIN:
				return avocado::backend::sin(x);
			case AVOCADO_UNARY_OP_SQUARE:
				return avocado::backend::square(x);
			case AVOCADO_UNARY_OP_SQRT:
				return avocado::backend::sqrt(x);
			case AVOCADO_UNARY_OP_TAN:
				return avocado::backend::tan(x);
			case AVOCADO_UNARY_OP_LOGICAL_NOT:
				return LogicalNot<T>::value(x);
		}
		return zero<T>();
	}
	template<typename T, typename U>
	void kernel_unary_op(T *dst, const T *src, U alpha, U beta, avSize_t elements, avUnaryOp_t operation)
	noexcept
	{
		if (beta == zero<U>())
			clear(dst, elements);

		for (avSize_t i = 0; i < elements; i++)
		{
			T value = static_cast<T>(alpha * static_cast<U>(src[i]));
			T result = unary_op(operation, value);
			dst[i] = static_cast<U>(result) + beta * static_cast<U>(dst[i]);
		}
	}
	template<typename T>
	void kernel_unary_logical_op(T *dst, const T *src, avSize_t elements, avUnaryOp_t operation)
	noexcept
	{
		clear(dst, elements);
		for (avSize_t i = 0; i < elements; i++)
			dst[i] = ~(src[i]);
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const avSize_t elements = reference::getTensor(aDesc).volume();
			if (reference::is_logical(operation))
			{
				switch (reference::dataTypeSize(reference::getTensor(aDesc).dtype()))
				{
					case 1:
						kernel_unary_logical_op(reference::getPointer<uint8_t>(cMem), reference::getPointer<uint8_t>(aMem), elements, operation);
						break;
					case 2:
						kernel_unary_logical_op(reference::getPointer<uint16_t>(cMem), reference::getPointer<uint16_t>(aMem), elements, operation);
						break;
					default:
					case 4:
						kernel_unary_logical_op(reference::getPointer<uint32_t>(cMem), reference::getPointer<uint32_t>(aMem), elements, operation);
						break;
				}
			}
			else
			{
				switch (reference::getTensor(cDesc).dtype())
				{
					case AVOCADO_DTYPE_FLOAT16:
						kernel_unary_op(reference::getPointer<float16>(cMem), reference::getPointer<float16>(aMem), reference::getAlphaValue(alpha), reference::getBetaValue(beta), elements,
								operation);
						break;
					case AVOCADO_DTYPE_BFLOAT16:
						kernel_unary_op(reference::getPointer<bfloat16>(cMem), reference::getPointer<bfloat16>(aMem), reference::getAlphaValue(alpha), reference::getBetaValue(beta), elements,
								operation);
						break;
					case AVOCADO_DTYPE_FLOAT32:
						kernel_unary_op(reference::getPointer<float>(cMem), reference::getPointer<float>(aMem), reference::getAlphaValue(alpha), reference::getBetaValue(beta), elements,
								operation);
						break;
					case AVOCADO_DTYPE_FLOAT64:
						kernel_unary_op(reference::getPointer<double>(cMem), reference::getPointer<double>(aMem), reference::getAlphaValue<double>(alpha), reference::getBetaValue<double>(beta),
								elements, operation);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

