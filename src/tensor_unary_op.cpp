/*
 * tensor_unary_op.cpp
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
	void kernel_unary_op(T *dst, const T *src, U alpha, U beta, av_int64 elements, avUnaryOp_t operation) noexcept
	{
		if (beta == zero<U>())
			clear(dst, elements);

		for (av_int64 i = 0; i < elements; i++)
		{
			U value = alpha * static_cast<U>(src[i]);
			U result = unary_op(operation, value);
			dst[i] = result + beta * static_cast<U>(dst[i]);
		}
	}
	template<typename T>
	void kernel_unary_logical_op(T *dst, const T *src, av_int64 elements, avUnaryOp_t operation) noexcept
	{
		clear(dst, elements);
		for (av_int64 i = 0; i < elements; i++)
			dst[i] = ~(src[i]);
	}
}

namespace avocado
{
	namespace backend
	{
		using namespace BACKEND_NAMESPACE;

		avStatus_t refUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			const av_int64 elements = getTensor(aDesc).volume();
			if (is_logical(operation))
			{
				switch (dataTypeSize(getTensor(aDesc).dtype()))
				{
					case 1:
						kernel_unary_logical_op(getPointer<uint8_t>(cMem), getPointer<uint8_t>(aMem), elements, operation);
						break;
					case 2:
						kernel_unary_logical_op(getPointer<uint16_t>(cMem), getPointer<uint16_t>(aMem), elements, operation);
						break;
					default:
					case 4:
					{
						const av_int64 tmp = elements * dataTypeSize(getTensor(aDesc).dtype()) / 4;
						kernel_unary_logical_op(getPointer<uint32_t>(cMem), getPointer<uint32_t>(aMem), tmp, operation);
						break;
					}
				}
			}
			else
			{
				switch (getTensor(cDesc).dtype())
				{
					case AVOCADO_DTYPE_FLOAT16:
						kernel_unary_op(getPointer<float16>(cMem), getPointer<float16>(aMem), getAlphaValue(alpha), getBetaValue(beta), elements,
								operation);
						break;
					case AVOCADO_DTYPE_BFLOAT16:
						kernel_unary_op(getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getAlphaValue(alpha), getBetaValue(beta), elements,
								operation);
						break;
					case AVOCADO_DTYPE_FLOAT32:
						kernel_unary_op(getPointer<float>(cMem), getPointer<float>(aMem), getAlphaValue(alpha), getBetaValue(beta), elements,
								operation);
						break;
					case AVOCADO_DTYPE_FLOAT64:
						kernel_unary_op(getPointer<double>(cMem), getPointer<double>(aMem), getAlphaValue<double>(alpha), getBetaValue<double>(beta),
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

