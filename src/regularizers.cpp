/*
 * regularizers.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/reference_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include "utils.hpp"

namespace
{
	using namespace avocado::backend;
	template<typename T>
	void kernel_regularizer_l2(T *gradient, const T *param, T coefficient, T offset, avSize_t elements)
	{
		for (avSize_t i = 0; i < elements; i++)
			gradient[i] += coefficient * (param[i] - offset);
	}
	template<typename T>
	T kernel_loss_l2(const T *param, T coefficient, T offset, avSize_t elements)
	{
		T result = zero<T>();
		for (avSize_t i = 0; i < elements; i++)
			result += square(param[i] - offset);
		return static_cast<T>(0.5) * coefficient * result;
	}
}

namespace avocado
{
	namespace backend
	{
//		avStatus_t refRegularizerL2(avContext_t context, avTensor_t gradient, const avTensor_t weight, const avScalar_t coefficient,
//				const avScalar_t offset, avScalar_t loss)
//		{
//			assert(context != nullptr);
//			assert(gradient != nullptr);
//			assert(weight != nullptr);
//			assert(coefficient != nullptr);
//			assert(offset != nullptr);
//			assert(same_shape(gradient, weight));
//			assert(same_type(gradient, weight));
//
//			switch (gradient->dtype)
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//				{
//					kernel_regularizer_l2(data<float>(gradient), data<float>(weight), getScalarValue<float>(coefficient),
//							getScalarValue<float>(offset), volume(weight));
//					if (loss != nullptr)
//					{
//						float l2_loss = kernel_loss_l2(data<float>(weight), getScalarValue<float>(coefficient), getScalarValue<float>(offset),
//								volume(weight));
//						setScalarValue(loss, l2_loss);
//					}
//					break;
//				}
//				case AVOCADO_DTYPE_FLOAT64:
//				{
//					kernel_regularizer_l2(data<double>(gradient), data<double>(weight), getScalarValue<double>(coefficient),
//							getScalarValue<double>(offset), volume(weight));
//					if (loss != nullptr)
//					{
//						double l2_loss = kernel_loss_l2(data<double>(weight), getScalarValue<double>(coefficient), getScalarValue<double>(offset),
//								volume(weight));
//						setScalarValue(loss, l2_loss);
//					}
//					break;
//				}
//				default:
//					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//			}
//			return AVOCADO_STATUS_SUCCESS;
//		}
	} /* namespace backend */
} /* namespace avocado */

