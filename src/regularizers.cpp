/*
 * regularizers.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>
#include <ReferenceBackend/reference_backend.h>

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
		avStatus_t refRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem,
				const avTensorDescriptor_t weightDesc, const avMemoryDescriptor_t weightMem, const void *coefficient, const void *offset, void *loss)
		{
			const avSize_t elements = getTensor(gradientDesc).volume();
			switch (getTensor(gradientDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_regularizer_l2(getPointer<float>(gradientMem), getPointer<float>(weightMem), getScalarValue<float>(coefficient),
							getScalarValue<float>(offset), elements);
					if (loss != nullptr)
					{
						float l2_loss = kernel_loss_l2(getPointer<float>(weightMem), getScalarValue<float>(coefficient),
								getScalarValue<float>(offset), elements);
						setScalarValue(loss, l2_loss);
					}
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_regularizer_l2(getPointer<double>(gradientMem), getPointer<double>(weightMem), getScalarValue<double>(coefficient),
							getScalarValue<double>(offset), elements);
					if (loss != nullptr)
					{
						double l2_loss = kernel_loss_l2(getPointer<double>(weightMem), getScalarValue<double>(coefficient),
								getScalarValue<double>(offset), elements);
						setScalarValue(loss, l2_loss);
					}
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	}
/* namespace backend */
} /* namespace avocado */

