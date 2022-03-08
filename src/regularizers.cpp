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
	void kernel_regularizer_l2(T *gradient, const T *param, T scale, T offset, av_int64 elements)
	{
		for (av_int64 i = 0; i < elements; i++)
			gradient[i] += scale * (param[i] - offset);
	}
	template<typename T>
	T kernel_loss_l2(const T *param, T scale, T offset, av_int64 elements)
	{
		T result = zero<T>();
		for (av_int64 i = 0; i < elements; i++)
			result += square(param[i] - offset);
		return static_cast<T>(0.5) * scale * result;
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t refRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem,
				const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const void *scale, const void *offset, void *loss)
		{
			const av_int64 elements = reference::getTensor(dwDesc).volume();
			switch (reference::getTensor(dwDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_regularizer_l2(reference::getPointer<float>(dwMem), reference::getPointer<float>(wMem),
							reference::getScalarValue<float>(scale), reference::getScalarValue<float>(offset), elements);
					if (loss != nullptr)
					{
						float l2_loss = kernel_loss_l2(reference::getPointer<float>(wMem), reference::getScalarValue<float>(scale),
								reference::getScalarValue<float>(offset), elements);
						reference::setScalarValue(loss, l2_loss);
					}
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_regularizer_l2(reference::getPointer<double>(dwMem), reference::getPointer<double>(wMem),
							reference::getScalarValue<double>(scale), reference::getScalarValue<double>(offset), elements);
					if (loss != nullptr)
					{
						double l2_loss = kernel_loss_l2(reference::getPointer<double>(wMem), reference::getScalarValue<double>(scale),
								reference::getScalarValue<double>(offset), elements);
						reference::setScalarValue(loss, l2_loss);
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

