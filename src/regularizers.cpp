/*
 * regularizers.cpp
 *
 *  Created on: Nov 25, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/reference_backend.h>

#include <Avocado/backend_descriptors.hpp>
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
		using namespace BACKEND_NAMESPACE;

		avStatus_t refRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem,
				const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const void *scale, const void *offset, void *loss)
		{
			const av_int64 elements = getTensor(dwDesc).volume();
			switch (getTensor(dwDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_regularizer_l2(getPointer<float>(dwMem), getPointer<float>(wMem), getScalarValue<float>(scale),
							getScalarValue<float>(offset), elements);
					if (loss != nullptr)
					{
						float l2_loss = kernel_loss_l2(getPointer<float>(wMem), getScalarValue<float>(scale), getScalarValue<float>(offset),
								elements);
						setScalarValue(loss, l2_loss);
					}
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_regularizer_l2(getPointer<double>(dwMem), getPointer<double>(wMem), getScalarValue<double>(scale),
							getScalarValue<double>(offset), elements);
					if (loss != nullptr)
					{
						double l2_loss = kernel_loss_l2(getPointer<double>(wMem), getScalarValue<double>(scale), getScalarValue<double>(offset),
								elements);
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

